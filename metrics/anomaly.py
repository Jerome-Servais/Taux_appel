"""
metrics/anomaly.py — Écarts atypiques et ruptures de tendance.

Chaque groupe est comparé à son propre historique, jamais aux autres groupes : un
profil stable mais différent des autres n'est pas une anomalie. Deux règles
garantissent qu'une alerte aurait pu être levée au moment des faits :

- la référence n'utilise que des périodes passées ;
- elle repose sur la médiane et l'écart absolu médian (MAD), qu'un pic isolé ou
  le début d'une rupture ne déforment pas.

En granularité journalière, la référence compare chaque jour aux mêmes jours de
semaine, car un lundi ne ressemble pas à un dimanche.

Deux détections complémentaires :
- z-score robuste : repère un écart ponctuel fort ;
- CUSUM : additionne de petits écarts pour repérer une rupture durable.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from config import BASELINE_SETTINGS, CUSUM_DEFAULTS_PER_DIM, DEFAULT_FREQ_CODE

MIN_EFFECTIF_DEFAULT = 20
MAD_TO_STD = 1.4826
Z_CLIP = 4.0


def _mad(values: np.ndarray) -> float:
    return float(np.nanmedian(np.abs(values - np.nanmedian(values))))


def _same_season_history(series: pd.Series, season: int) -> list[pd.Series]:
    """Découpe une série en sous-séries de même phase (ex. tous les lundis)."""
    positions = np.arange(len(series)) % season
    return [series[positions == phase] for phase in range(season)]


def add_deviation_from_own_history(
    df: pd.DataFrame,
    value_col: str,
    group_col: str = "profil_partenaire",
    freq_code: str = DEFAULT_FREQ_CODE,
    window: int | None = None,
    min_effectif: int = MIN_EFFECTIF_DEFAULT,
    n_col: str = "n_traitements",
) -> pd.DataFrame:
    """Ajoute hist_ref, hist_scale, zscore et pct_deviation par `group_col`.

    hist_ref est la médiane des périodes passées et hist_scale leur dispersion,
    avec un plancher égal au bruit d'un comptage : sans lui, quelques périodes très
    régulières donneraient une dispersion quasi nulle et des alertes injustifiées.
    Les lignes sous `min_effectif` n'ont pas de score.
    """
    settings = BASELINE_SETTINGS[freq_code]
    window = window or settings["window"]
    min_periods = min(settings["min_periods"], window)
    season = settings["season"]

    df = df.sort_values([group_col, "period"]).copy()
    df["hist_ref"] = np.nan
    df["hist_scale"] = np.nan

    for _, group in df.groupby(group_col, sort=False):
        counts = group[n_col] if n_col in group.columns else None
        for part in _same_season_history(group[value_col].astype(float), season):
            past = part.shift(1).rolling(window, min_periods=min_periods)
            reference = past.median()
            spread = past.apply(_mad, raw=True) * MAD_TO_STD

            noise = 0.0
            if counts is not None:
                past_counts = counts.loc[part.index].shift(1).rolling(window, min_periods=min_periods).median()
                noise = reference.abs() / np.sqrt(past_counts.clip(lower=1))

            df.loc[part.index, "hist_ref"] = reference
            df.loc[part.index, "hist_scale"] = np.maximum(spread, noise)

    reliable = df[n_col] >= min_effectif if n_col in df.columns else pd.Series(True, index=df.index)
    df["zscore"] = np.where(
        (df["hist_scale"] > 0) & reliable, (df[value_col] - df["hist_ref"]) / df["hist_scale"], np.nan
    )
    df["pct_deviation"] = np.where(
        (df["hist_ref"].abs() > 0) & reliable,
        (df[value_col] - df["hist_ref"]) / df["hist_ref"].abs() * 100,
        np.nan,
    )
    return df


def flag_alerts(
    df: pd.DataFrame,
    threshold_type: str = "zscore",
    threshold_value: float = 3.0,
    min_effectif: int = 0,
    n_col: str = "n_traitements",
) -> pd.DataFrame:
    """Marque is_alert selon le seuil choisi (threshold_type : 'zscore' ou 'pct').

    Aucune alerte sur une ligne jugée non fiable (is_reliable=False) ou dont
    l'effectif est inférieur à `min_effectif`.
    """
    df = df.copy()
    if threshold_type == "zscore":
        raw_alert = df["zscore"].abs() >= threshold_value
    elif threshold_type == "pct":
        raw_alert = df["pct_deviation"].abs() >= threshold_value
    else:
        raise ValueError("threshold_type doit être 'zscore' ou 'pct'")

    if min_effectif > 0 and n_col in df.columns:
        raw_alert = raw_alert & (df[n_col] >= min_effectif)
    if "is_reliable" in df.columns:
        raw_alert = raw_alert & df["is_reliable"]
    df["is_alert"] = raw_alert
    return df


def _online_cusum(
    counts: np.ndarray, k: float, h: float, season: int, window: int, min_periods: int
) -> np.ndarray:
    """CUSUM bilatéral sur le log des comptages, sans regarder le futur.

    Retourne +1 ou -1 aux périodes de rupture, 0 ailleurs. Après une rupture, la
    référence repart de zéro pour apprendre le nouveau niveau, ce qui évite de
    signaler la même rupture à chaque période suivante.
    """
    values = np.log1p(counts)
    signals = np.zeros(len(values), dtype=int)
    history: dict[int, list[float]] = {phase: [] for phase in range(season)}
    cusum_up = cusum_down = 0.0

    for t, value in enumerate(values):
        phase = t % season
        past = np.array(history[phase][-window:])
        if len(past) < min_periods:
            history[phase].append(value)
            continue

        reference = np.median(past)
        noise = 1 / np.sqrt(np.expm1(reference) + 1)
        scale = max(MAD_TO_STD * _mad(past), noise)
        z = np.clip((value - reference) / scale, -Z_CLIP, Z_CLIP)  # un pic seul ne suffit pas

        cusum_up = max(0.0, cusum_up + z - k)
        cusum_down = min(0.0, cusum_down + z + k)
        if cusum_up > h or cusum_down < -h:
            signals[t] = 1 if cusum_up > h else -1
            cusum_up = cusum_down = 0.0
            history = {phase: [] for phase in range(season)}
        history[phase].append(value)
    return signals


def add_cusum_breaks(
    df: pd.DataFrame,
    value_col: str,
    group_col: str = "profil_partenaire",
    freq_code: str = DEFAULT_FREQ_CODE,
    k: float | None = None,
    h: float | None = None,
    min_effectif: int = MIN_EFFECTIF_DEFAULT,
    n_col: str = "n_traitements",
) -> pd.DataFrame:
    """Ajoute `cusum_break` et `cusum_direction` (+1 hausse, -1 baisse) par groupe.

    `value_col` doit être un comptage. `cusum_break` n'est vrai qu'à la période où
    la rupture devient détectable. k est la tolérance et h le seuil de déclenchement,
    tous deux en écarts-types.
    """
    defaults = CUSUM_DEFAULTS_PER_DIM.get(group_col, CUSUM_DEFAULTS_PER_DIM["profil_partenaire"])
    k = defaults["k"] if k is None else k
    h = defaults["h"] if h is None else h
    settings = BASELINE_SETTINGS[freq_code]

    df = df.sort_values([group_col, "period"]).copy()
    df["cusum_direction"] = 0
    for _, group in df.groupby(group_col, sort=False):
        df.loc[group.index, "cusum_direction"] = _online_cusum(
            group[value_col].to_numpy(dtype=float), k, h,
            settings["season"], settings["window"], settings["min_periods"],
        )

    if min_effectif > 0 and n_col in df.columns:
        df.loc[df[n_col] < min_effectif, "cusum_direction"] = 0
    df["cusum_break"] = df["cusum_direction"] != 0
    return df


def period_over_period_table(df: pd.DataFrame, period_a: pd.Timestamp, period_b: pd.Timestamp, value_col: str) -> pd.DataFrame:
    """Comparaison de deux périodes, profil par profil."""
    a = df[df["period"] == period_a][["profil_partenaire", value_col]].rename(columns={value_col: "value_a"})
    b = df[df["period"] == period_b][["profil_partenaire", value_col]].rename(columns={value_col: "value_b"})
    merged = a.merge(b, on="profil_partenaire", how="outer")
    merged["delta_abs"] = merged["value_b"] - merged["value_a"]
    merged["delta_pct"] = np.where(
        merged["value_a"].abs() > 0, merged["delta_abs"] / merged["value_a"].abs() * 100, np.nan
    )
    return merged.sort_values("delta_abs", ascending=False)
