"""
metrics/anomaly.py — Écarts atypiques et ruptures de tendance.

Principe clé (section 2.4 du cadrage) : la comparaison se fait par rapport à
l'HISTORIQUE PROPRE du profil, jamais par rapport aux autres profils. Un profil
au comportement stable mais différent des autres n'est pas une anomalie ; un
profil qui décroche par rapport à sa propre trajectoire l'est.

Deux détections complémentaires :
- z-score / écart en % vs historique propre (fenêtre glissante ou expanding)
- CUSUM standardisé, pour repérer une rupture de tendance soutenue plutôt
  qu'un pic ponctuel (section 2.3 du cadrage)
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def add_deviation_from_own_history(
    df: pd.DataFrame, value_col: str, window: int | None = 6
) -> pd.DataFrame:
    """Ajoute hist_mean, hist_std, zscore, pct_deviation par profil.

    hist_mean/hist_std sont calculés sur les périodes STRICTEMENT antérieures
    (shift(1)) du même profil, en fenêtre glissante de `window` périodes si
    fourni, sinon en expanding sur tout l'historique disponible.
    """
    df = df.sort_values(["profil_partenaire", "period"]).copy()

    def _hist_mean(s: pd.Series) -> pd.Series:
        shifted = s.shift(1)
        return shifted.rolling(window, min_periods=3).mean() if window else shifted.expanding(min_periods=3).mean()

    def _hist_std(s: pd.Series) -> pd.Series:
        shifted = s.shift(1)
        return shifted.rolling(window, min_periods=3).std() if window else shifted.expanding(min_periods=3).std()

    df["hist_mean"] = df.groupby("profil_partenaire")[value_col].transform(_hist_mean)
    df["hist_std"] = df.groupby("profil_partenaire")[value_col].transform(_hist_std)

    df["zscore"] = np.where(
        df["hist_std"] > 0, (df[value_col] - df["hist_mean"]) / df["hist_std"], np.nan
    )
    df["pct_deviation"] = np.where(
        df["hist_mean"].abs() > 0,
        (df[value_col] - df["hist_mean"]) / df["hist_mean"].abs() * 100,
        np.nan,
    )
    return df


def flag_alerts(
    df: pd.DataFrame, threshold_type: str = "zscore", threshold_value: float = 2.0
) -> pd.DataFrame:
    """Marque is_alert selon le type de seuil choisi dans l'interface.

    threshold_type: 'zscore' ou 'pct'
    Une alerte n'est jamais levée sur une ligne jugée non fiable (is_reliable=False,
    cf. garde-fou volume minimum, metrics/lift.py::apply_min_volume_mask).
    """
    df = df.copy()
    if threshold_type == "zscore":
        raw_alert = df["zscore"].abs() >= threshold_value
    elif threshold_type == "pct":
        raw_alert = df["pct_deviation"].abs() >= threshold_value
    else:
        raise ValueError("threshold_type doit être 'zscore' ou 'pct'")

    raw_alert = raw_alert.fillna(False)
    if "is_reliable" in df.columns:
        df["is_alert"] = raw_alert & df["is_reliable"]
    else:
        df["is_alert"] = raw_alert
    return df


def _cusum_breaks_1d(x: np.ndarray, k: float, h: float) -> np.ndarray:
    """CUSUM standardisé (référence = moyenne/écart-type de la série fournie).

    k : slack (en écarts-types) toléré autour de la référence avant d'accumuler
    h : seuil de déclenchement (en écarts-types cumulés) au-delà duquel on
        considère qu'il y a rupture
    """
    mean = np.nanmean(x)
    std = np.nanstd(x)
    if not std or np.isnan(std):
        return np.zeros(len(x), dtype=bool)

    z = (x - mean) / std
    c_pos = np.zeros(len(z))
    c_neg = np.zeros(len(z))
    for i in range(1, len(z)):
        c_pos[i] = max(0.0, c_pos[i - 1] + z[i] - k)
        c_neg[i] = min(0.0, c_neg[i - 1] + z[i] + k)
    return (c_pos > h) | (c_neg < -h)


def add_cusum_breaks(df: pd.DataFrame, value_col: str, k: float = 0.5, h: float = 5.0) -> pd.DataFrame:
    """Ajoute une colonne booléenne `cusum_break` par profil (détection de rupture,
    section 2.3 : 'un profil stable qui décroche')."""
    df = df.sort_values(["profil_partenaire", "period"]).copy()
    df["cusum_break"] = False
    for _, group in df.groupby("profil_partenaire"):
        x = group[value_col].to_numpy(dtype=float)
        breaks = _cusum_breaks_1d(x, k=k, h=h)
        df.loc[group.index, "cusum_break"] = breaks
    return df


def period_over_period_table(df: pd.DataFrame, period_a: pd.Timestamp, period_b: pd.Timestamp, value_col: str) -> pd.DataFrame:
    """Comparaison période sur période (section 2.3), profil par profil."""
    a = df[df["period"] == period_a][["profil_partenaire", value_col]].rename(columns={value_col: "value_a"})
    b = df[df["period"] == period_b][["profil_partenaire", value_col]].rename(columns={value_col: "value_b"})
    merged = a.merge(b, on="profil_partenaire", how="outer")
    merged["delta_abs"] = merged["value_b"] - merged["value_a"]
    merged["delta_pct"] = np.where(
        merged["value_a"].abs() > 0, merged["delta_abs"] / merged["value_a"].abs() * 100, np.nan
    )
    return merged.sort_values("delta_abs", ascending=False)
