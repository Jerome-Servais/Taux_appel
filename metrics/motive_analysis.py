"""
metrics/motive_analysis.py — Détection et qualification par motif / sous-motif.

Un changement de motif peut laisser le volume total intact : si les réclamations
remplacent d'autres demandes, aucune alerte de volume ne se déclenche. Le module
surveille donc la PART de chaque motif, puis qualifie l'écart :

1. `detect_motif_share_shifts` : CUSUM sur la part de chaque motif, par profil.
2. `motif_distribution_drilldown` : compare une période à une référence antérieure
   au début de l'anomalie. Une référence incluant l'anomalie masquerait l'écart.
3. `chisq_independence_test` : teste l'écart, mesure sa taille et nomme la catégorie
   qui en est la cause.

Tous les calculs utilisent des volumes (`n_traitements`), pas des nombres de lignes.
"""

from __future__ import annotations

import time
from typing import Literal

import numpy as np
import pandas as pd
from scipy.stats import chi2_contingency, fisher_exact

from config import SHARE_CUSUM_DEFAULTS

MIN_EXPECTED = 5.0
RARE_LABEL = "Autres catégories (regroupées)"


# ---------------------------------------------------------------------------
# Test du chi²
# ---------------------------------------------------------------------------

def _merge_rare_categories(table: pd.DataFrame) -> pd.DataFrame:
    """Fusionne les plus petites catégories tant qu'un effectif attendu reste sous 5.
    La validité du chi² repose sur cette condition. S'il ne reste que deux lignes, le
    test exact de Fisher prend le relais."""
    table = table.copy()
    while len(table) > 2:
        row_totals = table.sum(axis=1).sort_values()
        expected_min = row_totals.iloc[0] * table.sum(axis=0).min() / table.to_numpy().sum()
        if expected_min >= MIN_EXPECTED:
            break
        smallest = row_totals.index[:2]
        merged = table.loc[smallest].sum()
        table = table.drop(index=smallest)
        table.loc[RARE_LABEL] = merged
    return table


def chisq_independence_test(
    anomaly_counts: pd.Series, reference_counts: pd.Series, alpha: float = 0.05
) -> dict:
    """
    Compare deux répartitions données en volumes par catégorie.

    Retourne statistic, pvalue, method ('chi2', 'fisher' ou 'insufficient_data'),
    cramers_v (taille de l'effet), top_category, conclusion et `table` : observé,
    attendu et résidu standardisé par catégorie. Un résidu au-delà de ±2 désigne une
    catégorie qui s'écarte nettement de la référence.
    """
    table = pd.DataFrame({"anomalie": anomaly_counts, "reference": reference_counts}).fillna(0)
    table = table[table.sum(axis=1) > 0].astype(float)
    empty = {
        "statistic": None, "pvalue": None, "method": "insufficient_data", "cramers_v": None,
        "top_category": None, "table": None,
        "conclusion": "Données insuffisantes pour réaliser un test.",
    }
    if len(table) < 2 or (table.sum() == 0).any():
        return empty

    table = _merge_rare_categories(table)
    if len(table) < 2:
        return empty

    chi2_value, pvalue, _, expected = chi2_contingency(table.to_numpy(), correction=False)
    statistic, method = chi2_value, "chi2"
    if (expected < MIN_EXPECTED).any() and table.shape == (2, 2):
        statistic, pvalue = fisher_exact(table.to_numpy())
        method = "fisher"

    total = table.to_numpy().sum()
    row_share = table.sum(axis=1).to_numpy() / total
    column_share = table.sum(axis=0).to_numpy()[0] / total
    variance = expected[:, 0] * (1 - row_share) * (1 - column_share)
    residuals = (table["anomalie"].to_numpy() - expected[:, 0]) / np.sqrt(np.where(variance > 0, variance, np.nan))

    detail = pd.DataFrame({
        "observe": table["anomalie"].astype(int),
        "attendu": expected[:, 0],
        "residu_standardise": residuals,
    }, index=table.index)
    top_category = detail["residu_standardise"].abs().idxmax()
    cramers_v = float(np.sqrt(chi2_value / total))

    if pvalue < alpha:
        direction = "plus" if detail.loc[top_category, "residu_standardise"] > 0 else "moins"
        conclusion = (
            f"Écart significatif (p = {pvalue:.2g}, V de Cramér = {cramers_v:.2f}) : "
            f"« {top_category} » pèse nettement {direction} que dans la référence."
        )
    else:
        conclusion = (
            f"Pas d'écart significatif (p = {pvalue:.2g}) : la répartition reste "
            "compatible avec la référence."
        )
    return {
        "statistic": float(statistic), "pvalue": float(pvalue), "method": method,
        "cramers_v": cramers_v, "top_category": top_category, "table": detail,
        "conclusion": conclusion,
    }


# ---------------------------------------------------------------------------
# Qualification d'une anomalie
# ---------------------------------------------------------------------------

def motif_distribution_drilldown(
    df_master: pd.DataFrame,
    anomaly_period: pd.Timestamp,
    anomaly_group: str,
    detail_level: Literal["motif", "sous_motif"] = "motif",
    anomaly_start: pd.Timestamp | None = None,
    reference_periods: int | None = None,
) -> dict:
    """
    Répartition de la fenêtre d'anomalie comparée aux périodes qui la précèdent.

    La fenêtre va de `anomaly_start` (début estimé, par défaut la période elle-même) à
    `anomaly_period`. `reference_periods` limite la référence aux N périodes
    précédentes. `df_master` contient period, profil_partenaire, `detail_level` et
    n_traitements. Les traitements non qualifiés sont exclus du test et comptés à part.
    """
    if detail_level not in ("motif", "sous_motif"):
        raise ValueError("detail_level doit être 'motif' ou 'sous_motif'")

    group = df_master[df_master["profil_partenaire"] == anomaly_group]
    if group.empty:
        raise ValueError(f"Aucune donnée pour le groupe {anomaly_group}")

    start = anomaly_period if anomaly_start is None else min(anomaly_start, anomaly_period)
    current = group[group["period"].between(start, anomaly_period)]
    if current.empty:
        raise ValueError(f"Aucune donnée pour la période {anomaly_period}")

    reference = group[group["period"] < start]
    if reference_periods is not None:
        kept = sorted(reference["period"].unique())[-reference_periods:]
        reference = reference[reference["period"].isin(kept)]

    def counts(frame: pd.DataFrame) -> pd.Series:
        return frame.groupby(detail_level)["n_traitements"].sum()

    anomaly_counts, reference_counts = counts(current), counts(reference)
    comparison = pd.DataFrame({"Volume": anomaly_counts, "Volume référence": reference_counts}).fillna(0)
    comparison["Part (%)"] = comparison["Volume"] / max(comparison["Volume"].sum(), 1) * 100
    comparison["Part référence (%)"] = comparison["Volume référence"] / max(comparison["Volume référence"].sum(), 1) * 100

    total = current["n_traitements"].sum()
    return {
        "comparison": comparison.sort_values("Volume", ascending=False),
        "chisq_result": chisq_independence_test(anomaly_counts, reference_counts),
        "n_reference_periods": reference["period"].nunique(),
        "pct_non_qualifie": float((total - anomaly_counts.sum()) / total * 100) if total else 0.0,
    }


# ---------------------------------------------------------------------------
# Détection d'un changement de part
# ---------------------------------------------------------------------------

def _share_cusum(
    successes: np.ndarray, trials: np.ndarray, odds_ratio: float, h: float, min_baseline: int
) -> list[dict]:
    """CUSUM binomial en ligne : une hausse de la part par rapport à son passé.

    Toutes les périodes passées alimentent la référence, y compris celles qui ont fait
    monter le score. Les écarter biaiserait la référence vers le bas, puisqu'on
    retirerait précisément les hausses dues au hasard, et multiplierait les fausses
    alertes.
    """
    alerts = []
    base_success = base_trials = 0.0
    score, last_clear = 0.0, 0

    for t, (x, n) in enumerate(zip(successes, trials)):
        if base_trials >= min_baseline:
            p0 = (base_success + 0.5) / (base_trials + 1)
            p1 = odds_ratio * p0 / (1 + (odds_ratio - 1) * p0)
            score = max(0.0, score + x * np.log(p1 / p0) + (n - x) * np.log((1 - p1) / (1 - p0)))

            if score > h:
                window = range(last_clear, t + 1)
                alerts.append({
                    "start": last_clear, "alert": t, "part_reference": p0,
                    "part_observee": sum(successes[i] for i in window) / max(sum(trials[i] for i in window), 1),
                })
                # La référence repart de zéro pour apprendre le nouveau niveau.
                base_success, base_trials, score = 0.0, 0.0, 0.0
                last_clear = t + 1
                continue

        base_success += x
        base_trials += n
        if score == 0.0:
            last_clear = t + 1
    return alerts


def detect_motif_share_shifts(
    motif_table: pd.DataFrame,
    odds_ratio: float = SHARE_CUSUM_DEFAULTS["odds_ratio"],
    h: float = SHARE_CUSUM_DEFAULTS["h"],
    min_baseline: int = SHARE_CUSUM_DEFAULTS["min_baseline"],
) -> pd.DataFrame:
    """
    Cherche, pour chaque couple (profil, motif), une hausse durable de la part du motif
    parmi les traitements qualifiés. Détecte une part dont les cotes doublent, à partir
    de `min_baseline` traitements de référence.

    Retourne une ligne par alerte : profil_partenaire, motif, debut_estime, period (date
    de détection), part_reference, part_observee.
    """
    columns = ["profil_partenaire", "motif", "debut_estime", "period", "part_reference", "part_observee"]
    qualified = motif_table[motif_table["motif"].notna()]
    if qualified.empty:
        return pd.DataFrame(columns=columns)

    counts = (
        qualified.groupby(["profil_partenaire", "period", "motif"])["n_traitements"].sum()
        .unstack("motif", fill_value=0)
    )
    rows = []
    for profile, frame in counts.groupby(level="profil_partenaire"):
        periods = frame.index.get_level_values("period")
        trials = frame.sum(axis=1).to_numpy(dtype=float)
        for motif in frame.columns:
            successes = frame[motif].to_numpy(dtype=float)
            for alert in _share_cusum(successes, trials, odds_ratio, h, min_baseline):
                rows.append({
                    "profil_partenaire": profile,
                    "motif": motif,
                    "debut_estime": periods[alert["start"]],
                    "period": periods[alert["alert"]],
                    "part_reference": alert["part_reference"],
                    "part_observee": alert["part_observee"],
                })
    return pd.DataFrame(rows, columns=columns).sort_values("period", ascending=False, ignore_index=True)


# ---------------------------------------------------------------------------
# Comparatif de performance
# ---------------------------------------------------------------------------

def performance_comparison_motif_aggregation(
    treatments_df: pd.DataFrame, detail_level: str = "motif", n_iterations: int = 50
) -> dict:
    """
    Compare deux façons de répondre à une même question (volume par période et motif) :
    - requête directe : agrégation de toutes les lignes à chaque demande ;
    - table pré-agrégée : construite une fois, puis ré-agrégée à chaque demande.

    La construction de la table est mesurée à part : elle a un coût, amorti au-delà
    d'un certain nombre de requêtes (`break_even_queries`).
    """
    keys = ["period", "profil_partenaire", "motif", "sous_motif"]
    query = ["period", detail_level]

    def timed(function) -> tuple[float, object]:
        start = time.perf_counter()
        result = function()
        return (time.perf_counter() - start) * 1000, result

    build_ms, cube = timed(lambda: treatments_df.groupby(keys, dropna=False, observed=True).size().rename("n").reset_index())

    def average_ms(function) -> float:
        function()  # échauffement
        return float(np.mean([timed(function)[0] for _ in range(n_iterations)]))

    raw_ms = average_ms(lambda: treatments_df.groupby(query, dropna=False).size())
    cube_ms = average_ms(lambda: cube.groupby(query, dropna=False)["n"].sum())
    gain = raw_ms - cube_ms
    return {
        "time_raw_ms": round(raw_ms, 3),
        "time_cube_ms": round(cube_ms, 3),
        "build_cube_ms": round(build_ms, 3),
        "speedup": round(raw_ms / cube_ms, 1) if cube_ms else float("inf"),
        "break_even_queries": int(np.ceil(build_ms / gain)) if gain > 0 else None,
        "rows_raw": len(treatments_df),
        "rows_cube": len(cube),
        "iterations": n_iterations,
    }
