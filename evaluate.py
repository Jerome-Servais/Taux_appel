"""
evaluate.py
===========
Évaluation reproductible, sur données simulées : `python evaluate.py`.

Les anomalies sont injectées à une date connue, ce qui permet de mesurer ce qu'un
tableau de bord ne montre pas : les fausses alertes et le délai de détection.

1. Volume : une rupture de volume est-elle détectée, et à quel prix ?
2. Motif : un motif qui gagne du poids sans changer le volume total est-il détecté ?
3. Prévision : les modèles font-ils mieux qu'une référence simple ?
"""

from __future__ import annotations

import argparse
import logging
import warnings

import numpy as np
import pandas as pd

from data.aggregator import build_master_table, resolve_profile_and_relation
from data.loader import load_demo_data
from data.motif_taxonomy import MOTIF_WEIGHTS
from forecasting.backtest import backtest_model
from forecasting.registry import MODEL_REGISTRY
from metrics.anomaly import add_cusum_breaks, add_deviation_from_own_history, flag_alerts
from metrics.motive_analysis import detect_motif_share_shifts

# Dates fixes : les données de démonstration de l'application suivent la date du jour.
DEMO_PERIOD = ("2025-05-01", "2026-10-31")
DAYS = pd.date_range(*DEMO_PERIOD)
BREAK_INDEX = 2 * len(DAYS) // 3
BREAK_DATE = DAYS[BREAK_INDEX]
PROFILES = {"Standard": (6000, 0.014, 0.010), "Digital": (2500, 0.006, 0.022)}
WEEKS_PER_PERIOD = {"W-MON": 1, "MS": 4.35}
PERIOD_LENGTH = {"W-MON": pd.Timedelta(days=7), "MS": pd.offsets.MonthBegin(1)}
SCENARIOS = [
    ("Stable", "Standard", "tous", 1.0),
    ("Appels +15 %", "Standard", "tous", 1.15),
    ("Appels +50 %", "Digital", "tous", 1.5),
    ("Appels x4,2", "Digital", "tous", 4.2),
    ("Appels x4,2, vue appels", "Digital", "appel", 4.2),
]


def simulate_daily_counts(rng, profile: str, canal: str, multiplier: float) -> pd.Series:
    """Comptages journaliers : effet week-end, campagnes email et rupture à date connue."""
    accounts, call_rate, email_rate = PROFILES[profile]
    weekend = DAYS.weekday >= 5
    calls = accounts * call_rate * np.where(weekend, 0.55, 1.0)
    calls = np.where(np.arange(len(DAYS)) >= BREAK_INDEX, calls * multiplier, calls)
    counts = rng.poisson(calls)
    if canal == "tous":
        campaigns = np.zeros(len(DAYS), dtype=bool)
        campaigns[rng.choice(len(DAYS), max(1, len(DAYS) // 90), replace=False)] = True
        emails = accounts * email_rate * np.where(weekend, 0.85, 1.0) * np.where(campaigns, 4.0, 1.0)
        counts = counts + rng.poisson(emails)
    return pd.Series(counts, index=DAYS)


def to_periods(daily: pd.Series, freq_code: str) -> pd.DataFrame:
    if freq_code == "W-MON":
        starts = daily.index - pd.to_timedelta(daily.index.weekday, unit="D")
        series = daily.groupby(starts).sum().iloc[1:-1]  # semaines complètes uniquement
    else:
        series = daily.groupby(daily.index.to_period("M").start_time).sum()
    return pd.DataFrame({
        "period": series.index, "profil_partenaire": "P",
        "n_traitements": series.values, "value": series.values.astype(float),
    })


def evaluate_volume(n_runs: int, threshold: float = 3.0) -> pd.DataFrame:
    rows = []
    for freq_code in WEEKS_PER_PERIOD:
        for label, profile, canal, multiplier in SCENARIOS:
            outcomes = {"Z-score": [], "CUSUM": []}
            for run in range(n_runs):
                rng = np.random.default_rng(50_000 + run)
                table = to_periods(simulate_daily_counts(rng, profile, canal, multiplier), freq_code)
                table = add_deviation_from_own_history(table, "value", freq_code=freq_code)
                table = flag_alerts(table, "zscore", threshold, min_effectif=20)
                table = add_cusum_breaks(table, "n_traitements", freq_code=freq_code).sort_values("period")
                # Une période n'est « avant » que si elle se termine avant la rupture. Une alerte
                # n'est visible qu'à la fin de sa période : le délai se mesure jusque-là.
                period_ends = table["period"] + PERIOD_LENGTH[freq_code]
                before = period_ends <= BREAK_DATE
                n_before = int(before.sum())
                # Seules les périodes dotées d'un historique de référence sont surveillées.
                years_watched = (before & table["hist_ref"].notna()).sum() * WEEKS_PER_PERIOD[freq_code] / 52
                for name, flags in (("Z-score", table["is_alert"]), ("CUSUM", table["cusum_break"])):
                    hits = np.where(flags.to_numpy()[n_before:])[0]
                    delay = (period_ends.iloc[n_before + hits[0]] - BREAK_DATE).days / 7 if len(hits) else np.nan
                    outcomes[name].append((flags[before].sum() / years_watched, delay))
            for name, values in outcomes.items():
                values = np.array(values, dtype=float)
                detected = ~np.isnan(values[:, 1])
                rows.append({
                    "Granularité": freq_code, "Scénario": label, "Détecteur": name,
                    "Fausses alertes / an": values[:, 0].mean(),
                    "Détecté (%)": detected.mean() * 100,
                    "Délai médian (semaines)": np.median(values[detected, 1]) if detected.any() else np.nan,
                })
    return pd.DataFrame(rows)


def simulate_motif_table(rng, multiplier: float, weekly_volume: int = 22, n_weeks: int = 78) -> pd.DataFrame:
    """Le motif « Réclamation » gagne du poids à la semaine 52, volume total inchangé."""
    names = list(MOTIF_WEIGHTS)
    target = names.index("Réclamation")
    base = np.array([MOTIF_WEIGHTS[name] for name in names])
    rows = []
    for week, period in enumerate(pd.date_range("2025-01-06", periods=n_weeks, freq="W-MON")):
        weights = base.copy()
        if week >= 52:
            weights[target] *= multiplier
        counts = rng.multinomial(rng.poisson(weekly_volume), weights / weights.sum())
        rows += [(period, "P", name, count) for name, count in zip(names, counts)]
    return pd.DataFrame(rows, columns=["period", "profil_partenaire", "motif", "n_traitements"])


def evaluate_motifs(n_runs: int) -> pd.DataFrame:
    shift_date = pd.Timestamp("2025-01-06") + pd.Timedelta(weeks=52)
    false_alerts = [len(detect_motif_share_shifts(simulate_motif_table(np.random.default_rng(70_000 + r), 1.0))) / 1.5
                    for r in range(n_runs)]
    delays = []
    for run in range(n_runs):
        alerts = detect_motif_share_shifts(simulate_motif_table(np.random.default_rng(80_000 + run), 3.5))
        alerts = alerts[(alerts["motif"] == "Réclamation") & (alerts["period"] >= shift_date)]
        delays.append((alerts["period"].min() - shift_date).days / 7 if len(alerts) else np.nan)
    delays = np.array(delays)
    return pd.DataFrame([{
        "Fausses alertes / an, par profil": np.mean(false_alerts),
        "Détecté (%)": (~np.isnan(delays)).mean() * 100,
        "Délai médian (semaines)": np.nanmedian(delays),
    }])


def evaluate_forecasts(models: list[str]) -> pd.DataFrame:
    treatments, portfolio = load_demo_data(*DEMO_PERIOD)
    resolved = resolve_profile_and_relation(treatments, portfolio)
    rows = []
    for freq_code, horizon, label in (("W-MON", 12, "Semaine, 12 périodes"), ("D", 28, "Jour, 28 périodes")):
        master = build_master_table(resolved, portfolio, freq_code)
        for target in (None, "Standard", "Digital"):
            subset = master if target is None else master[master["profil_partenaire"] == target]
            series = subset.groupby("period")["n_traitements"].sum().sort_index()
            for model in models:
                result = backtest_model(series, model, horizon, n_folds=8)
                rows.append({
                    "Série": f"{target or 'Global'} ({label})", "Modèle": model, "MAE": result["mae"],
                    "RMSE": result["rmse"], "WAPE (%)": result["wape"], "Biais": result["bias"],
                    "Couverture IC (%)": result["coverage"], "Échecs": result["n_failures"],
                })
    return pd.DataFrame(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs", type=int, default=300, help="simulations par scénario")
    parser.add_argument("--all-models", action="store_true", help="inclut les modèles bayésiens, plus lents")
    args = parser.parse_args()

    warnings.filterwarnings("ignore")
    np.random.seed(0)  # Prophet tire ses intervalles avec le générateur global
    logging.getLogger("cmdstanpy").setLevel(logging.ERROR)
    logging.getLogger("pymc").setLevel(logging.ERROR)
    pd.set_option("display.width", 160)
    models = [m for m in MODEL_REGISTRY if args.all_models or not m.startswith("bayesian")]

    print("\n## Rupture de volume\n")
    print(evaluate_volume(args.runs).round(1).to_string(index=False))
    print("\n## Changement de motif (part multipliée par 3,5, volume inchangé)\n")
    print(evaluate_motifs(args.runs).round(2).to_string(index=False))
    print("\n## Prévision (8 replis glissants)\n")
    print(evaluate_forecasts(models).round(1).to_string(index=False))


if __name__ == "__main__":
    main()
