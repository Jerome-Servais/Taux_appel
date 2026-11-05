import numpy as np
import pandas as pd
import pytest
from scipy.stats import chi2_contingency

from metrics.motive_analysis import (
    RARE_LABEL,
    chisq_independence_test,
    detect_motif_share_shifts,
    motif_distribution_drilldown,
    performance_comparison_motif_aggregation,
)


def series(**counts):
    return pd.Series(counts, dtype=float)


def test_test_uses_volumes_not_rows():
    # Une ligne par catégorie, mais des volumes importants.
    result = chisq_independence_test(series(A=300, B=100), series(A=200, B=200))
    expected_p = chi2_contingency(np.array([[300, 200], [100, 200]]), correction=False)[1]
    assert result["method"] == "chi2"
    assert result["pvalue"] == pytest.approx(expected_p)


def test_test_names_the_category_responsible():
    result = chisq_independence_test(
        series(Reclamation=60, Facture=30, Declaration=30),
        series(Reclamation=30, Facture=60, Declaration=60),
    )
    assert result["top_category"] == "Reclamation"
    assert result["cramers_v"] > 0.2
    assert "plus" in result["conclusion"]


def test_identical_distributions_are_not_significant():
    result = chisq_independence_test(series(A=50, B=50), series(A=100, B=100))
    assert result["pvalue"] > 0.9


def test_rare_categories_are_merged_by_row():
    anomaly = series(A=50, B=40, C=1, D=1, E=1)
    reference = series(A=100, B=90, C=2, D=1, E=2)
    result = chisq_independence_test(anomaly, reference)
    assert RARE_LABEL in result["table"].index
    assert (result["table"]["attendu"] >= 5).all()


def test_small_two_by_two_uses_fisher():
    result = chisq_independence_test(series(A=2, B=3), series(A=30, B=8))
    assert result["method"] == "fisher"


@pytest.mark.parametrize("anomaly, reference", [
    (pd.Series(dtype=float), pd.Series(dtype=float)),
    (series(A=10), series(A=20)),
    (series(A=10, B=5), pd.Series(dtype=float)),
])
def test_insufficient_data_is_reported(anomaly, reference):
    result = chisq_independence_test(anomaly, reference)
    assert result["method"] == "insufficient_data"
    assert result["pvalue"] is None


@pytest.fixture
def motif_table():
    rows = []
    for week, period in enumerate(pd.date_range("2026-01-05", periods=20, freq="W-MON")):
        share = 30 if week >= 15 else 10
        rows += [
            (period, "P", "Reclamation", "X", share),
            (period, "P", "Facture", "Y", 100 - share),
            (period, "P", None, None, 10),
        ]
    return pd.DataFrame(rows, columns=["period", "profil_partenaire", "motif", "sous_motif", "n_traitements"])


def test_drilldown_keeps_anomaly_out_of_the_reference(motif_table):
    periods = sorted(motif_table["period"].unique())
    result = motif_distribution_drilldown(
        motif_table, periods[-1], "P", "motif", anomaly_start=periods[15]
    )
    assert result["n_reference_periods"] == 15
    assert result["comparison"].loc["Reclamation", "Part référence (%)"] == pytest.approx(10)
    assert result["comparison"].loc["Reclamation", "Part (%)"] == pytest.approx(30)
    assert result["chisq_result"]["pvalue"] < 1e-6


def test_drilldown_reports_unqualified_share(motif_table):
    last = motif_table["period"].max()
    assert motif_distribution_drilldown(motif_table, last, "P")["pct_non_qualifie"] == pytest.approx(100 * 10 / 110)


def test_drilldown_rejects_unknown_inputs(motif_table):
    last = motif_table["period"].max()
    with pytest.raises(ValueError):
        motif_distribution_drilldown(motif_table, last, "inconnu")
    with pytest.raises(ValueError):
        motif_distribution_drilldown(motif_table, last, "P", "autre")


def simulated_table(rng, share_after, n_weeks=78, break_week=52, volume=25):
    rows = []
    for week, period in enumerate(pd.date_range("2025-01-06", periods=n_weeks, freq="W-MON")):
        share = share_after if week >= break_week else 0.12
        total = rng.poisson(volume)
        target = rng.binomial(total, share)
        rows += [
            (period, "P", "Cible", "c", target),
            (period, "P", "Autre", "a", total - target),
        ]
    return pd.DataFrame(rows, columns=["period", "profil_partenaire", "motif", "sous_motif", "n_traitements"])


def test_share_shift_is_detected_soon_after_it_starts():
    delays = []
    for seed in range(30):
        table = simulated_table(np.random.default_rng(seed), 0.32)
        alerts = detect_motif_share_shifts(table)
        alerts = alerts[(alerts["motif"] == "Cible") & (alerts["period"] >= pd.Timestamp("2026-01-05"))]
        delays.append((alerts["period"].min() - pd.Timestamp("2026-01-05")).days / 7 if len(alerts) else np.nan)
    assert np.mean(np.isnan(delays)) <= 0.1
    assert np.nanmedian(delays) <= 6


def test_share_shift_rarely_fires_without_change():
    alerts = sum(len(detect_motif_share_shifts(simulated_table(np.random.default_rng(s), 0.12))) for s in range(30))
    assert alerts <= 15  # 30 séries de 18 mois, 2 motifs surveillés


def test_share_shift_without_motifs_is_empty():
    table = pd.DataFrame({"period": [pd.Timestamp("2026-01-05")], "profil_partenaire": ["P"],
                          "motif": [None], "n_traitements": [5]})
    assert detect_motif_share_shifts(table).empty


def test_performance_comparison_returns_measures():
    n = 20000
    data = pd.DataFrame({
        "period": np.repeat(pd.date_range("2026-01-05", periods=4, freq="W-MON"), n // 4),
        "profil_partenaire": "P", "motif": "M", "sous_motif": "S",
    })
    result = performance_comparison_motif_aggregation(data, n_iterations=3)
    assert result["rows_cube"] == 4
    assert result["time_raw_ms"] > 0
