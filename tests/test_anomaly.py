import numpy as np
import pandas as pd
import pytest

from metrics.anomaly import (
    add_cusum_breaks,
    add_deviation_from_own_history,
    flag_alerts,
    period_over_period_table,
)


def weekly(values, group="A"):
    values = np.asarray(values)
    return pd.DataFrame({
        "period": pd.date_range("2025-01-06", periods=len(values), freq="W-MON"),
        "profil_partenaire": group,
        "value": values / 1000,
        "n_traitements": values,
    })


def poisson_series(rng, level, n=80):
    return rng.poisson(level, n)


def test_score_uses_only_past_periods():
    data = weekly([500] * 40 + [900] * 10)
    first = add_deviation_from_own_history(data, "value").reset_index(drop=True)
    changed = data.copy()
    changed.loc[45:, ["value", "n_traitements"]] = [0.9, 100000]
    second = add_deviation_from_own_history(changed, "value").reset_index(drop=True)
    assert first.loc[:40, "zscore"].equals(second.loc[:40, "zscore"])


def test_isolated_spike_does_not_shift_the_reference():
    rng = np.random.default_rng(0)
    values = poisson_series(rng, 500, 40)
    values[20] = 5000
    scored = add_deviation_from_own_history(weekly(values), "value")
    assert scored.iloc[25]["hist_ref"] == pytest.approx(0.5, rel=0.05)


def test_daily_scores_compare_same_weekdays():
    days = pd.date_range("2025-01-06", periods=70, freq="D")
    counts = np.where(days.weekday >= 5, 100, 1000)
    data = pd.DataFrame({
        "period": days, "profil_partenaire": "A", "value": counts / 1000, "n_traitements": counts,
    })
    scored = add_deviation_from_own_history(data, "value", freq_code="D")
    assert scored["zscore"].abs().max() < 1


def test_flags_respect_minimum_volume():
    data = pd.DataFrame({
        "zscore": [3.5, 0.5, 4.0, 1.0], "pct_deviation": [30, 5, 40, 10], "n_traitements": [10, 5, 20, 15],
    })
    assert list(flag_alerts(data, "zscore", 3.0, 15)["is_alert"]) == [False, False, True, False]
    assert list(flag_alerts(data, "pct", 25.0, 15)["is_alert"]) == [False, False, True, False]


def test_low_volume_rows_have_no_score():
    scored = add_deviation_from_own_history(weekly([10] * 30), "value", min_effectif=20)
    assert scored["zscore"].isna().all()


def test_cusum_is_silent_before_the_break():
    rng = np.random.default_rng(1)
    values = np.concatenate([rng.poisson(500, 52), rng.poisson(800, 26)])
    result = add_cusum_breaks(weekly(values), "n_traitements").reset_index(drop=True)
    assert not result.loc[:51, "cusum_break"].any()


def test_cusum_detects_a_step_quickly_and_only_once():
    rng = np.random.default_rng(2)
    values = np.concatenate([rng.poisson(500, 52), rng.poisson(800, 26)])
    result = add_cusum_breaks(weekly(values), "n_traitements").reset_index(drop=True)
    detected = result.index[result["cusum_break"]]
    assert 52 <= detected[0] <= 56
    assert result["cusum_direction"].iloc[detected[0]] == 1
    assert len(detected) == 1


def test_cusum_detects_drops():
    rng = np.random.default_rng(3)
    values = np.concatenate([rng.poisson(800, 52), rng.poisson(400, 20)])
    result = add_cusum_breaks(weekly(values), "n_traitements")
    assert result.loc[result["cusum_break"], "cusum_direction"].iloc[0] == -1


def test_cusum_ignores_a_single_spike():
    rng = np.random.default_rng(4)
    values = poisson_series(rng, 500)
    values[40] = 3000
    assert not add_cusum_breaks(weekly(values), "n_traitements")["cusum_break"].any()


def test_cusum_rarely_fires_on_stable_series():
    false_alarms = sum(
        add_cusum_breaks(weekly(poisson_series(np.random.default_rng(seed), 500)), "n_traitements")["cusum_break"].sum()
        for seed in range(50)
    )
    assert false_alarms <= 10  # 50 séries de 80 semaines


def test_cusum_groups_are_independent():
    rng = np.random.default_rng(5)
    stable = weekly(rng.poisson(500, 60), "stable")
    moving = weekly(np.concatenate([rng.poisson(500, 40), rng.poisson(900, 20)]), "moving")
    result = add_cusum_breaks(pd.concat([stable, moving], ignore_index=True), "n_traitements")
    assert not result[result["profil_partenaire"] == "stable"]["cusum_break"].any()
    assert result[result["profil_partenaire"] == "moving"]["cusum_break"].any()


def test_cusum_respects_minimum_volume():
    rng = np.random.default_rng(6)
    values = np.concatenate([rng.poisson(10, 40), rng.poisson(16, 20)])
    assert not add_cusum_breaks(weekly(values), "n_traitements", min_effectif=100)["cusum_break"].any()


def test_period_comparison_keeps_the_requested_order():
    """Période A plus récente que B : l'écart doit rester B - A, sans inversion."""
    df = pd.DataFrame({
        "period": pd.to_datetime(["2026-01-05", "2026-01-12"]),
        "profil_partenaire": "P", "taux": [0.2, 0.5],
    })
    row = period_over_period_table(df, pd.Timestamp("2026-01-12"), pd.Timestamp("2026-01-05"), "taux").iloc[0]
    assert (row["value_a"], row["value_b"]) == (0.5, 0.2)
    assert row["delta_abs"] == pytest.approx(-0.3)
