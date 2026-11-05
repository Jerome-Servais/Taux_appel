import numpy as np
import pandas as pd
import pytest

from forecasting.backtest import backtest_all_models, backtest_model, generate_walk_forward_folds
from forecasting.registry import (
    MODEL_REGISTRY,
    NegBinGLMModel,
    SeasonalBaselineModel,
    _seasonal_design,
    _uses_seasonal_effects,
    forecast_series,
)


def weekly_series(n=78, level=800, seed=0):
    rng = np.random.default_rng(seed)
    index = pd.date_range("2025-01-06", periods=n, freq="W-MON")
    return pd.Series(rng.negative_binomial(40, 40 / (40 + level), n), index=index, dtype=float)


def daily_series(n=210, seed=0):
    rng = np.random.default_rng(seed)
    index = pd.date_range("2025-01-06", periods=n, freq="D")
    return pd.Series(rng.poisson(np.where(index.dayofweek >= 5, 40, 120)), index=index, dtype=float)


def test_design_keeps_all_columns_on_a_short_horizon():
    short = pd.date_range("2026-01-07", periods=3, freq="D")  # mercredi à vendredi
    full = pd.date_range("2026-01-05", periods=14, freq="D")
    assert list(_seasonal_design(short, np.arange(3), "D").columns) == list(_seasonal_design(full, np.arange(14), "D").columns)
    assert _seasonal_design(short, np.arange(3), "D").loc[0, "dow_2"] == 1.0


def test_future_month_is_not_treated_as_january():
    weeks = pd.date_range("2026-11-02", periods=4, freq="W-MON")
    assert _seasonal_design(weeks, np.arange(4), "W-MON")["month_11"].sum() == 4


@pytest.mark.parametrize("freq, n, expected", [
    ("D", 100, True), ("W-MON", 78, False), ("W-MON", 104, True), ("MS", 17, False), ("MS", 24, True),
])
def test_month_effects_require_two_years(freq, n, expected):
    assert _uses_seasonal_effects(freq, n) is expected


def test_glm_stays_near_the_level_on_weekly_data():
    series = weekly_series()
    forecast, note = forecast_series(series, "negbin_glm", 12)
    assert note is None
    assert forecast["yhat"].mean() == pytest.approx(series.mean(), rel=0.25)


def test_daily_glm_learns_the_weekend():
    forecast = NegBinGLMModel().fit(daily_series()).predict(7)
    weekend = forecast["ds"].dt.dayofweek >= 5
    assert forecast.loc[weekend, "yhat"].mean() < forecast.loc[~weekend, "yhat"].mean() / 2


def test_baseline_aligns_weekdays():
    series = daily_series(70)
    forecast = SeasonalBaselineModel().fit(series).predict(14)
    weekend = forecast["ds"].dt.dayofweek >= 5
    assert forecast.loc[weekend, "yhat"].max() < forecast.loc[~weekend, "yhat"].min()


@pytest.mark.parametrize("name", [n for n in MODEL_REGISTRY if not n.startswith("bayesian")])
@pytest.mark.parametrize("series", [weekly_series(), daily_series()], ids=["semaine", "jour"])
def test_models_return_ordered_intervals(name, series):
    forecast = MODEL_REGISTRY[name]().fit(series).predict(5)
    assert len(forecast) == 5
    assert (forecast["yhat_lower"] <= forecast["yhat_upper"]).all()
    assert (forecast["yhat"] >= 0).all()


def test_backtest_folds_stay_inside_the_series():
    for train_end, test_end in generate_walk_forward_folds(100, 12, 5):
        assert train_end >= 50
        assert test_end <= 100


def test_backtest_reports_wape():
    result = backtest_model(weekly_series(), "baseline", 8, 3)
    assert result["n_failures"] == 0
    assert result["wape"] == pytest.approx(result["mae"] / weekly_series().mean() * 100, rel=0.1)


def test_negbin_covers_overdispersed_series_better_than_poisson():
    series = weekly_series()
    assert backtest_model(series, "negbin_glm", 8, 5)["coverage"] > backtest_model(series, "poisson_glm", 8, 5)["coverage"]


def test_leaderboard_compares_with_the_reference(monkeypatch):
    import forecasting.backtest as backtest

    fast = {k: v for k, v in MODEL_REGISTRY.items() if k in ("baseline", "linear")}
    monkeypatch.setattr(backtest, "MODEL_REGISTRY", fast)
    board = backtest_all_models(weekly_series(), 8, 3).set_index("model")
    assert board.loc["baseline", "gain_vs_reference"] == pytest.approx(0)


def test_short_history_is_rejected():
    with pytest.raises(ValueError):
        forecast_series(weekly_series(3), "linear", 2)


def test_negbin_keeps_the_trend_of_the_poisson_glm():
    """Même moyenne que Poisson : l'optimiseur ne doit pas s'arrêter loin de l'optimum."""
    y = np.r_[np.full(40, 2000.0), np.full(25, 2500.0)] + np.tile([60.0, -40.0, 15.0, -35.0, 10.0], 13)
    series = pd.Series(y, index=pd.date_range("2025-01-06", periods=len(y), freq="W-MON"))

    poisson = MODEL_REGISTRY["poisson_glm"]().fit(series).predict(6)["yhat"].to_numpy()
    negbin = MODEL_REGISTRY["negbin_glm"]().fit(series).predict(6)["yhat"].to_numpy()

    assert np.abs(negbin / poisson - 1).max() < 0.02
