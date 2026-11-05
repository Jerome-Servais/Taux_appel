"""
forecasting/registry.py — MODEL_REGISTRY réutilisable, appliqué par profil.

Chaque modèle est entraîné sur la série d'un profil ou sur le volume global.
`forecast_series` isole les échecs : si un modèle échoue, l'application bascule sur
la régression linéaire et le signale. La validation (backtest.py) instancie au
contraire les modèles directement, pour que ces échecs restent visibles.
"""

from __future__ import annotations

import logging
import warnings

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.linear_model import LinearRegression

warnings.filterwarnings("ignore")
logging.getLogger("cmdstanpy").setLevel(logging.WARNING)
logging.getLogger("prophet").setLevel(logging.WARNING)


# ---------------------------------------------------------------------------
# Utilitaires communs
# ---------------------------------------------------------------------------

def infer_freq(index: pd.DatetimeIndex) -> str:
    freq = pd.infer_freq(index)
    if freq is not None:
        return freq
    diff = index[-1] - index[-2]
    if diff.days >= 27:
        return "MS"
    if diff.days >= 6:
        return "W-MON"
    return "D"


def future_index(last_ts: pd.Timestamp, freq: str, horizon: int) -> pd.DatetimeIndex:
    return pd.date_range(start=last_ts, periods=horizon + 1, freq=freq)[1:]


def compute_dispersion_ratio(ts: pd.Series) -> float:
    """Ratio variance/moyenne empirique (indicateur simple de sur-dispersion).

    Un ratio proche de 1 est cohérent avec une hypothèse Poisson. Un ratio
    nettement supérieur à 1 (cf. config.OVERDISPERSION_WARNING_RATIO) suggère
    que Poisson sous-estimera l'incertitude — typiquement quand on mélange des
    canaux hétérogènes (appels + emails) ou en présence de pics ponctuels
    (campagnes) : une Binomiale négative (dispersion estimée) est alors
    préférable aux variantes Poisson du registre.
    """
    y = ts.dropna().to_numpy(dtype=float)
    if len(y) < 3 or y.mean() == 0:
        return float("nan")
    return float(y.var(ddof=1) / y.mean())


def _uses_seasonal_effects(freq: str, n_points: int) -> bool:
    """Le jour de semaine est toujours estimable en journalier. Les effets de mois
    exigent deux cycles annuels : sur 18 mois, le modèle confondrait les mois avec la
    tendance, et la prévision s'écarterait fortement du niveau réel."""
    if freq == "D":
        return True
    if freq == "W-MON":
        return n_points >= 104
    return n_points >= 24


def _seasonal_design(index: pd.DatetimeIndex, t: np.ndarray, freq: str, use_seasonal: bool = True) -> pd.DataFrame:
    """Tendance et indicatrices calendaires. Les catégories sont fixes : un futur court
    (une partie de la semaine, un seul mois) garde ainsi les colonnes de l'entraînement."""
    design = {"t": t.astype(float)}
    if use_seasonal and freq == "D":
        for day in range(1, 7):
            design[f"dow_{day}"] = (index.dayofweek == day).astype(float)
    elif use_seasonal:
        for month in range(2, 13):
            design[f"month_{month}"] = (index.month == month).astype(float)
    return pd.DataFrame(design)


def _negbin_interval(mean: np.ndarray, alpha: float) -> tuple[np.ndarray, np.ndarray]:
    """Intervalle à 95 % d'une binomiale négative de variance mean + alpha × mean²."""
    mean = np.clip(mean, 1e-6, None)
    size = 1.0 / max(alpha, 1e-6)
    prob = size / (size + mean)
    return stats.nbinom.ppf(0.025, size, prob), stats.nbinom.ppf(0.975, size, prob)


def _estimate_alpha(y: np.ndarray, mean: np.ndarray) -> float:
    """Dispersion alpha estimée par la méthode des moments."""
    mean = np.clip(mean, 1e-6, None)
    return max(float(np.sum((y - mean) ** 2 - mean) / np.sum(mean ** 2)), 1e-6)


# ---------------------------------------------------------------------------
# Interface commune
# ---------------------------------------------------------------------------

class BaseForecastModel:
    name = "base"

    def __init__(self, **kwargs):
        self.params = kwargs
        self.model_note: str | None = None

    def fit(self, ts: pd.Series) -> "BaseForecastModel":
        raise NotImplementedError

    def predict(self, horizon: int) -> pd.DataFrame:
        """Retourne un DataFrame avec les colonnes : ds, yhat, yhat_lower, yhat_upper."""
        raise NotImplementedError


# ---------------------------------------------------------------------------
# Modèles
# ---------------------------------------------------------------------------

class SeasonalBaselineModel(BaseForecastModel):
    """Référence : médiane des dernières périodes, par jour de semaine en journalier.
    Son intervalle vient de ses propres erreurs passées. Un modèle plus complexe ne se
    justifie que s'il fait mieux."""

    name = "baseline"
    RECENT = {"D": 28, "W-MON": 8, "MS": 3}

    def fit(self, ts: pd.Series) -> "SeasonalBaselineModel":
        self.ts = ts
        self.freq = infer_freq(ts.index)
        self.y = ts.to_numpy(dtype=float)
        return self

    def _point_forecast(self, history: np.ndarray, horizon: int) -> np.ndarray:
        recent = history[-self.RECENT.get(self.freq, 8):]
        if self.freq != "D":
            return np.repeat(np.median(recent), horizon)
        by_position = [np.median(recent[::-1][offset::7]) for offset in range(7)]
        return np.array([by_position[6 - i % 7] for i in range(horizon)])

    def predict(self, horizon: int) -> pd.DataFrame:
        idx = future_index(self.ts.index[-1], self.freq, horizon)
        yhat = self._point_forecast(self.y, horizon)

        first = max(self.RECENT.get(self.freq, 8), 4)
        step = max(1, (len(self.y) - horizon - first) // 40)
        errors = [
            self.y[t:t + horizon] - self._point_forecast(self.y[:t], horizon)
            for t in range(first, len(self.y) - horizon, step)
        ]
        low, high = np.quantile(np.concatenate(errors), [0.025, 0.975]) if errors else (0.0, 0.0)
        return pd.DataFrame({
            "ds": idx, "yhat": yhat,
            "yhat_lower": np.clip(yhat + low, 0, None), "yhat_upper": np.clip(yhat + high, 0, None),
        })


class LinearTrendModel(BaseForecastModel):
    name = "linear"

    def fit(self, ts: pd.Series) -> "LinearTrendModel":
        self.ts = ts
        self.freq = infer_freq(ts.index)
        self.n = len(ts)
        self.use_seasonal = _uses_seasonal_effects(self.freq, self.n)
        X = _seasonal_design(ts.index, np.arange(self.n), self.freq, self.use_seasonal)
        y = ts.to_numpy(dtype=float)
        self.model = LinearRegression().fit(X, y)
        self.resid_std = float(np.std(y - self.model.predict(X)))
        return self

    def predict(self, horizon: int) -> pd.DataFrame:
        idx = future_index(self.ts.index[-1], self.freq, horizon)
        X = _seasonal_design(idx, np.arange(self.n, self.n + horizon), self.freq, self.use_seasonal)
        yhat = np.clip(self.model.predict(X), 0, None)
        lower = np.clip(yhat - 1.96 * self.resid_std, 0, None)
        upper = yhat + 1.96 * self.resid_std
        return pd.DataFrame({"ds": idx, "yhat": yhat, "yhat_lower": lower, "yhat_upper": upper})


class SarimaModel(BaseForecastModel):
    name = "sarima"

    def fit(self, ts: pd.Series) -> "SarimaModel":
        from statsmodels.tsa.statespace.sarimax import SARIMAX

        self.ts = ts
        self.freq = infer_freq(ts.index)
        y = ts.to_numpy(dtype=float)

        order = (1, 1, 1)
        seasonal_order = (0, 0, 0, 0)
        if self.freq == "D" and len(y) >= 21:
            seasonal_order = (1, 0, 1, 7)
        elif self.freq == "MS" and len(y) >= 24:
            seasonal_order = (1, 0, 0, 12)

        try:
            self.res = SARIMAX(
                y, order=order, seasonal_order=seasonal_order,
                enforce_stationarity=False, enforce_invertibility=False,
            ).fit(disp=False)
        except Exception as e:
            self.res = SARIMAX(
                y, order=(1, 1, 1), seasonal_order=(0, 0, 0, 0),
                enforce_stationarity=False, enforce_invertibility=False,
            ).fit(disp=False)
            self.model_note = f"SARIMA saisonnier non convergent, repli sur ARIMA(1,1,1) simple ({type(e).__name__})."
        return self

    def predict(self, horizon: int) -> pd.DataFrame:
        idx = future_index(self.ts.index[-1], self.freq, horizon)
        fc = self.res.get_forecast(steps=horizon)
        yhat = np.clip(np.asarray(fc.predicted_mean), 0, None)
        ci = np.asarray(fc.conf_int(alpha=0.05))
        lower = np.clip(ci[:, 0], 0, None)
        upper = ci[:, 1]
        return pd.DataFrame({"ds": idx, "yhat": yhat, "yhat_lower": lower, "yhat_upper": upper})


class ProphetModel(BaseForecastModel):
    name = "prophet"

    def fit(self, ts: pd.Series) -> "ProphetModel":
        from prophet import Prophet

        self.ts = ts
        self.freq = infer_freq(ts.index)
        df = pd.DataFrame({"ds": ts.index, "y": ts.to_numpy(dtype=float)})
        self.model = Prophet(interval_width=0.95, weekly_seasonality=(self.freq == "D"))
        self.model.fit(df)
        return self

    def predict(self, horizon: int) -> pd.DataFrame:
        future = self.model.make_future_dataframe(periods=horizon, freq=self.freq)
        fc = self.model.predict(future)
        # Prophet ne garantit pas l'ordre de ses lignes : on retrie et on ne garde que le futur.
        fc = fc.sort_values("ds")
        fc_future = fc[fc["ds"] > self.ts.index[-1]].head(horizon)
        yhat = np.clip(fc_future["yhat"].to_numpy(), 0, None)
        lower = np.clip(fc_future["yhat_lower"].to_numpy(), 0, None)
        upper = fc_future["yhat_upper"].to_numpy()
        return pd.DataFrame({"ds": fc_future["ds"].to_numpy(), "yhat": yhat, "yhat_lower": lower, "yhat_upper": upper})


class XGBoostPoissonModel(BaseForecastModel):
    """Les arbres ne prolongent pas une tendance : la prévision reste au dernier niveau
    appris. La dispersion est mesurée sur la fin de l'historique, exclue d'un premier
    ajustement : sur les données d'entraînement, l'intervalle serait bien trop étroit."""

    name = "xgboost_poisson"

    def _new_model(self, y: np.ndarray):
        from xgboost import XGBRegressor

        return XGBRegressor(
            objective="count:poisson", n_estimators=200, max_depth=3, learning_rate=0.08,
            base_score=max(float(y.mean()), 1e-3), n_jobs=1, random_state=42,
        )

    def fit(self, ts: pd.Series) -> "XGBoostPoissonModel":
        self.ts = ts
        self.freq = infer_freq(ts.index)
        self.n = len(ts)
        self.use_seasonal = _uses_seasonal_effects(self.freq, self.n)
        X = _seasonal_design(ts.index, np.arange(self.n), self.freq, self.use_seasonal)
        y = ts.to_numpy(dtype=float)

        split = max(int(self.n * 0.75), 1)
        holdout = self._new_model(y[:split]).fit(X.iloc[:split], y[:split])
        self.alpha = _estimate_alpha(y[split:], holdout.predict(X.iloc[split:]))
        self.model = self._new_model(y).fit(X, y)
        return self

    def predict(self, horizon: int) -> pd.DataFrame:
        idx = future_index(self.ts.index[-1], self.freq, horizon)
        X = _seasonal_design(idx, np.arange(self.n, self.n + horizon), self.freq, self.use_seasonal)
        yhat = np.clip(self.model.predict(X), 0, None)
        lower, upper = _negbin_interval(yhat, self.alpha)
        return pd.DataFrame({"ds": idx, "yhat": yhat, "yhat_lower": lower, "yhat_upper": upper})


class PoissonGLMModel(BaseForecastModel):
    name = "poisson_glm"

    def fit(self, ts: pd.Series) -> "PoissonGLMModel":
        import statsmodels.api as sm

        self.ts = ts
        self.freq = infer_freq(ts.index)
        self.n = len(ts)
        self.use_seasonal = _uses_seasonal_effects(self.freq, self.n)
        t = np.arange(self.n)
        X = sm.add_constant(_seasonal_design(ts.index, t, self.freq, self.use_seasonal)).astype(float)
        self.feature_cols = X.columns
        y = ts.to_numpy(dtype=float)
        self.res = sm.GLM(y, X, family=sm.families.Poisson()).fit()
        return self

    def predict(self, horizon: int) -> pd.DataFrame:
        idx = future_index(self.ts.index[-1], self.freq, horizon)
        t_future = np.arange(self.n, self.n + horizon)
        import statsmodels.api as sm

        X = sm.add_constant(_seasonal_design(idx, t_future, self.freq, self.use_seasonal), has_constant="add")
        X = X.reindex(columns=self.feature_cols, fill_value=0).astype(float)
        yhat = np.clip(self.res.predict(X), 0, None)
        # bande prédictive (pas seulement l'IC de la moyenne) via les quantiles de Poisson autour de yhat
        mu_safe = np.clip(yhat, 1e-6, None)
        lower = stats.poisson.ppf(0.025, mu=mu_safe)
        upper = stats.poisson.ppf(0.975, mu=mu_safe)
        return pd.DataFrame({"ds": idx, "yhat": yhat, "yhat_lower": lower, "yhat_upper": upper})


class NegBinGLMModel(BaseForecastModel):
    """GLM Binomiale négative : même moyenne que Poisson, mais une dispersion alpha
    estimée sur les résidus de Poisson (méthode des moments). Adapté quand la variance
    dépasse la moyenne (mélange de canaux, campagnes ponctuelles...), ce que Poisson ne
    capture pas et qui donnerait des intervalles trop étroits. L'ajustement par IRLS
    évite les arrêts prématurés d'une maximisation jointe des coefficients et d'alpha.
    """

    name = "negbin_glm"

    def fit(self, ts: pd.Series) -> "NegBinGLMModel":
        import statsmodels.api as sm

        self.ts = ts
        self.freq = infer_freq(ts.index)
        self.n = len(ts)
        self.use_seasonal = _uses_seasonal_effects(self.freq, self.n)
        t = np.arange(self.n)
        X = sm.add_constant(_seasonal_design(ts.index, t, self.freq, self.use_seasonal)).astype(float)
        self.feature_cols = X.columns
        y = ts.to_numpy(dtype=float)
        poisson = sm.GLM(y, X, family=sm.families.Poisson()).fit()
        self.alpha = _estimate_alpha(y, np.asarray(poisson.predict(X)))
        self.res = sm.GLM(y, X, family=sm.families.NegativeBinomial(alpha=self.alpha)).fit()
        return self

    def predict(self, horizon: int) -> pd.DataFrame:
        idx = future_index(self.ts.index[-1], self.freq, horizon)
        t_future = np.arange(self.n, self.n + horizon)
        import statsmodels.api as sm

        X = sm.add_constant(_seasonal_design(idx, t_future, self.freq, self.use_seasonal), has_constant="add")
        X = X.reindex(columns=self.feature_cols, fill_value=0).astype(float)
        yhat = np.clip(self.res.predict(X), 0, None)
        mu_safe = np.clip(yhat, 1e-6, None)
        # paramétrage scipy nbinom (variance = mu + alpha*mu^2) : n = 1/alpha, p = n/(n+mu)
        n_param = 1.0 / self.alpha
        p_param = n_param / (n_param + mu_safe)
        lower = stats.nbinom.ppf(0.025, n_param, p_param)
        upper = stats.nbinom.ppf(0.975, n_param, p_param)
        return pd.DataFrame({"ds": idx, "yhat": yhat, "yhat_lower": lower, "yhat_upper": upper})


class _BayesianRegression(BaseForecastModel):
    """Régression bayésienne de comptage : tendance et effets calendaires."""

    def __init__(self, draws: int = 500, tune: int = 500, chains: int = 2, **kwargs):
        super().__init__(**kwargs)
        self.draws, self.tune, self.chains = draws, tune, chains

    def _observation_model(self, pm, mu, y):
        raise NotImplementedError

    def _simulate(self, rng, mu: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    def _scaled_design(self, index: pd.DatetimeIndex, t: np.ndarray) -> np.ndarray:
        X = _seasonal_design(index, (t - self.t_mean) / self.t_std, self.freq, self.use_seasonal)
        return X.to_numpy(dtype=float)

    def fit(self, ts: pd.Series) -> "_BayesianRegression":
        import pymc as pm

        self.ts = ts
        self.freq = infer_freq(ts.index)
        self.n = len(ts)
        self.use_seasonal = _uses_seasonal_effects(self.freq, self.n)
        t = np.arange(self.n)
        self.t_mean, self.t_std = float(t.mean()), float(t.std()) if t.std() > 0 else 1.0
        X = self._scaled_design(ts.index, t)
        y = ts.to_numpy(dtype=float)

        with pm.Model():
            intercept = pm.Normal("intercept", mu=np.log(max(y.mean(), 1.0)), sigma=1.5)
            betas = pm.Normal("betas", mu=0, sigma=0.5, shape=X.shape[1])
            mu = pm.math.exp(intercept + pm.math.dot(X, betas))
            self._observation_model(pm, mu, y)
            self.idata = pm.sample(
                draws=self.draws, tune=self.tune, chains=self.chains,
                cores=1, progressbar=False, random_seed=42,
            )
        posterior = self.idata.posterior
        self.intercepts = posterior["intercept"].values.reshape(-1)
        self.betas = posterior["betas"].values.reshape(-1, X.shape[1])
        self.posterior = posterior
        return self

    def predict(self, horizon: int) -> pd.DataFrame:
        idx = future_index(self.ts.index[-1], self.freq, horizon)
        X = self._scaled_design(idx, np.arange(self.n, self.n + horizon))
        log_mu = self.intercepts[:, None] + self.betas @ X.T
        draws = self._simulate(np.random.default_rng(0), np.exp(np.clip(log_mu, -20, 20)))
        return pd.DataFrame({
            "ds": idx,
            "yhat": draws.mean(axis=0),
            "yhat_lower": np.percentile(draws, 2.5, axis=0),
            "yhat_upper": np.percentile(draws, 97.5, axis=0),
        })


class BayesianPoissonModel(_BayesianRegression):
    name = "bayesian_poisson"

    def _observation_model(self, pm, mu, y):
        pm.Poisson("obs", mu=mu, observed=y)

    def _simulate(self, rng, mu):
        return rng.poisson(mu)


class BayesianNegBinModel(_BayesianRegression):
    """Ajoute une dispersion estimée, utile quand la variance dépasse la moyenne."""

    name = "bayesian_negbin"

    def _observation_model(self, pm, mu, y):
        dispersion = pm.HalfNormal("disp", sigma=20.0)
        pm.NegativeBinomial("obs", mu=mu, alpha=dispersion, observed=y)

    def _simulate(self, rng, mu):
        dispersion = np.clip(self.posterior["disp"].values.reshape(-1)[:, None], 1e-3, None)
        return rng.poisson(rng.gamma(shape=dispersion, scale=mu / dispersion))


# ---------------------------------------------------------------------------
# Registre + orchestrateur
# ---------------------------------------------------------------------------

MODEL_REGISTRY: dict[str, type[BaseForecastModel]] = {
    "baseline": SeasonalBaselineModel,
    "linear": LinearTrendModel,
    "sarima": SarimaModel,
    "prophet": ProphetModel,
    "xgboost_poisson": XGBoostPoissonModel,
    "poisson_glm": PoissonGLMModel,
    "negbin_glm": NegBinGLMModel,
    "bayesian_poisson": BayesianPoissonModel,
    "bayesian_negbin": BayesianNegBinModel,
}

MIN_HISTORY_POINTS = 5


def forecast_series(
    ts: pd.Series, model_name: str, horizon: int, model_kwargs: dict | None = None
) -> tuple[pd.DataFrame, str | None]:
    """Ajuste `model_name` sur `ts` et prévoit `horizon` périodes futures.

    Retourne (forecast_df, note). `note` est non-nul si un repli automatique
    sur la régression linéaire a été nécessaire (modèle non disponible,
    historique trop court, non-convergence...).
    """
    ts = ts.dropna()
    if len(ts) < MIN_HISTORY_POINTS:
        raise ValueError(
            f"Historique trop court ({len(ts)} points) pour produire une prévision "
            f"({MIN_HISTORY_POINTS} minimum). Élargissez la période d'analyse."
        )

    model_cls = MODEL_REGISTRY.get(model_name)
    if model_cls is None:
        raise ValueError(f"Modèle de prévision inconnu : {model_name}")

    if model_name in ("linear", "baseline"):
        model = model_cls(**(model_kwargs or {})).fit(ts)
        return model.predict(horizon), model.model_note

    try:
        model = model_cls(**(model_kwargs or {})).fit(ts)
        forecast_df = model.predict(horizon)
        return forecast_df, model.model_note
    except Exception as e:
        fallback = LinearTrendModel().fit(ts)
        forecast_df = fallback.predict(horizon)
        note = (
            f"Le modèle « {model_name} » a échoué ({type(e).__name__}: {e}) ; "
            f"repli automatique sur une régression linéaire."
        )
        return forecast_df, note
