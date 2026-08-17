"""
forecasting/registry.py — MODEL_REGISTRY réutilisable, appliqué par profil.

Reprend le pattern du projet call center (BaseForecastModel + MODEL_REGISTRY),
mais chaque modèle est ici entraîné sur la série d'UN profil (ou sur le volume
global), horizon et granularité étant paramétrés depuis la sidebar.

Points de vigilance repris du projet précédent (section 5.5 du cadrage) :
- Prophet ne garantit pas l'ordre de ses lignes en sortie : on retrie
  systématiquement par `ds` et on ne garde que les dates strictement futures.
- Chaque modèle est isolé par un try/except au niveau de `forecast_series` :
  si un modèle échoue (historique trop court, non-convergence...), on bascule
  automatiquement sur la régression linéaire plutôt que de faire planter l'app.
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


def _seasonal_design(index: pd.DatetimeIndex, t: np.ndarray, freq: str, use_seasonal: bool = True) -> pd.DataFrame:
    """Features calendaires simples (indicatrices) selon la granularité.

    Pour la granularité mensuelle, les indicatrices de mois ne sont ajoutées
    que si l'historique couvre au moins 2 cycles annuels (24 points) : sinon
    chaque mois n'aurait qu'une seule observation historique, et le coefficient
    correspondant surapprendrait ce point unique plutôt que d'estimer une vraie
    saisonnalité.
    """
    df = pd.DataFrame({"t": t.astype(float)})
    if not use_seasonal:
        return df
    if freq == "D":
        df["dow"] = index.dayofweek
        df = pd.get_dummies(df, columns=["dow"], drop_first=True, dtype=float)
    else:
        df["month"] = index.month
        df = pd.get_dummies(df, columns=["month"], drop_first=True, dtype=float)
    return df


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

class LinearTrendModel(BaseForecastModel):
    name = "linear"

    def fit(self, ts: pd.Series) -> "LinearTrendModel":
        self.ts = ts
        self.freq = infer_freq(ts.index)
        self.n = len(ts)
        x = np.arange(self.n).reshape(-1, 1)
        y = ts.to_numpy(dtype=float)
        self.model = LinearRegression().fit(x, y)
        self.resid_std = float(np.std(y - self.model.predict(x)))
        return self

    def predict(self, horizon: int) -> pd.DataFrame:
        idx = future_index(self.ts.index[-1], self.freq, horizon)
        x_future = np.arange(self.n, self.n + horizon).reshape(-1, 1)
        yhat = np.clip(self.model.predict(x_future), 0, None)
        z = 1.96
        lower = np.clip(yhat - z * self.resid_std, 0, None)
        upper = yhat + z * self.resid_std
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
        # Bug déjà rencontré (section 5.5) : Prophet ne garantit pas l'ordre en sortie.
        # On retrie explicitement par ds et on ne garde que les dates futures.
        fc = fc.sort_values("ds")
        fc_future = fc[fc["ds"] > self.ts.index[-1]].head(horizon)
        yhat = np.clip(fc_future["yhat"].to_numpy(), 0, None)
        lower = np.clip(fc_future["yhat_lower"].to_numpy(), 0, None)
        upper = fc_future["yhat_upper"].to_numpy()
        return pd.DataFrame({"ds": fc_future["ds"].to_numpy(), "yhat": yhat, "yhat_lower": lower, "yhat_upper": upper})


class XGBoostPoissonModel(BaseForecastModel):
    name = "xgboost_poisson"

    def fit(self, ts: pd.Series) -> "XGBoostPoissonModel":
        from xgboost import XGBRegressor

        self.ts = ts
        self.freq = infer_freq(ts.index)
        self.n = len(ts)
        self.use_seasonal = True if self.freq == "D" else (self.n >= 24)
        t = np.arange(self.n)
        X = _seasonal_design(ts.index, t, self.freq, self.use_seasonal)
        self.feature_cols = X.columns
        y = ts.to_numpy(dtype=float)

        self.model = XGBRegressor(
            objective="count:poisson", n_estimators=200, max_depth=3,
            learning_rate=0.08, base_score=max(float(y.mean()), 1e-3),
        )
        self.model.fit(X, y)
        self.resid_std = float(np.std(y - self.model.predict(X)))
        return self

    def predict(self, horizon: int) -> pd.DataFrame:
        idx = future_index(self.ts.index[-1], self.freq, horizon)
        t_future = np.arange(self.n, self.n + horizon)
        X = _seasonal_design(idx, t_future, self.freq, self.use_seasonal).reindex(columns=self.feature_cols, fill_value=0)
        yhat = np.clip(self.model.predict(X), 0, None)
        z = 1.96
        lower = np.clip(yhat - z * self.resid_std, 0, None)
        upper = yhat + z * self.resid_std
        return pd.DataFrame({"ds": idx, "yhat": yhat, "yhat_lower": lower, "yhat_upper": upper})


class PoissonGLMModel(BaseForecastModel):
    name = "poisson_glm"

    def fit(self, ts: pd.Series) -> "PoissonGLMModel":
        import statsmodels.api as sm

        self.ts = ts
        self.freq = infer_freq(ts.index)
        self.n = len(ts)
        self.use_seasonal = True if self.freq == "D" else (self.n >= 24)
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
    """GLM Binomiale négative (fréquentiste) : ajoute un paramètre de dispersion
    estimé par MLE, adapté quand la variance dépasse la moyenne (mélange de
    canaux, campagnes ponctuelles...) — ce que Poisson ne capture pas et qui
    conduirait à des intervalles de confiance trop étroits.
    """

    name = "negbin_glm"

    def fit(self, ts: pd.Series) -> "NegBinGLMModel":
        import statsmodels.api as sm

        self.ts = ts
        self.freq = infer_freq(ts.index)
        self.n = len(ts)
        self.use_seasonal = True if self.freq == "D" else (self.n >= 24)
        t = np.arange(self.n)
        X = sm.add_constant(_seasonal_design(ts.index, t, self.freq, self.use_seasonal)).astype(float)
        self.feature_cols = X.columns
        y = ts.to_numpy(dtype=float)
        self.res = sm.NegativeBinomial(y, X).fit(disp=False)
        self.alpha = max(float(self.res.params.iloc[-1]), 1e-6)
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


class BayesianPoissonModel(BaseForecastModel):
    """Régression de Poisson bayésienne simplifiée (tendance log-linéaire).

    Volontairement légère (peu de draws/tune, 2 chaînes, 1 cœur) pour rester
    utilisable de façon interactive dans l'app. Ce n'est pas un modèle
    saisonnier complet : à considérer comme une estimation d'ordre de grandeur
    avec incertitude, pas comme une prévision fine.
    """

    name = "bayesian_poisson"

    def __init__(self, draws: int = 400, tune: int = 400, chains: int = 2, **kwargs):
        super().__init__(**kwargs)
        self.draws, self.tune, self.chains = draws, tune, chains

    def fit(self, ts: pd.Series) -> "BayesianPoissonModel":
        import pymc as pm

        self.ts = ts
        self.freq = infer_freq(ts.index)
        self.n = len(ts)
        t = np.arange(self.n)
        self.t_mean = float(t.mean())
        self.t_std = float(t.std()) if t.std() > 0 else 1.0
        t_scaled = (t - self.t_mean) / self.t_std
        y = ts.to_numpy(dtype=float)

        with pm.Model():
            alpha = pm.Normal("alpha", mu=np.log(max(y.mean(), 1.0)), sigma=1.5)
            beta = pm.Normal("beta", mu=0, sigma=0.5)
            mu = pm.math.exp(alpha + beta * t_scaled)
            pm.Poisson("obs", mu=mu, observed=y)
            self.idata = pm.sample(
                draws=self.draws, tune=self.tune, chains=self.chains,
                cores=1, progressbar=False, random_seed=42,
            )

        self.alpha_samples = self.idata.posterior["alpha"].values.reshape(-1)
        self.beta_samples = self.idata.posterior["beta"].values.reshape(-1)
        return self

    def predict(self, horizon: int) -> pd.DataFrame:
        idx = future_index(self.ts.index[-1], self.freq, horizon)
        t_future = np.arange(self.n, self.n + horizon)
        t_scaled = (t_future - self.t_mean) / self.t_std

        log_mu = self.alpha_samples[:, None] + self.beta_samples[:, None] * t_scaled[None, :]
        mu = np.exp(np.clip(log_mu, -20, 20))

        rng = np.random.default_rng(0)
        draws = rng.poisson(mu)
        yhat = draws.mean(axis=0)
        lower = np.percentile(draws, 2.5, axis=0)
        upper = np.percentile(draws, 97.5, axis=0)
        return pd.DataFrame({"ds": idx, "yhat": yhat, "yhat_lower": lower, "yhat_upper": upper})


class BayesianNegBinModel(BaseForecastModel):
    """Version 'sur-dispersion' du modèle bayésien précédent : ajoute un
    paramètre de dispersion (alpha) avec un prior faiblement informatif, pour
    ne pas sous-estimer l'incertitude quand les données mélangent plusieurs
    canaux/régimes (là où Poisson serait trop optimiste).
    """

    name = "bayesian_negbin"

    def __init__(self, draws: int = 400, tune: int = 400, chains: int = 2, **kwargs):
        super().__init__(**kwargs)
        self.draws, self.tune, self.chains = draws, tune, chains

    def fit(self, ts: pd.Series) -> "BayesianNegBinModel":
        import pymc as pm

        self.ts = ts
        self.freq = infer_freq(ts.index)
        self.n = len(ts)
        t = np.arange(self.n)
        self.t_mean = float(t.mean())
        self.t_std = float(t.std()) if t.std() > 0 else 1.0
        t_scaled = (t - self.t_mean) / self.t_std
        y = ts.to_numpy(dtype=float)

        with pm.Model():
            alpha_trend = pm.Normal("alpha_trend", mu=np.log(max(y.mean(), 1.0)), sigma=1.5)
            beta = pm.Normal("beta", mu=0, sigma=0.5)
            mu = pm.math.exp(alpha_trend + beta * t_scaled)
            disp = pm.HalfNormal("disp", sigma=5.0)  # grand -> proche de Poisson, petit -> forte sur-dispersion
            pm.NegativeBinomial("obs", mu=mu, alpha=disp, observed=y)
            self.idata = pm.sample(
                draws=self.draws, tune=self.tune, chains=self.chains,
                cores=1, progressbar=False, random_seed=42,
            )

        self.alpha_trend_samples = self.idata.posterior["alpha_trend"].values.reshape(-1)
        self.beta_samples = self.idata.posterior["beta"].values.reshape(-1)
        self.disp_samples = self.idata.posterior["disp"].values.reshape(-1)
        return self

    def predict(self, horizon: int) -> pd.DataFrame:
        idx = future_index(self.ts.index[-1], self.freq, horizon)
        t_future = np.arange(self.n, self.n + horizon)
        t_scaled = (t_future - self.t_mean) / self.t_std

        log_mu = self.alpha_trend_samples[:, None] + self.beta_samples[:, None] * t_scaled[None, :]
        mu = np.exp(np.clip(log_mu, -20, 20))
        disp = np.clip(self.disp_samples[:, None], 1e-3, None)

        rng = np.random.default_rng(0)
        # Binomiale négative via mélange Poisson-Gamma (robuste à un alpha non entier) :
        # lambda ~ Gamma(shape=disp, scale=mu/disp), puis y ~ Poisson(lambda)
        lam = rng.gamma(shape=disp, scale=mu / disp)
        draws = rng.poisson(lam)
        yhat = draws.mean(axis=0)
        lower = np.percentile(draws, 2.5, axis=0)
        upper = np.percentile(draws, 97.5, axis=0)
        return pd.DataFrame({"ds": idx, "yhat": yhat, "yhat_lower": lower, "yhat_upper": upper})


# ---------------------------------------------------------------------------
# Registre + orchestrateur
# ---------------------------------------------------------------------------

MODEL_REGISTRY: dict[str, type[BaseForecastModel]] = {
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

    if model_name == "linear":
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
