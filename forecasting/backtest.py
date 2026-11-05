"""
forecasting/backtest.py — Validation par repli temporel (walk-forward backtesting).

Principe : pour un historique et un horizon donnés, on découpe la série en
plusieurs replis. Pour chaque repli, on entraîne le modèle uniquement sur les
données antérieures à une date de coupure T, on prévoit les `horizon` périodes
suivantes, et on compare au réel. On répète en reculant la coupure T, jusqu'à
épuiser l'historique disponible.

Ça donne une erreur de prévision réellement hors-échantillon (contrairement au
simple ajustement affiché dans l'onglet Prévision), et permet de vérifier si
l'intervalle de confiance couvre la bonne proportion de valeurs réelles — un
modèle Poisson trop confiant sur une série sur-dispersée aura une couverture
nettement inférieure à 95 %, ce qui est visible ici.

Important : on instancie les modèles DIRECTEMENT (pas via
forecasting.registry.forecast_series), qui masque volontairement les échecs
par un repli silencieux sur la régression linéaire dans l'app (pour ne jamais
planter). Ici on veut au contraire savoir explicitement quand un modèle échoue,
plutôt que de mesurer par erreur les performances de son repli.
"""

from __future__ import annotations

import time

import numpy as np
import pandas as pd

from forecasting.registry import MIN_HISTORY_POINTS, MODEL_REGISTRY


def generate_walk_forward_folds(
    n_total: int, horizon: int, n_folds: int = 5, min_train_size: int | None = None
) -> list[tuple[int, int]]:
    """Retourne une liste de (train_end, test_end) : indices délimitant
    l'entraînement (0:train_end) et le test (train_end:test_end) de chaque repli.

    Le dernier repli est toujours ancré sur les données les plus récentes ;
    les replis précédents sont étalés en arrière, autant que l'historique
    disponible le permet. Si l'historique est trop court pour un seul repli
    robuste, retourne une liste vide.
    """
    if min_train_size is None:
        min_train_size = max(MIN_HISTORY_POINTS, n_total // 2)

    last_train_end = n_total - horizon
    if last_train_end < min_train_size:
        return []

    available_range = last_train_end - min_train_size
    actual_n_folds = min(n_folds, available_range + 1)
    if actual_n_folds <= 1:
        train_ends = [last_train_end]
    else:
        step = available_range / (actual_n_folds - 1)
        train_ends = sorted({int(round(min_train_size + i * step)) for i in range(actual_n_folds)})

    return [(te, te + horizon) for te in train_ends]


def backtest_model(ts: pd.Series, model_name: str, horizon: int, n_folds: int = 5) -> dict:
    """Backteste UN modèle sur plusieurs replis glissants.

    Retourne un dict avec les métriques agrégées : mae, wape, rmse, bias (erreur
    signée moyenne — positif = le modèle surestime), coverage (% de valeurs
    réelles dans l'intervalle de confiance ; cible ~95 %), et des informations
    de diagnostic (nombre de replis utilisés/échoués, temps moyen d'ajustement).
    """
    ts = ts.dropna()
    folds = generate_walk_forward_folds(len(ts), horizon, n_folds)
    label_base = {"model": model_name}

    if not folds:
        return {
            **label_base, "n_folds": 0, "n_failures": 0, "mae": np.nan, "wape": np.nan, "rmse": np.nan,
            "bias": np.nan, "coverage": np.nan, "avg_fit_time_s": np.nan,
            "error": "Historique trop court pour un backtest robuste.",
        }

    model_cls = MODEL_REGISTRY[model_name]
    all_abs_err, all_sq_err, all_signed_err, all_in_interval, all_actual = [], [], [], [], []
    fit_times = []
    n_failures = 0

    for train_end, test_end in folds:
        train_ts = ts.iloc[:train_end]
        actual = ts.iloc[train_end:test_end].to_numpy(dtype=float)
        h = len(actual)
        if h == 0:
            continue

        t0 = time.time()
        try:
            model = model_cls().fit(train_ts)
            forecast_df = model.predict(h)
        except Exception:
            n_failures += 1
            continue
        fit_times.append(time.time() - t0)

        yhat = forecast_df["yhat"].to_numpy()[:h]
        lower = forecast_df["yhat_lower"].to_numpy()[:h]
        upper = forecast_df["yhat_upper"].to_numpy()[:h]
        err = yhat - actual

        all_actual.append(actual)
        all_abs_err.append(np.abs(err))
        all_sq_err.append(err ** 2)
        all_signed_err.append(err)
        all_in_interval.append((actual >= lower) & (actual <= upper))

    if not all_abs_err:
        return {
            **label_base, "n_folds": 0, "n_failures": n_failures, "mae": np.nan, "wape": np.nan, "rmse": np.nan,
            "bias": np.nan, "coverage": np.nan, "avg_fit_time_s": np.nan,
            "error": "Tous les replis ont échoué pour ce modèle.",
        }

    abs_err = np.concatenate(all_abs_err)
    sq_err = np.concatenate(all_sq_err)
    signed_err = np.concatenate(all_signed_err)
    in_interval = np.concatenate(all_in_interval)

    total_actual = np.concatenate(all_actual).sum()
    return {
        **label_base,
        "n_folds": len(all_abs_err),
        "n_failures": n_failures,
        "mae": float(np.mean(abs_err)),
        "wape": float(abs_err.sum() / total_actual * 100) if total_actual else np.nan,
        "rmse": float(np.sqrt(np.mean(sq_err))),
        "bias": float(np.mean(signed_err)),
        "coverage": float(np.mean(in_interval) * 100),
        "avg_fit_time_s": float(np.mean(fit_times)),
        "error": None,
    }


def backtest_all_models(ts: pd.Series, horizon: int, n_folds: int = 5) -> pd.DataFrame:
    """Backteste tous les modèles du registre, classés par MAE croissante. La colonne
    `gain_vs_reference` donne le gain de MAE en % par rapport au modèle de référence :
    s'il est faible, le modèle complexe ne se justifie pas."""
    rows = [backtest_model(ts, name, horizon, n_folds) for name in MODEL_REGISTRY]
    df = pd.DataFrame(rows)
    reference = df.loc[df["model"] == "baseline", "mae"].iloc[0]
    df["gain_vs_reference"] = (1 - df["mae"] / reference) * 100
    return df.sort_values("mae", na_position="last").reset_index(drop=True)
