"""
metrics/lift.py — Indice de représentation (lift) + intervalle de confiance.

    lift(profil) = (% de traitements générés par le profil) / (% du profil dans le portefeuille)

Deux méthodes d'intervalle de confiance sont proposées :

1. Approximation binomiale (par défaut, rapide) : le lift est traité comme un
   risque relatif (relative risk) entre deux proportions indépendantes — la part
   de traitements du profil et sa part dans le portefeuille — via la méthode log
   de Katz, avec correction de continuité (+0.5) quand un effectif est nul.
2. Bootstrap (plus lent, plus robuste) : ré-échantillonnage des traitements
   individuels et des comptes du portefeuille sur une période donnée, à utiliser
   ponctuellement (une période, un ou plusieurs profils) plutôt que sur toute la
   série pour rester réactif dans l'app.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats


def add_lift(master: pd.DataFrame, rate_definition: str) -> pd.DataFrame:
    """Ajoute pct_traitements, pct_portfolio, lift, et les effectifs bruts
    utilisés (numerator / total_numerator / total_accounts), par (période, profil)."""
    df = master.copy()
    numerator_col = rate_definition

    totals = (
        df.groupby("period")
        .agg(total_numerator=(numerator_col, "sum"), total_accounts=("n_accounts", "sum"))
        .reset_index()
    )
    df = df.merge(totals, on="period", how="left")

    df["pct_traitements"] = np.where(
        df["total_numerator"] > 0, df[numerator_col] / df["total_numerator"], np.nan
    )
    df["pct_portfolio"] = np.where(
        df["total_accounts"] > 0, df["n_accounts"] / df["total_accounts"], np.nan
    )
    df["lift"] = np.where(
        (df["pct_portfolio"] > 0) & df["pct_portfolio"].notna(),
        df["pct_traitements"] / df["pct_portfolio"],
        np.nan,
    )
    return df


def add_lift_ci_binomial(df: pd.DataFrame, rate_definition: str, ci_level: float = 0.95) -> pd.DataFrame:
    """Ajoute lift_ci_low / lift_ci_high via la méthode log de Katz (risque relatif).

    a = numérateur du profil (traitements ou comptes distincts) sur la période
    n1 = numérateur total (tous profils) sur la période
    b = nombre de comptes du profil
    n2 = nombre de comptes total sur la période
    """
    df = df.copy()
    z = stats.norm.ppf(1 - (1 - ci_level) / 2)

    a = df[rate_definition].to_numpy(dtype=float)
    n1 = df["total_numerator"].to_numpy(dtype=float)
    b = df["n_accounts"].to_numpy(dtype=float)
    n2 = df["total_accounts"].to_numpy(dtype=float)

    valid = (n1 > 0) & (n2 > 0)

    a_c = np.where(a == 0, a + 0.5, a)
    b_c = np.where(b == 0, b + 0.5, b)
    n1_c = np.where(a == 0, n1 + 1, n1)
    n2_c = np.where(b == 0, n2 + 1, n2)

    with np.errstate(divide="ignore", invalid="ignore"):
        p1 = a_c / n1_c
        p2 = b_c / n2_c
        rr = p1 / p2
        se_log = np.sqrt(1 / a_c - 1 / n1_c + 1 / b_c - 1 / n2_c)
        log_rr = np.log(rr)
        low = np.exp(log_rr - z * se_log)
        high = np.exp(log_rr + z * se_log)

    df["lift_ci_low"] = np.where(valid, low, np.nan)
    df["lift_ci_high"] = np.where(valid, high, np.nan)
    df["lift_ci_method"] = "binomial"
    return df


def bootstrap_lift_ci(
    treatments_period_df: pd.DataFrame,
    portfolio_period_df: pd.DataFrame,
    profil_partenaire: str,
    rate_definition: str,
    n_bootstrap: int = 300,
    ci_level: float = 0.95,
    seed: int = 123,
) -> tuple[float, float, float]:
    """Bootstrap du lift pour UN (période, profil) donné, à partir des données brutes.

    treatments_period_df : traitements (résolus) déjà filtrés sur la période concernée
    portfolio_period_df : comptes du portefeuille (statique, pas de notion de snapshot)

    Retourne (lift_median, ci_low, ci_high). À utiliser ponctuellement (période/profil
    sélectionnés), pas sur l'ensemble de la série, pour rester réactif dans l'app.
    """
    rng = np.random.default_rng(seed)

    if rate_definition == "n_comptes_distincts":
        base = treatments_period_df.drop_duplicates("account_id")
    else:
        base = treatments_period_df
    treatment_labels = base["profil_partenaire"].to_numpy()
    account_labels = portfolio_period_df["profil_partenaire"].to_numpy()

    n1, n2 = len(treatment_labels), len(account_labels)
    if n1 == 0 or n2 == 0:
        return np.nan, np.nan, np.nan

    lifts = np.empty(n_bootstrap)
    for i in range(n_bootstrap):
        samp_treatments = rng.choice(treatment_labels, size=n1, replace=True)
        samp_accounts = rng.choice(account_labels, size=n2, replace=True)
        p1 = np.mean(samp_treatments == profil_partenaire)
        p2 = np.mean(samp_accounts == profil_partenaire)
        lifts[i] = p1 / p2 if p2 > 0 else np.nan

    lifts = lifts[~np.isnan(lifts)]
    if len(lifts) == 0:
        return np.nan, np.nan, np.nan

    alpha = 1 - ci_level
    low, high = np.percentile(lifts, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return float(np.median(lifts)), float(low), float(high)


def apply_min_volume_mask(df: pd.DataFrame, rate_definition: str, min_accounts: int, min_calls: int) -> pd.DataFrame:
    """Garde-fou n°1 : marque comme non fiables (`is_reliable=False`) les lignes
    dont le volume est insuffisant pour interpréter un lift ou un taux. On ne
    supprime pas la ligne (elle reste visible mais grisée/filtrable dans l'UI).
    """
    df = df.copy()
    df["is_reliable"] = (df["n_accounts"] >= min_accounts) & (df[rate_definition] >= min_calls)
    return df
