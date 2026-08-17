"""
metrics/call_rate.py — Taux de traitement par profil.

Le taux peut être défini de deux façons (contrôle interface) :
- nombre de traitements / nombre de comptes du profil
- nombre de comptes distincts traités / nombre de comptes du profil

Dans les deux cas, le taux n'est calculable que si le profil a des comptes sur
la période (n_accounts > 0) : on retourne NaN sinon, plutôt qu'une division par
zéro silencieuse.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def add_call_rate(master: pd.DataFrame, rate_definition: str) -> pd.DataFrame:
    """Ajoute une colonne `taux` à la table maître.

    rate_definition : 'n_traitements' ou 'n_comptes_distincts' (voir config.RATE_DEFINITIONS)
    """
    if rate_definition not in ("n_traitements", "n_comptes_distincts"):
        raise ValueError("rate_definition doit être 'n_traitements' ou 'n_comptes_distincts'")

    df = master.copy()
    numerator = df[rate_definition]
    denom = df["n_accounts"]
    df["taux"] = np.where(denom > 0, numerator / denom, np.nan)
    df["rate_numerator"] = numerator
    return df


def evolution_table(df: pd.DataFrame, profil_ids: list[str]) -> pd.DataFrame:
    """Sous-ensemble trié, prêt pour un graphique d'évolution temporelle."""
    subset = df[df["profil_partenaire"].isin(profil_ids)].copy()
    return subset.sort_values(["profil_partenaire", "period"])


def period_over_period_change(df: pd.DataFrame, value_col: str = "taux") -> pd.DataFrame:
    """Ajoute la variation période sur période (absolue et en %), par profil."""
    df = df.sort_values(["profil_partenaire", "period"]).copy()
    df["delta_abs"] = df.groupby("profil_partenaire")[value_col].diff()
    prev = df.groupby("profil_partenaire")[value_col].shift(1)
    df["delta_pct"] = np.where(
        (prev.notna()) & (prev != 0), df["delta_abs"] / prev * 100, np.nan
    )
    return df
