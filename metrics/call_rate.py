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
