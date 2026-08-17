"""
data/aggregator.py — Résolution profil/relation + agrégation par granularité.

Point d'attention (leçon du projet call center) : on vérifie systématiquement
l'alignement des index après chaque groupby/merge, et on complète explicitement
les zéros (un profil sans traitement sur une période doit apparaître avec
n_traitements = 0, pas être absent du tableau).

Le portefeuille est STATIQUE (pas d'historisation) : un compte a un seul
profil_partenaire et une seule relation_contractuelle, fixes dans le temps.
Traitements.csv n'a pas de profil directement — il faut le résoudre par une
simple fusion sur account_id (resolve_profile_and_relation), avec repli sur
"Autres" quand account_id est manquant OU quand le compte n'est pas trouvé
dans le portefeuille.
"""

from __future__ import annotations

import pandas as pd

from config import FALLBACK_SEGMENT_LABEL


# ---------------------------------------------------------------------------
# Résolution profil / relation contractuelle (fusion directe + repli Autres)
# ---------------------------------------------------------------------------

def resolve_profile_and_relation(treatments_df: pd.DataFrame, portfolio_df: pd.DataFrame) -> pd.DataFrame:
    """Ajoute profil_partenaire / relation_contractuelle à chaque traitement.

    - account_id renseigné : fusion directe sur account_id (portefeuille
      statique, pas de notion de date). Si le compte n'apparaît pas dans le
      portefeuille, repli sur "Autres" pour les deux colonnes.
    - account_id manquant : repli direct sur "Autres" pour les deux colonnes,
      sans fusion (impossible de relier le traitement à un compte).

    Ajoute aussi `compte_identifie` (bool) : sert d'indicateur de qualité de
    donnée, car le lift/taux de "Autres" mélange alors des comptes réels
    classés "Autres" ET des traitements non rattachables (voir
    compute_data_quality plus bas).
    """
    df = treatments_df.copy()
    df["compte_identifie"] = df["account_id"].notna()

    with_account = df[df["compte_identifie"]].copy()
    with_account["account_id"] = with_account["account_id"].astype("int64")
    without_account = df[~df["compte_identifie"]].copy()

    port = portfolio_df.copy()
    port["account_id"] = port["account_id"].astype("int64")

    joined = with_account.merge(port, on="account_id", how="left")
    # compte non trouvé dans le portefeuille -> Autres
    joined["profil_partenaire"] = joined["profil_partenaire"].fillna(FALLBACK_SEGMENT_LABEL)
    joined["relation_contractuelle"] = joined["relation_contractuelle"].fillna(FALLBACK_SEGMENT_LABEL)

    without_account["profil_partenaire"] = FALLBACK_SEGMENT_LABEL
    without_account["relation_contractuelle"] = FALLBACK_SEGMENT_LABEL

    result = pd.concat([joined, without_account], ignore_index=True, sort=False)
    return result.sort_values("date_contact").reset_index(drop=True)


def compute_data_quality(resolved_df: pd.DataFrame) -> dict:
    """Indicateur de qualité de donnée séparé (décision produit) : la part de
    traitements non identifiables, globalement / par canal, et la part du
    panier "Autres" qui vient de traitements sans compte réel plutôt que de
    vrais comptes classés "Autres" dans le portefeuille."""
    overall_pct = float((~resolved_df["compte_identifie"]).mean() * 100) if len(resolved_df) else 0.0
    by_canal = (
        resolved_df.groupby("type_traitement")["compte_identifie"]
        .apply(lambda s: float((~s).mean() * 100))
        .to_dict()
    )
    autres_mask = resolved_df["profil_partenaire"] == FALLBACK_SEGMENT_LABEL
    n_autres_total = int(autres_mask.sum())
    n_autres_sans_compte = int((autres_mask & ~resolved_df["compte_identifie"]).sum())
    part_autres_sans_compte = (
        float(n_autres_sans_compte / n_autres_total * 100) if n_autres_total > 0 else 0.0
    )
    return {
        "pct_non_identifie_global": overall_pct,
        "pct_non_identifie_par_canal": by_canal,
        "n_autres_total": n_autres_total,
        "n_autres_sans_compte": n_autres_sans_compte,
        "part_autres_sans_compte": part_autres_sans_compte,
    }


def apply_filters(
    resolved_df: pd.DataFrame, portfolio_df: pd.DataFrame, canal: str = "tous", relation: str = "tous"
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Filtre les traitements (résolus) et le portefeuille par canal et/ou
    relation contractuelle, AVANT agrégation. Le filtre de relation s'applique
    aux deux tables (numérateur ET dénominateur), sinon le taux/lift serait
    faux (comparerait un numérateur filtré à un dénominateur non filtré)."""
    t, p = resolved_df, portfolio_df
    if canal != "tous":
        t = t[t["type_traitement"] == canal]
    if relation != "tous":
        t = t[t["relation_contractuelle"] == relation]
        p = p[p["relation_contractuelle"] == relation]
    return t, p


# ---------------------------------------------------------------------------
# Agrégation temporelle
# ---------------------------------------------------------------------------

def build_period_index(start, end, freq_code: str) -> pd.DatetimeIndex:
    """Grille complète de périodes (bornes de début de période) sur [start, end]."""
    return pd.date_range(start=start, end=end, freq=freq_code)


def _floor_to_period(timestamps: pd.Series, freq_code: str) -> pd.Series:
    """Ramène chaque timestamp au début de sa période (jour / semaine / mois)."""
    if freq_code == "D":
        return timestamps.dt.floor("D")
    if freq_code == "MS":
        return timestamps.dt.to_period("M").dt.start_time
    if freq_code == "W-MON":
        return (timestamps - pd.to_timedelta(timestamps.dt.weekday, unit="D")).dt.floor("D")
    raise ValueError(f"Granularité non supportée : {freq_code}")


def _floor_scalar_to_period(ts, freq_code: str) -> pd.Timestamp:
    """Équivalent scalaire de _floor_to_period, pour caler les bornes (start/end)."""
    ts = pd.Timestamp(ts)
    if freq_code == "D":
        return ts.floor("D")
    if freq_code == "MS":
        return ts.to_period("M").start_time
    if freq_code == "W-MON":
        return (ts - pd.Timedelta(days=ts.weekday())).floor("D")
    raise ValueError(f"Granularité non supportée : {freq_code}")


def _period_end(period_start: pd.Timestamp, freq_code: str) -> pd.Timestamp:
    """Dernier instant couvert par une période, selon la granularité."""
    if freq_code == "D":
        return period_start + pd.Timedelta(days=1) - pd.Timedelta(seconds=1)
    if freq_code == "W-MON":
        return period_start + pd.Timedelta(days=7) - pd.Timedelta(seconds=1)
    if freq_code == "MS":
        return period_start + pd.offsets.MonthBegin(1) - pd.Timedelta(seconds=1)
    raise ValueError(f"Granularité non supportée : {freq_code}")


def aggregate_treatments(treatments_df: pd.DataFrame, freq_code: str) -> pd.DataFrame:
    """Agrège les traitements (déjà résolus/filtrés) par (période, profil) :
    n_traitements, n_comptes_distincts."""
    df = treatments_df.copy()
    df["period"] = _floor_to_period(df["date_contact"], freq_code)

    agg = (
        df.groupby(["period", "profil_partenaire"])
        .agg(n_traitements=("traitement_id", "count"), n_comptes_distincts=("account_id", "nunique"))
        .reset_index()
    )
    return agg


def portfolio_size_by_profile(portfolio_df: pd.DataFrame) -> pd.DataFrame:
    """Nombre de comptes par profil (portefeuille statique : une seule valeur
    par profil, valable pour toutes les périodes — pas d'historique à projeter)."""
    return (
        portfolio_df.groupby("profil_partenaire").size().rename("n_accounts").reset_index()
    )


def build_master_table(
    treatments_df: pd.DataFrame,
    portfolio_df: pd.DataFrame,
    freq_code: str,
    start_date=None,
    end_date=None,
    drop_incomplete_last_period: bool = True,
) -> pd.DataFrame:
    """Table maître tidy : period | profil_partenaire | n_traitements | n_comptes_distincts | n_accounts.

    `treatments_df` est supposé déjà résolu (resolve_profile_and_relation) et
    filtré (apply_filters) le cas échéant : ce module ne connaît pas la notion
    de canal/relation, uniquement des colonnes déjà prêtes à agréger.

    n_accounts est une CONSTANTE par profil sur toute la période (portefeuille
    statique) : elle est simplement répétée sur chaque période de la grille.

    Complète les zéros pour toute combinaison (période, profil) sans traitement.
    Si `drop_incomplete_last_period` est vrai (par défaut), la dernière période
    est retirée si elle dépasse la dernière donnée réellement disponible
    (jour/semaine/mois en cours, partiel) — sinon taux/lift/prévisions seraient
    biaisés par un simple effet de troncature.
    """
    if start_date is None:
        start_date = treatments_df["date_contact"].min()
    if end_date is None:
        end_date = treatments_df["date_contact"].max()

    data_limit_ts = treatments_df["date_contact"].max()

    start_date = _floor_scalar_to_period(start_date, freq_code)
    end_date = _floor_scalar_to_period(end_date, freq_code)

    period_index = build_period_index(start_date, end_date, freq_code)

    if drop_incomplete_last_period:
        while len(period_index) > 0 and _period_end(period_index[-1], freq_code) > data_limit_ts:
            period_index = period_index[:-1]

    profiles = sorted(portfolio_df["profil_partenaire"].unique())

    treatments_agg = aggregate_treatments(treatments_df, freq_code)
    portfolio_sizes = portfolio_size_by_profile(portfolio_df)

    full_grid = pd.MultiIndex.from_product(
        [period_index, profiles], names=["period", "profil_partenaire"]
    ).to_frame(index=False)

    master = full_grid.merge(treatments_agg, on=["period", "profil_partenaire"], how="left")
    master = master.merge(portfolio_sizes, on="profil_partenaire", how="left")

    master["n_traitements"] = master["n_traitements"].fillna(0).astype(int)
    master["n_comptes_distincts"] = master["n_comptes_distincts"].fillna(0).astype(int)
    master["n_accounts"] = master["n_accounts"].fillna(0).astype(float)

    assert not master.duplicated(subset=["period", "profil_partenaire"]).any(), (
        "Alignement incorrect après merge : doublons (period, profil_partenaire) détectés."
    )

    return master.sort_values(["profil_partenaire", "period"]).reset_index(drop=True)
