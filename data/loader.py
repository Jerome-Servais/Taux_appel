"""
data/loader.py — Chargement des données brutes.

Deux sources possibles :
1. Générateur de données synthétiques (démonstration, aucune donnée réelle requise)
2. Import CSV respectant le schéma réel (traitements + portefeuille) :
   - Traitements.csv  : traitement_id | account_id (parfois vide) | date_contact | type_traitement (appel/email)
   - Portefeuille.csv : account_id | profil_partenaire | relation_contractuelle

Portefeuille simplifié (pas d'historisation) : chaque compte a un seul profil
et une seule relation contractuelle, fixes dans le temps — pas de
`date_snapshot`, pas de jointure "as of", juste une fusion directe sur
`account_id`.

Point de vigilance : `account_id` est parfois vide dans Traitements.csv
(impossible de le relier au portefeuille). Dans ce cas, le profil ET la
relation contractuelle sont fixés à "Autres" — la résolution proprement dite
(fusion avec le portefeuille + repli sur "Autres") se fait dans
data/aggregator.py::resolve_profile_and_relation, une fois les deux tables
chargées ici.
"""

from __future__ import annotations

import re
import unicodedata

import numpy as np
import pandas as pd

from config import (
    DATE_CONTACT_FORMAT,
    DEMO_END_DATE,
    DEMO_START_DATE,
    FALLBACK_SEGMENT_LABEL,
    PORTFOLIO_ALIASES,
    TRAITEMENTS_ALIASES,
)


# ---------------------------------------------------------------------------
# Génération de données synthétiques
# ---------------------------------------------------------------------------

# 5 profils au total, "Autres" INCLUS (schéma réel) : les comptes "Autres" sont
# des comptes réels non classés, à distinguer des traitements sans compte
# identifié (qui sont eux aussi rattachés à "Autres", mais sans compte réel
# derrière — voir l'indicateur de qualité de donnée dans l'app).
#
# "Digital" a volontairement un taux d'appel bas mais un taux d'email élevé
# (profil qui préfère l'email) : utile pour montrer qu'agréger tous les
# canaux peut masquer un comportement propre à un canal.
DEMO_PROFILES = {
    "Standard": {"n_accounts": 6000, "appel_rate": 0.014, "email_rate": 0.010, "relation_autres_share": 0.10},
    "Premium":  {"n_accounts": 1500, "appel_rate": 0.030, "email_rate": 0.020, "relation_autres_share": 0.05},
    "Digital":  {"n_accounts": 2500, "appel_rate": 0.006, "email_rate": 0.022, "relation_autres_share": 0.08},
    "Fragile":  {"n_accounts": 120,  "appel_rate": 0.020, "email_rate": 0.015, "relation_autres_share": 0.15},
    FALLBACK_SEGMENT_LABEL: {"n_accounts": 900, "appel_rate": 0.025, "email_rate": 0.018, "relation_autres_share": 1.0},
}

# Taux de traitements sans account_id, par canal (une donnée d'email non
# authentifiée se relie moins souvent à un compte qu'un appel pris par un agent).
MISSING_ACCOUNT_RATE = {"appel": 0.03, "email": 0.09}


def generate_synthetic_portfolio(profiles: dict | None = None, seed: int = 42) -> pd.DataFrame:
    """Génère un portefeuille statique : un profil et une relation
    contractuelle par compte, fixes (pas d'historique, pas de migration)."""
    rng = np.random.default_rng(seed)
    profiles = profiles or DEMO_PROFILES

    rows = []
    account_id = 0
    for pname, cfg in profiles.items():
        for _ in range(cfg["n_accounts"]):
            relation = "Autres" if rng.random() < cfg["relation_autres_share"] else "Client actif"
            rows.append((account_id, pname, relation))
            account_id += 1

    return pd.DataFrame(rows, columns=["account_id", "profil_partenaire", "relation_contractuelle"])


def generate_synthetic_treatments(
    portfolio_df: pd.DataFrame,
    start_date=DEMO_START_DATE,
    end_date=DEMO_END_DATE,
    profiles: dict | None = None,
    seed: int = 7,
) -> pd.DataFrame:
    """Génère la table brute des traitements (1 ligne = 1 appel ou 1 email).

    Ingrédients volontaires pour pouvoir tester les fonctionnalités :
    - rupture de tendance sur le canal APPEL uniquement du profil "Digital"
      (démontre qu'agréger "tous canaux" peut diluer un signal propre à un canal)
    - bruit relatif fort sur "Fragile" (petit volume -> forte variance)
    - pics de campagne email périodiques -> sur-dispersion réaliste du canal email
      (variance >> moyenne : Poisson devient inadapté, cf. Binomiale négative)
    - une fraction de traitements sans account_id, selon le canal
    """
    rng = np.random.default_rng(seed)
    profiles = profiles or DEMO_PROFILES
    dates = pd.date_range(start_date, end_date, freq="D")

    # portefeuille statique : la composition par profil ne varie pas dans le temps
    accounts_by_profile = portfolio_df.groupby("profil_partenaire")["account_id"].apply(list).to_dict()

    breakpoint_date = pd.Timestamp(start_date) + (pd.Timestamp(end_date) - pd.Timestamp(start_date)) * 2 // 3

    # quelques jours de campagne email par trimestre environ, tirés à l'avance
    n_campaigns = max(1, int(len(dates) / 90))
    campaign_days = set(rng.choice(dates, size=n_campaigns, replace=False))

    rows = []
    treatment_id = 0
    for d in dates:
        weekday = d.weekday()
        weekend_factor_appel = 0.55 if weekday >= 5 else 1.0
        weekend_factor_email = 0.85 if weekday >= 5 else 1.0
        campaign_multiplier = 4.0 if d in campaign_days else 1.0

        for pname, cfg in profiles.items():
            accounts = accounts_by_profile.get(pname, [])
            n_accounts = len(accounts)
            if n_accounts == 0:
                continue

            appel_rate = cfg["appel_rate"]
            email_rate = cfg["email_rate"]

            if pname == "Digital" and d >= breakpoint_date:
                appel_rate = appel_rate * 4.2  # rupture nette, canal APPEL uniquement
            if pname == "Fragile":
                noise = rng.uniform(0.4, 1.8)
                appel_rate, email_rate = appel_rate * noise, email_rate * noise

            lam_appel = n_accounts * appel_rate * weekend_factor_appel
            lam_email = n_accounts * email_rate * weekend_factor_email * campaign_multiplier

            for canal, lam in (("appel", lam_appel), ("email", lam_email)):
                n_today = rng.poisson(lam)
                if n_today == 0:
                    continue
                callers = rng.choice(accounts, size=n_today, replace=True)
                seconds_offsets = rng.integers(0, 24 * 3600, size=n_today)
                missing_mask = rng.random(n_today) < MISSING_ACCOUNT_RATE[canal]
                for aid, offset, is_missing in zip(callers, seconds_offsets, missing_mask):
                    ts = pd.Timestamp(d) + pd.Timedelta(seconds=int(offset))
                    acc = None if is_missing else int(aid)
                    rows.append((treatment_id, acc, ts, canal))
                    treatment_id += 1

    treatments_df = pd.DataFrame(rows, columns=["traitement_id", "account_id", "date_contact", "type_traitement"])
    treatments_df["account_id"] = treatments_df["account_id"].astype("Int64")  # nullable int
    return treatments_df.sort_values("date_contact").reset_index(drop=True)


def load_demo_data():
    """Génère et retourne (treatments_df, portfolio_df) de démonstration."""
    portfolio_df = generate_synthetic_portfolio()
    treatments_df = generate_synthetic_treatments(portfolio_df)
    return treatments_df, portfolio_df


# ---------------------------------------------------------------------------
# Import de données réelles (CSV) — colonnes normalisées avec table d'alias
# ---------------------------------------------------------------------------

def _normalize_colname(col: str) -> str:
    col = str(col).strip().lower()
    col = unicodedata.normalize("NFKD", col).encode("ascii", "ignore").decode("ascii")
    col = re.sub(r"[^a-z0-9]+", "_", col).strip("_")
    return col


# Repli par mots-clés si aucun alias exact ne correspond (config.py garde la
# liste principale) : chaque champ canonique peut être retrouvé si le nom de
# colonne normalisé contient tous les mots-clés d'au moins un des groupes
# ci-dessous. Sert de filet de sécurité pour des en-têtes non anticipés.
_FALLBACK_KEYWORDS: dict[str, list[list[str]]] = {
    "traitement_id": [["traitement", "id"], ["treatment", "id"]],
    "account_id": [["account", "id"], ["compte", "id"]],
    "date_contact": [["date"]],
    "type_traitement": [["type"], ["canal"]],
    "profil_partenaire": [["profil"]],
    "relation_contractuelle": [["relation"]],
}


def _map_columns(df: pd.DataFrame, alias_map: dict[str, list[str]], label: str) -> pd.DataFrame:
    """Renomme les colonnes du fichier importé vers le schéma interne canonique,
    en tolérant les variantes de casse/accents/espaces (voir config.py), avec un
    repli par mots-clés (_FALLBACK_KEYWORDS) si aucun alias exact ne correspond."""
    normalized_to_original = {_normalize_colname(c): c for c in df.columns}
    rename, missing = {}, []
    used_originals: set[str] = set()

    for canonical, aliases in alias_map.items():
        found = next(
            (normalized_to_original[a] for a in aliases
             if a in normalized_to_original and normalized_to_original[a] not in used_originals),
            None,
        )
        if found is None:
            for norm, orig in normalized_to_original.items():
                if orig in used_originals:
                    continue
                keyword_groups = _FALLBACK_KEYWORDS.get(canonical, [])
                if any(all(kw in norm for kw in group) for group in keyword_groups):
                    found = orig
                    break
        if found is None:
            missing.append(canonical)
        else:
            rename[found] = canonical
            used_originals.add(found)

    if missing:
        raise ValueError(
            f"Colonnes manquantes dans le fichier {label} : {missing}. "
            f"Colonnes trouvées dans le fichier : {list(df.columns)}."
        )
    return df.rename(columns=rename)


def _canonicalize_fallback_label(series: pd.Series) -> pd.Series:
    """Uniformise l'écriture de la catégorie "Autres" (casse/accents), pour que
    les traitements sans compte identifié (repli programmatique) et les vrais
    comptes "Autres" du portefeuille se retrouvent bien dans la même catégorie."""
    normalized = series.astype(str).str.strip().str.lower()
    is_autres = normalized.isin(["autres", "autre", "other", "others"])
    return series.mask(is_autres, FALLBACK_SEGMENT_LABEL)


def _canonicalize_categorical(series: pd.Series) -> pd.Series:
    """Neutralise les variantes de casse/espaces d'une même catégorie
    (ex. "Premium", "premium ", "PREMIUM" -> une seule catégorie).

    Sans ça, une incohérence de saisie côté système source fragmente un même
    profil réel en plusieurs sous-catégories distinctes pour un groupby — leur
    volume individuel semble alors bien plus faible que le vrai total.

    Pour chaque valeur normalisée (espaces retirés, minuscule), la graphie la
    plus fréquente est choisie comme forme d'affichage canonique.
    """
    stripped = series.astype(str).str.strip()
    key = stripped.str.lower()
    canonical_map = stripped.groupby(key).agg(lambda x: x.value_counts().idxmax())
    return key.map(canonical_map)


def _parse_date_contact(series: pd.Series) -> pd.Series:
    """Parse date_contact au format JJ/MM/AAAA HH:mm:ss (confirmé par
    l'utilisateur). Sans format explicite, pandas suppose par défaut le format
    américain (mois/jour/année) et inverserait silencieusement jour et mois
    pour toute date où le jour est <= 12 — d'où un format imposé plutôt qu'une
    inférence automatique.

    Un repli avec dayfirst=True absorbe les petites variantes (secondes
    absentes, séparateur différent) tout en préservant l'interprétation
    jour-d'abord si le format exact ne matche pas parfaitement.
    """
    try:
        return pd.to_datetime(series, format=DATE_CONTACT_FORMAT)
    except (ValueError, TypeError):
        try:
            return pd.to_datetime(series, dayfirst=True)
        except (ValueError, TypeError) as e:
            raise ValueError(
                f"Impossible de lire la colonne date_contact au format attendu "
                f"(JJ/MM/AAAA HH:mm:ss, ex. 05/03/2026 14:30:00). Détail : {e}"
            )


def load_treatments_from_csv(file) -> pd.DataFrame:
    df = pd.read_csv(file)
    df = _map_columns(df, TRAITEMENTS_ALIASES, "traitements")
    df["date_contact"] = _parse_date_contact(df["date_contact"])
    df["type_traitement"] = df["type_traitement"].astype(str).str.strip().str.lower()
    df["account_id"] = pd.to_numeric(df["account_id"], errors="coerce").astype("Int64")
    return df[["traitement_id", "account_id", "date_contact", "type_traitement"]]


def load_portfolio_from_csv(file) -> pd.DataFrame:
    df = pd.read_csv(file)
    df = _map_columns(df, PORTFOLIO_ALIASES, "portefeuille")
    df["profil_partenaire"] = _canonicalize_fallback_label(_canonicalize_categorical(df["profil_partenaire"]))
    df["relation_contractuelle"] = _canonicalize_fallback_label(
        _canonicalize_categorical(df["relation_contractuelle"])
    )
    return df[["account_id", "profil_partenaire", "relation_contractuelle"]]
