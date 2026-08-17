"""
config.py — Constantes et valeurs par défaut de l'application.

Centraliser ici tout ce qui pourrait changer avec la vraie source de données
(noms de colonnes, seuils par défaut) pour éviter les valeurs en dur dispersées
dans le code.

Schéma réel (portefeuille simplifié, sans historisation) :
- Traitements.csv : traitement_id | account_id (parfois vide) | date_contact | type_traitement (appel/email)
- Portefeuille.csv : account_id | profil_partenaire (5 valeurs, dont "Autres") | relation_contractuelle (client actif / autres)

Le portefeuille n'est PAS versionné (pas de date_snapshot) : chaque compte a un
seul profil et une seule relation contractuelle, fixes dans le temps.

Le nom exact des colonnes dans les fichiers réels peut varier (accents, casse,
espaces) : voir data/loader.py::_map_columns pour la table d'alias utilisée à
l'import.
"""

from datetime import date, timedelta

# --- Schéma interne attendu après normalisation des colonnes (voir data/loader.py) ---
TRAITEMENTS_SCHEMA = ["traitement_id", "account_id", "date_contact", "type_traitement"]
PORTFOLIO_SCHEMA = ["account_id", "profil_partenaire", "relation_contractuelle"]

# Alias acceptés à l'import CSV (après normalisation : minuscules, sans accents,
# espaces/slashs -> underscore). Ajouter ici toute variante rencontrée en pratique.
TRAITEMENTS_ALIASES = {
    "traitement_id": ["traitement_id", "traitements_id", "id_traitement", "treatment_id", "id"],
    "account_id": ["account_id", "id_compte", "compte_id", "account"],
    "date_contact": [
        "date_contact", "date_de_contact", "date_de_creation", "date_de_creation_de_contact",
        "date_creation", "date_creation_contact", "timestamp", "date",
    ],
    "type_traitement": [
        "type_traitement", "type_traitements", "type_de_traitement", "type_de_traitements",
        "type", "canal", "canal_traitement", "type_contact", "type_de_contact",
    ],
}
PORTFOLIO_ALIASES = {
    "account_id": ["account_id", "id_compte", "compte_id"],
    "profil_partenaire": ["profil_partenaire", "profil_compte", "profil_id", "profil"],
    "relation_contractuelle": [
        "relation_contractuelle", "relation_contractuelle_compte", "relation",
    ],
}

# Libellé canonique utilisé pour le profil/la relation contractuelle "fourre-tout",
# et pour les traitements dont le compte n'a pas pu être identifié (compte
# manquant -> profil ET relation = ce libellé).
FALLBACK_SEGMENT_LABEL = "Autres"

# Format de date_contact dans Traitements.csv (confirmé par l'utilisateur) :
# jour/mois/année heure:minute:seconde. Un parsing sans format explicite
# suppose par défaut le format américain (mois/jour/année) et inverserait
# silencieusement jour et mois pour toute date où le jour est <= 12.
DATE_CONTACT_FORMAT = "%d/%m/%Y %H:%M:%S"

# --- Granularités temporelles disponibles ---
GRANULARITIES = {
    "Jour": "D",
    "Semaine": "W-MON",
    "Mois": "MS",
}

# --- Définitions du taux de traitement ---
RATE_DEFINITIONS = {
    "Nombre de traitements": "n_traitements",
    "Comptes distincts traités": "n_comptes_distincts",
}

# --- Canal ---
# Les valeurs de canal sont découvertes dynamiquement depuis la colonne
# type_traitement des données chargées (voir ui/controls.py), pas figées ici :
# le vocabulaire réel ("appel"/"email", ou tout autre libellé) peut varier
# d'une source à l'autre.
ALL_CHANNELS_LABEL = "Tous les canaux"

# --- Modes d'affichage ---
DISPLAY_MODES = {
    "Taux de traitement classique": "rate",
    "Indice de sur-sollicitation (lift)": "lift",
}

# --- Cibles de prévision ---
FORECAST_TARGETS = {
    "Charge globale (volume total)": "global",
    "Comportement par profil": "profile",
    "Les deux": "both",
}

# --- Horizons de prévision ---
FORECAST_HORIZON_UNITS = {
    "Semaine(s)": 1,
    "Mois": 4,
    "Trimestre(s)": 13,
}

# --- Seuils par défaut (garde-fous méthodologiques, section 4) ---
DEFAULT_MIN_ACCOUNTS = 30
DEFAULT_MIN_CALLS = 30
DEFAULT_ZSCORE_THRESHOLD = 2.0
DEFAULT_PCT_THRESHOLD = 25.0  # en %

# --- Intervalle de confiance ---
CI_METHODS = {
    "Approximation binomiale (rapide)": "binomial",
    "Bootstrap (plus robuste, plus lent)": "bootstrap",
}
DEFAULT_CI_LEVEL = 0.95
DEFAULT_N_BOOTSTRAP = 300

# --- Diagnostic de sur-dispersion (Poisson vs Binomiale négative) ---
# Ratio variance/moyenne au-delà duquel on recommande une alternative à Poisson
# (mélange de canaux hétérogènes, campagnes email, etc. -> sur-dispersion typique).
OVERDISPERSION_WARNING_RATIO = 1.5

# --- Période par défaut pour le générateur de données de démonstration ---
# On cale la fin sur le dernier jour du mois précédent (mois complet), pour
# éviter qu'un mois en cours (partiel) ne fausse les taux et les prévisions.
_today = date.today()
DEMO_END_DATE = date(_today.year, _today.month, 1) - timedelta(days=1)
DEMO_START_DATE = DEMO_END_DATE - timedelta(days=540)  # ~18 mois d'historique

# --- Registre de modèles de prévision disponibles (voir forecasting/registry.py) ---
FORECAST_MODEL_LABELS = {
    "linear": "Régression linéaire",
    "sarima": "SARIMA",
    "prophet": "Prophet",
    "xgboost_poisson": "XGBoost (Poisson)",
    "poisson_glm": "GLM Poisson (fréquentiste)",
    "negbin_glm": "GLM Binomiale négative (sur-dispersion)",
    "bayesian_poisson": "Bayésien Poisson (PyMC)",
    "bayesian_negbin": "Bayésien Binomiale négative (PyMC)",
}
