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

# Colonnes facultatives de Traitements.csv
TRAITEMENTS_OPTIONAL_ALIASES = {
    "motif": ["motif", "motif_contact", "motif_de_contact", "reason"],
    "sous_motif": ["sous_motif", "sous_motif_contact", "sous_motif_de_contact", "sub_reason"],
}

# Libellé canonique utilisé pour le profil/la relation contractuelle "fourre-tout",
# et pour les traitements dont le compte n'a pas pu être identifié (compte
# manquant -> profil ET relation = ce libellé).
FALLBACK_SEGMENT_LABEL = "Autres"

# Format de date_contact : jour/mois/année heure:minute:seconde. Un parsing sans
# format explicite suppose le format américain et inverserait jour et mois pour
# toute date dont le jour est <= 12.
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

DEFAULT_FREQ_CODE = "W-MON"

# --- Seuils par défaut (garde-fous méthodologiques) ---
DEFAULT_MIN_ACCOUNTS = 30
DEFAULT_MIN_CALLS = 30
DEFAULT_ZSCORE_THRESHOLD = 3.0
DEFAULT_PCT_THRESHOLD = 25.0  # en %

# --- Garde-fou petits effectifs par dimension de détection ---
# Un sous-motif crée des sous-populations beaucoup plus fines que le profil seul
# → seuil minimal plus élevé pour éviter le bruit statistique
# n >= 20 est un compromis standard (CLT approximative, test du chi² nécessitant
# des effectifs attendus >= 5 par cellule, d'où le facteur de sécurité)
MIN_EFFECTIF_DEFAULTS: dict[str, int] = {
    "profil_partenaire": 20,
    "motif": 25,
    "sous_motif": 30,
}

# --- Référence glissante de la détection d'anomalies, par granularité ---
# window : périodes passées conservées ; min_periods : historique minimal pour
# calculer un score ; season : 7 compare chaque jour aux mêmes jours de semaine.
BASELINE_SETTINGS: dict[str, dict[str, int]] = {
    "D": {"window": 8, "min_periods": 4, "season": 7},
    "W-MON": {"window": 26, "min_periods": 10, "season": 1},
    "MS": {"window": 9, "min_periods": 6, "season": 1},
}

# --- Paramètres CUSUM par dimension (k = tolérance, h = seuil, en écarts-types) ---
# Motif et sous-motif sont des séries plus bruitées : leurs seuils sont plus larges.
CUSUM_DEFAULTS_PER_DIM: dict[str, dict[str, float]] = {
    "profil_partenaire": {"k": 0.75, "h": 6.0},
    "motif": {"k": 0.75, "h": 6.0},
    "sous_motif": {"k": 1.0, "h": 7.0},
}

# --- Surveillance de la part d'un motif (CUSUM binomial) ---
# odds_ratio : hausse à détecter (2 = les cotes du motif doublent) ;
# h : seuil de déclenchement ; min_baseline : traitements de référence requis.
SHARE_CUSUM_DEFAULTS = {"odds_ratio": 2.0, "h": 5.5, "min_baseline": 200}

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
    "negbin_glm": "GLM Binomiale négative (sur-dispersion)",
    "baseline": "Référence (saisonnière naïve)",
    "linear": "Régression linéaire",
    "sarima": "SARIMA",
    "prophet": "Prophet",
    "xgboost_poisson": "XGBoost (Poisson)",
    "poisson_glm": "GLM Poisson (fréquentiste)",
    "bayesian_poisson": "Bayésien Poisson (PyMC)",
    "bayesian_negbin": "Bayésien Binomiale négative (PyMC)",
}
