"""
ui/controls.py — Widgets de personnalisation (sidebar).

Le dict retourné par `render_customization_controls` pilote tout le recalcul en
aval : résolution, agrégation, taux, lift, alertes et prévision.

Canal et relation contractuelle sont traités comme deux filtres indépendants
au même titre que le profil, plutôt que croisés
systématiquement — ce qui éviterait de diviser les volumes déjà surveillés par
le garde-fou de fiabilité statistique.
"""

from __future__ import annotations

from datetime import date

import pandas as pd
import streamlit as st

from config import (
    ALL_CHANNELS_LABEL,
    CI_METHODS,
    DEFAULT_CI_LEVEL,
    DEFAULT_MIN_ACCOUNTS,
    DEFAULT_N_BOOTSTRAP,
    DEFAULT_PCT_THRESHOLD,
    DEFAULT_ZSCORE_THRESHOLD,
    DISPLAY_MODES,
    FORECAST_HORIZON_UNITS,
    FORECAST_MODEL_LABELS,
    FORECAST_TARGETS,
    GRANULARITIES,
    RATE_DEFINITIONS,
)


def render_data_source_controls() -> dict:
    """Choix de la source de données : démo synthétique ou import CSV réel."""
    st.sidebar.header("📂 Source de données")
    source_label = st.sidebar.radio(
        "Données à analyser",
        ["Démonstration (synthétique)", "Importer mes fichiers CSV"],
        help=(
            "Le générateur de démonstration simule un portefeuille déséquilibré "
            "sur ~18 mois (appels + emails, comptes parfois non identifiés, "
            "rupture de tendance sur un canal), pour tester toutes les "
            "fonctionnalités sans données réelles."
        ),
    )
    result = {"source": "demo", "treatments_file": None, "portfolio_file": None}
    if source_label == "Importer mes fichiers CSV":
        result["source"] = "upload"
        result["treatments_file"] = st.sidebar.file_uploader(
            "Traitements (traitement_id, account_id, date_contact, type_traitement, motif et sous_motif facultatifs)",
            type="csv",
        )
        result["portfolio_file"] = st.sidebar.file_uploader(
            "Portefeuille (account_id, profil_partenaire, relation_contractuelle)",
            type="csv",
        )
    return result


def render_customization_controls(
    profiles: list[str], relations: list[str], canaux: list[str], date_min: date, date_max: date
) -> dict:
    """Rend tous les contrôles de personnalisation et retourne les réglages choisis."""
    st.sidebar.header("⚙️ Personnalisation")

    # --- Mode d'affichage : taux classique ou lift ---
    display_label = st.sidebar.radio("Mode d'affichage", list(DISPLAY_MODES.keys()))
    display_mode = DISPLAY_MODES[display_label]

    # --- Définition du taux : traitements ou comptes distincts ---
    rate_label = st.sidebar.radio("Numérateur du taux", list(RATE_DEFINITIONS.keys()))
    rate_definition = RATE_DEFINITIONS[rate_label]

    # --- Granularité temporelle ---
    gran_label = st.sidebar.radio("Granularité temporelle", list(GRANULARITIES.keys()), index=1)
    freq_code = GRANULARITIES[gran_label]

    st.sidebar.divider()

    # --- Segmentation : canal, profil, relation contractuelle (axes indépendants) ---
    st.sidebar.subheader("Segmentation")
    canal_options = [ALL_CHANNELS_LABEL] + canaux
    canal_label = st.sidebar.radio(
        "Canal", canal_options,
        format_func=lambda v: v if v == ALL_CHANNELS_LABEL else v.capitalize(),
        help="Valeurs détectées automatiquement dans la colonne type_traitement de vos données.",
    )
    canal = "tous" if canal_label == ALL_CHANNELS_LABEL else canal_label

    selected_profiles = st.sidebar.multiselect(
        "Profils à comparer", options=profiles, default=profiles
    )

    relation_options = ["Tous"] + relations
    relation_label = st.sidebar.radio("Relation contractuelle", relation_options)
    relation = "tous" if relation_label == "Tous" else relation_label

    # --- Période d'analyse ---
    date_range = st.sidebar.date_input(
        "Période d'analyse",
        value=(date_min, date_max),
        min_value=date_min,
        max_value=date_max,
    )
    if isinstance(date_range, tuple) and len(date_range) == 2:
        start_date, end_date = date_range
    else:
        start_date, end_date = date_min, date_max

    st.sidebar.divider()

    # --- Fiabilité statistique : volume minimum ---
    st.sidebar.subheader("Fiabilité statistique")
    min_volume = st.sidebar.slider(
        "Volume minimum (comptes et traitements) pour afficher un indicateur",
        min_value=0, max_value=300, value=DEFAULT_MIN_ACCOUNTS, step=5,
        help=(
            "Garde-fou méthodologique : en dessous de ce seuil, un profil peut "
            "sembler sur/sous-sollicité par pur bruit d'échantillonnage plutôt "
            "que par un vrai effet."
        ),
    )

    ci_label = st.sidebar.radio(
        "Méthode d'intervalle de confiance (lift)", list(CI_METHODS.keys())
    )
    ci_method = CI_METHODS[ci_label]
    n_bootstrap = DEFAULT_N_BOOTSTRAP
    if ci_method == "bootstrap":
        n_bootstrap = st.sidebar.slider(
            "Nombre de tirages bootstrap", min_value=50, max_value=1000,
            value=DEFAULT_N_BOOTSTRAP, step=50,
        )

    st.sidebar.divider()

    # --- Seuil de détection d'anomalie ---
    st.sidebar.subheader("Détection d'anomalie")
    threshold_type_label = st.sidebar.radio("Type de seuil", ["Z-score", "Écart en %"])
    threshold_type = "zscore" if threshold_type_label == "Z-score" else "pct"
    if threshold_type == "zscore":
        threshold_value = st.sidebar.slider(
            "Seuil (écarts-types vs historique propre du profil)",
            min_value=1.0, max_value=4.0, value=DEFAULT_ZSCORE_THRESHOLD, step=0.1,
        )
    else:
        threshold_value = st.sidebar.slider(
            "Seuil (% d'écart vs historique propre du profil)",
            min_value=5.0, max_value=100.0, value=DEFAULT_PCT_THRESHOLD, step=5.0,
        )

    st.sidebar.divider()

    # --- Prévision ---
    st.sidebar.subheader("Prévision")
    target_label = st.sidebar.radio("Cible de prévision", list(FORECAST_TARGETS.keys()))
    forecast_target = FORECAST_TARGETS[target_label]

    horizon_unit_label = st.sidebar.radio("Unité d'horizon", list(FORECAST_HORIZON_UNITS.keys()), index=1)
    horizon_n_units = st.sidebar.slider(f"Horizon ({horizon_unit_label.lower()})", 1, 12, 3)
    horizon_periods = _convert_horizon_to_periods(horizon_n_units, horizon_unit_label, freq_code)

    model_label_to_key = {v: k for k, v in FORECAST_MODEL_LABELS.items()}
    model_label = st.sidebar.selectbox(
        "Modèle de prévision", list(FORECAST_MODEL_LABELS.values()), index=0,
        help=(
            "Les variantes « Binomiale négative » gèrent la sur-dispersion "
            "(variance > moyenne), fréquente quand on mélange plusieurs canaux "
            "ou en présence de pics ponctuels (campagnes) : un diagnostic est "
            "affiché dans l'onglet Prévision pour vous aider à choisir."
        ),
    )
    forecast_model = model_label_to_key[model_label]

    return {
        "display_mode": display_mode,
        "rate_definition": rate_definition,
        "freq_code": freq_code,
        "gran_label": gran_label,
        "canal": canal,
        "canal_label": canal_label,
        "relation": relation,
        "selected_profiles": selected_profiles,
        "start_date": pd.Timestamp(start_date),
        "end_date": pd.Timestamp(end_date),
        "min_accounts": min_volume,
        "min_calls": min_volume,
        "ci_method": ci_method,
        "ci_level": DEFAULT_CI_LEVEL,
        "n_bootstrap": n_bootstrap,
        "threshold_type": threshold_type,
        "threshold_value": threshold_value,
        "forecast_target": forecast_target,
        "horizon_periods": horizon_periods,
        "forecast_model": forecast_model,
    }


def _convert_horizon_to_periods(n_units: int, unit_label: str, freq_code: str) -> int:
    """Convertit un horizon exprimé en semaines/mois/trimestres vers un nombre
    de périodes de la granularité d'analyse choisie (jour / semaine / mois)."""
    days = {"Semaine(s)": 7, "Mois": 30, "Trimestre(s)": 91}[unit_label] * n_units
    if freq_code == "D":
        return max(1, round(days))
    if freq_code == "W-MON":
        return max(1, round(days / 7))
    if freq_code == "MS":
        return max(1, round(days / 30))
    raise ValueError(f"Granularité non supportée : {freq_code}")
