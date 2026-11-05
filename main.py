"""
main.py — Point d'entrée Streamlit.

Orchestration : sidebar (source de données + personnalisation) -> résolution
profil/relation -> filtres (canal/relation) -> agrégation -> métriques (taux,
lift, anomalies) -> rendu (vue synthétique / évolution / alertes / prévision).

Points de conception :
- Traitements.csv n'a pas de profil : il est résolu par fusion directe avec le
  portefeuille (statique, sans historique) sur account_id, avec repli sur
  "Autres" (compte manquant ou non trouvé).
- Canal (appel/email) et relation contractuelle sont deux filtres indépendants.
- Un indicateur de qualité de donnée séparé rend visible la part de "Autres"
  qui vient de traitements non identifiables plutôt que de vrais comptes "Autres".
- Un diagnostic de sur-dispersion (variance/moyenne) aide à choisir entre les
  variantes Poisson et Binomiale négative du registre de prévision.
"""

from __future__ import annotations

import os
import sys

# Rend les imports (config, data.*, metrics.*, ...) robustes quel que soit le
# répertoire depuis lequel `streamlit run` est invoqué.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from config import (
    FORECAST_MODEL_LABELS,
    OVERDISPERSION_WARNING_RATIO,
)
from data.loader import (
    load_demo_data,
    load_portfolio_from_csv,
    load_treatments_from_csv,
)
from data.aggregator import (
    _period_end,
    aggregate_treatments_by_motif,
    floor_to_period,
    apply_filters,
    build_master_table,
    compute_data_quality,
    resolve_profile_and_relation,
)
from metrics.call_rate import add_call_rate
from metrics.motive_analysis import (
    detect_motif_share_shifts,
    motif_distribution_drilldown,
    performance_comparison_motif_aggregation,
)
from metrics.lift import add_lift, add_lift_ci_binomial, apply_min_volume_mask, bootstrap_lift_ci
from metrics.anomaly import (
    add_cusum_breaks,
    add_deviation_from_own_history,
    flag_alerts,
    period_over_period_table,
)
from forecasting.backtest import backtest_all_models
from forecasting.registry import compute_dispersion_ratio, forecast_series
from ui.charts import build_forecast_chart, build_lift_bar_chart, build_time_series_chart
from ui.controls import render_customization_controls, render_data_source_controls

st.set_page_config(page_title="Taux de traitement & sur-sollicitation par profil", layout="wide")


# ---------------------------------------------------------------------------
# Chargement des données (mise en cache)
# ---------------------------------------------------------------------------

@st.cache_data(show_spinner="Génération des données de démonstration...")
def _cached_demo_data():
    return load_demo_data()


@st.cache_data(show_spinner="Résolution profil / relation contractuelle...")
def _cached_resolve(treatments_df, portfolio_df):
    return resolve_profile_and_relation(treatments_df, portfolio_df)


@st.cache_data(show_spinner="Agrégation des données...")
def _cached_master_table(treatments_df, portfolio_df, freq_code, start_date, end_date):
    return build_master_table(treatments_df, portfolio_df, freq_code, start_date, end_date)


@st.cache_data(show_spinner="Calcul de la prévision...")
def _cached_forecast(ts, model_name, horizon):
    return forecast_series(ts, model_name, horizon)


@st.cache_data(show_spinner="Agrégation des motifs...")
def _cached_motif_table(treatments_df, freq_code):
    return aggregate_treatments_by_motif(treatments_df, freq_code)


def _load_raw_data(source_settings: dict):
    if source_settings["source"] == "demo":
        return _cached_demo_data()

    treatments_file = source_settings["treatments_file"]
    portfolio_file = source_settings["portfolio_file"]
    if treatments_file is None or portfolio_file is None:
        st.info("Importez les deux fichiers CSV (traitements et portefeuille) pour continuer, ou basculez sur la démonstration.")
        st.stop()
    try:
        treatments_df = load_treatments_from_csv(treatments_file)
        portfolio_df = load_portfolio_from_csv(portfolio_file)
    except ValueError as e:
        st.error(str(e))
        st.stop()
    return treatments_df, portfolio_df


# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------

st.title("📞✉️ Suivi et prédiction du taux de traitement par profil")
st.caption(
    "Outil de pilotage paramétrable : les questions méthodologiques en suspens "
    "(granularité, définition du taux, canal, seuils...) se tranchent ici, en "
    "direct, via les contrôles de la barre latérale."
)

source_settings = render_data_source_controls()
treatments_raw, portfolio_df = _load_raw_data(source_settings)
resolved_all = _cached_resolve(treatments_raw, portfolio_df)

all_profiles = sorted(portfolio_df["profil_partenaire"].unique())
all_relations = sorted(portfolio_df["relation_contractuelle"].unique())
all_canaux = sorted(resolved_all["type_traitement"].dropna().unique())
date_min = resolved_all["date_contact"].min().date()
date_max = resolved_all["date_contact"].max().date()

settings = render_customization_controls(all_profiles, all_relations, all_canaux, date_min, date_max)

if not settings["selected_profiles"]:
    st.warning("Sélectionnez au moins un profil dans la barre latérale pour afficher les analyses.")
    st.stop()

# --- Filtres canal / relation (avant agrégation) ---
filtered_treatments, filtered_portfolio = apply_filters(
    resolved_all, portfolio_df, settings["canal"], settings["relation"]
)

if filtered_treatments.empty or filtered_portfolio.empty:
    st.warning("Aucune donnée ne correspond à cette combinaison de filtres (canal / relation contractuelle).")
    st.stop()

# --- Indicateur de qualité de donnée (traitements sans compte identifié) ---
period_mask = filtered_treatments["date_contact"].between(settings["start_date"], settings["end_date"])
dq = compute_data_quality(filtered_treatments[period_mask])

with st.expander("ℹ️ Qualité de donnée — traitements sans compte identifié", expanded=False):
    col1, col2, col3 = st.columns(3)
    col1.metric("Non identifiés (global)", f"{dq['pct_non_identifie_global']:.1f} %")
    for canal_name, pct in dq["pct_non_identifie_par_canal"].items():
        (col2 if canal_name == "appel" else col3).metric(f"Non identifiés — {canal_name}", f"{pct:.1f} %")
    st.caption(
        f"Sur les {dq['n_autres_total']} traitements actuellement rattachés au profil « Autres », "
        f"**{dq['part_autres_sans_compte']:.1f} %** viennent de traitements sans compte identifié "
        f"({dq['n_autres_sans_compte']} traitements) plutôt que de vrais comptes classés « Autres » "
        f"dans le portefeuille. Le taux/lift de « Autres » doit donc être interprété avec prudence : "
        f"son numérateur est gonflé par une population qui ne contribue à aucun dénominateur."
    )

# --- Pipeline de calcul ---
master = _cached_master_table(
    filtered_treatments, filtered_portfolio, settings["freq_code"], settings["start_date"], settings["end_date"]
)
master = add_call_rate(master, settings["rate_definition"])
master = add_lift(master, settings["rate_definition"])
master = add_lift_ci_binomial(master, settings["rate_definition"], settings["ci_level"])
master = apply_min_volume_mask(master, settings["rate_definition"], settings["min_accounts"], settings["min_calls"])

value_col = "lift" if settings["display_mode"] == "lift" else "taux"
y_title = "Lift (indice de sur-représentation)" if value_col == "lift" else "Taux de traitement"
reference_line = 1.0 if value_col == "lift" else None

master = add_deviation_from_own_history(
    master, value_col=value_col, group_col="profil_partenaire",
    freq_code=settings["freq_code"], min_effectif=20,
)
master = flag_alerts(
    master, threshold_type=settings["threshold_type"], threshold_value=settings["threshold_value"],
    min_effectif=20, n_col="n_traitements",
)
master = add_cusum_breaks(
    master, value_col="rate_numerator", group_col="profil_partenaire",
    freq_code=settings["freq_code"], min_effectif=20, n_col="n_traitements",
)

last_period = master["period"].max()
selected = settings["selected_profiles"]
canal_caption = settings["canal_label"]

# Les parts de motif demandent du volume : la surveillance journalière est faite par semaine.
motif_freq = "W-MON" if settings["freq_code"] == "D" else settings["freq_code"]
treatments_in_range = filtered_treatments[period_mask]
motif_table = _cached_motif_table(treatments_in_range, motif_freq)
motif_table = motif_table[motif_table["period"] <= last_period]
motif_shifts = detect_motif_share_shifts(motif_table)
motif_shifts = motif_shifts[motif_shifts["profil_partenaire"].isin(selected)]

if not (set(selected) & set(master["profil_partenaire"].unique())):
    st.warning(
        "Aucun des profils sélectionnés n'a de données pour cette combinaison de filtres "
        "(canal / relation contractuelle). Ajustez la sélection dans la barre latérale — "
        "par exemple, le profil « Autres » n'a jamais de relation « Client actif » dans les données de démonstration."
    )
    st.stop()

tab_synth, tab_evol, tab_alert, tab_qualif, tab_forecast = st.tabs(
    ["📊 Vue synthétique", "📈 Évolution", "🚨 Alertes", "🔍 Qualification", "🔮 Prévision"]
)


# ---------------------------------------------------------------------------
# Onglet 1 — Vue synthétique
# ---------------------------------------------------------------------------
with tab_synth:
    st.subheader(f"Dernière période disponible : {last_period.date()} · Canal : {canal_caption}")
    snapshot = master[master["period"] == last_period].copy()

    if settings["ci_method"] == "bootstrap":
        period_end = _period_end(last_period, settings["freq_code"])
        period_treatments = filtered_treatments[
            filtered_treatments["date_contact"].between(last_period, period_end)
        ]
        # portefeuille statique (pas de notion de snapshot) : on l'utilise tel quel
        period_portfolio = filtered_portfolio

        lift_med, lift_low, lift_high = [], [], []
        for pid in snapshot["profil_partenaire"]:
            med, low, high = bootstrap_lift_ci(
                period_treatments, period_portfolio, pid, settings["rate_definition"],
                n_bootstrap=settings["n_bootstrap"], ci_level=settings["ci_level"],
            )
            lift_med.append(med)
            lift_low.append(low)
            lift_high.append(high)
        snapshot["lift_ci_low"] = lift_low
        snapshot["lift_ci_high"] = lift_high

    fig_bar = build_lift_bar_chart(snapshot, value_col="lift")
    st.plotly_chart(fig_bar, width='stretch')

    display_cols = [
        "profil_partenaire", "n_accounts", settings["rate_definition"], "taux",
        "lift", "lift_ci_low", "lift_ci_high", "is_reliable",
    ]
    display_df = snapshot[display_cols].sort_values("lift", ascending=False).rename(
        columns={
            "profil_partenaire": "Profil", "n_accounts": "Comptes",
            settings["rate_definition"]: "Numérateur taux", "taux": "Taux de traitement",
            "lift": "Lift", "lift_ci_low": "IC bas", "lift_ci_high": "IC haut",
            "is_reliable": "Volume fiable",
        }
    )
    st.dataframe(
        display_df.style.format({
            "Taux de traitement": "{:.3f}", "Lift": "{:.2f}", "IC bas": "{:.2f}", "IC haut": "{:.2f}",
        }),
        width='stretch', hide_index=True,
    )
    st.caption(
        "Les profils grisés (volume fiable = False) sont sous le seuil de volume minimum "
        "défini dans la barre latérale : leur lift peut refléter du bruit d'échantillonnage "
        "plutôt qu'un vrai signal."
    )


# ---------------------------------------------------------------------------
# Onglet 2 — Évolution temporelle
# ---------------------------------------------------------------------------
with tab_evol:
    fig_ts = build_time_series_chart(
        master, selected, value_col, y_title,
        ci_low_col="lift_ci_low" if value_col == "lift" else None,
        ci_high_col="lift_ci_high" if value_col == "lift" else None,
        alert_col="is_alert",
        reference_line=reference_line,
    )
    st.plotly_chart(fig_ts, width='stretch')
    st.caption(
        "Les marqueurs ✕ signalent une période en alerte (écart significatif par rapport à "
        "l'historique PROPRE du profil, pas par rapport aux autres profils)."
    )

    st.subheader("📈 Évolution des motifs / sous-motifs")
    detail_sel = st.selectbox("Niveau de détail", ["motif", "sous_motif"], key="evol_det")
    evol_df = _cached_motif_table(treatments_in_range, settings["freq_code"])
    evol_df = evol_df[
        evol_df["profil_partenaire"].isin(selected)
        & evol_df["period"].between(master["period"].min(), last_period)
    ]
    display_df = (
        evol_df.fillna({detail_sel: "Non qualifié"})
        .groupby(["period", "profil_partenaire", detail_sel])["n_traitements"].sum().reset_index()
        .rename(columns={"n_traitements": "Volume", detail_sel: "Catégorie"})
    )
    display_df["Part du profil (%)"] = (
        display_df["Volume"] / display_df.groupby(["period", "profil_partenaire"])["Volume"].transform("sum") * 100
    )
    alert_keys = set(master.loc[master["is_alert"], ["profil_partenaire", "period"]].itertuples(index=False, name=None))
    display_df["Alerte"] = [
        "⚠ Oui" if key in alert_keys else "Non"
        for key in zip(display_df["profil_partenaire"], display_df["period"])
    ]
    st.dataframe(
        display_df.sort_values(["period", "Volume"], ascending=[True, False]).rename(
            columns={"period": "Période", "profil_partenaire": "Profil"}
        ),
        column_config={
            "Période": st.column_config.DateColumn(format="YYYY-MM-DD"),
            "Part du profil (%)": st.column_config.NumberColumn(format="%.1f"),
        },
        width="stretch",
        hide_index=True,
    )
    st.caption("Volume : nombre de traitements. Alerte : la période est en alerte pour ce profil.")

    st.subheader("Comparaison période sur période")
    periods_available = sorted(master["period"].unique())
    if len(periods_available) >= 2:
        col1, col2 = st.columns(2)
        with col1:
            period_a = st.selectbox("Période A", periods_available, index=len(periods_available) - 2, format_func=lambda p: pd.Timestamp(p).date().isoformat())
        with col2:
            period_b = st.selectbox("Période B", periods_available, index=len(periods_available) - 1, format_func=lambda p: pd.Timestamp(p).date().isoformat())

        comp = period_over_period_table(
            master[master["profil_partenaire"].isin(selected)], period_a, period_b, value_col
        ).rename(columns={
            "profil_partenaire": "Profil", "value_a": "Période A", "value_b": "Période B",
            "delta_abs": "Écart", "delta_pct": "Écart (%)",
        })
        st.dataframe(
            comp.style.format({"Période A": "{:.3f}", "Période B": "{:.3f}", "Écart": "{:+.3f}", "Écart (%)": "{:+.1f}"}),
            width='stretch', hide_index=True,
        )


# ---------------------------------------------------------------------------
# Onglet 3 — Alertes
# ---------------------------------------------------------------------------
with tab_alert:
    st.subheader(f"Écarts ponctuels sur la dernière période — {last_period.date()}")
    alerts_last = master[(master["period"] == last_period) & (master["profil_partenaire"].isin(selected))]
    alerts_flagged = alerts_last[alerts_last["is_alert"]]

    if alerts_flagged.empty:
        st.success("Aucun écart ponctuel sur la dernière période pour les profils sélectionnés.")
    else:
        cols = ["profil_partenaire", value_col, "hist_ref", "zscore", "pct_deviation"]
        st.dataframe(
            alerts_flagged[cols].rename(columns={
                "profil_partenaire": "Profil", value_col: "Valeur", "hist_ref": "Référence",
                "zscore": "Z-score", "pct_deviation": "Écart (%)",
            }).style.format({"Valeur": "{:.3f}", "Référence": "{:.3f}", "Z-score": "{:.2f}", "Écart (%)": "{:.1f}"}),
            width='stretch', hide_index=True,
        )

    st.subheader("Ruptures de volume (CUSUM)")
    breaks = master[(master["cusum_break"]) & (master["profil_partenaire"].isin(selected))].copy()
    if breaks.empty:
        st.info("Aucune rupture de volume détectée sur la période et les profils sélectionnés.")
    else:
        breaks["Sens"] = breaks["cusum_direction"].map({1: "Hausse", -1: "Baisse"})
        breaks["period"] = breaks["period"].dt.date
        st.dataframe(
            breaks.sort_values("period", ascending=False)[["profil_partenaire", "period", "Sens", value_col]].rename(
                columns={"profil_partenaire": "Profil", "period": "Détectée le", value_col: "Valeur"}
            ).style.format({"Valeur": "{:.3f}"}),
            width='stretch', hide_index=True,
        )
        st.caption("La référence n'utilise que le passé : chaque rupture aurait été signalée à cette date.")

    st.subheader("Changements de motif")
    if motif_shifts.empty:
        st.info("Aucun motif n'a pris durablement plus de poids sur la période.")
    else:
        st.dataframe(
            motif_shifts.assign(
                debut_estime=motif_shifts["debut_estime"].dt.date,
                period=motif_shifts["period"].dt.date,
                part_reference=motif_shifts["part_reference"] * 100,
                part_observee=motif_shifts["part_observee"] * 100,
            ).rename(columns={
                "profil_partenaire": "Profil", "motif": "Motif", "debut_estime": "Début estimé",
                "period": "Détecté le", "part_reference": "Part de référence (%)",
                "part_observee": "Part observée (%)",
            }).style.format({"Part de référence (%)": "{:.1f}", "Part observée (%)": "{:.1f}"}),
            width='stretch', hide_index=True,
        )
        st.caption(
            "Le volume total peut rester stable alors qu'un motif en remplace d'autres : "
            "cette surveillance voit des changements invisibles dans les alertes de volume."
        )


# ---------------------------------------------------------------------------
# Onglet — Qualification des anomalies (motif / sous-motif)
# ---------------------------------------------------------------------------
with tab_qualif:
    st.subheader("Qualification d'une anomalie par motif / sous-motif")
    st.caption(
        "Compare la répartition des motifs pendant l'anomalie à celle des périodes "
        "précédentes, teste l'écart et identifie la catégorie qui en est la cause."
    )

    def _motif_period(period: pd.Timestamp) -> pd.Timestamp:
        return floor_to_period(pd.Series([period]), motif_freq).iloc[0]

    events = [
        {
            "label": (
                f"Motif · {row.profil_partenaire} · {row.motif} · détecté le {row.period.date()} "
                f"({row.part_reference * 100:.0f} % → {row.part_observee * 100:.0f} %)"
            ),
            "profil": row.profil_partenaire, "start": row.debut_estime, "end": row.period, "motif": row.motif,
        }
        for row in motif_shifts.itertuples()
    ]
    volume_events = master[
        (master["is_alert"] | master["cusum_break"]) & master["profil_partenaire"].isin(selected)
    ].sort_values("period", ascending=False)
    for row in volume_events.itertuples():
        kind = "Rupture" if row.cusum_break else "Écart"
        period = _motif_period(row.period)
        events.append({
            "label": f"{kind} de volume · {row.profil_partenaire} · {row.period.date()}",
            "profil": row.profil_partenaire, "start": period, "end": period, "motif": None,
        })

    if not events:
        st.info("Aucune anomalie à qualifier sur la période et les profils sélectionnés.")
    else:
        col_q1, col_q2 = st.columns([3, 2])
        with col_q1:
            event = st.selectbox(
                "Anomalie à qualifier", events, format_func=lambda item: item["label"], key="qualif_event"
            )
        with col_q2:
            qual_detail = st.radio(
                "Niveau de détail", ["Motif", "Sous-motif"], horizontal=True, key="qualif_detail"
            )
            qual_ref_n = st.slider("Périodes de référence", 4, 52, 26, key="qualif_ref_n")

        detail_col = "motif" if qual_detail == "Motif" else "sous_motif"
        table = motif_table
        if detail_col == "sous_motif" and event["motif"] is not None:
            table = table[table["motif"] == event["motif"]]

        result = motif_distribution_drilldown(
            table, event["end"], event["profil"], detail_col,
            anomaly_start=event["start"], reference_periods=qual_ref_n,
        )
        chi2_result = result["chisq_result"]
        comparison = result["comparison"]

        st.caption(
            f"Fenêtre analysée : {event['start'].date()} au {event['end'].date()}, comparée aux "
            f"{result['n_reference_periods']} périodes précédentes. "
            f"Traitements non qualifiés exclus : {result['pct_non_qualifie']:.1f} %."
        )

        fig_qual = go.Figure()
        fig_qual.add_bar(x=comparison.index, y=comparison["Part (%)"], name="Anomalie", marker_color="#d62728")
        fig_qual.add_bar(x=comparison.index, y=comparison["Part référence (%)"], name="Référence", marker_color="#1f77b4")
        fig_qual.update_layout(barmode="group", yaxis_title="Part des traitements qualifiés (%)")
        st.plotly_chart(fig_qual, width='stretch')

        if chi2_result["pvalue"] is None:
            st.info(chi2_result["conclusion"])
        elif chi2_result["pvalue"] < 0.05:
            st.warning(chi2_result["conclusion"])
        else:
            st.success(chi2_result["conclusion"])

        if chi2_result["table"] is not None:
            st.dataframe(
                chi2_result["table"].rename(columns={
                    "observe": "Observé", "attendu": "Attendu", "residu_standardise": "Résidu standardisé",
                }).style.format({"Attendu": "{:.1f}", "Résidu standardisé": "{:+.1f}"}),
                width='stretch',
            )
            st.caption(f"Méthode : {chi2_result['method']}. Un résidu au-delà de ±2 signale un écart net.")

    st.divider()
    st.subheader("⚡ Performance : requête directe ou table pré-agrégée")
    st.caption(
        "Mesure le temps de réponse d'une agrégation motif/période, sur toutes les lignes "
        "ou sur une table pré-agrégée, ainsi que le coût de construction de cette table."
    )
    if st.button("Lancer le comparatif de performance", key="run_perf_motif"):
        with st.spinner("Mesure en cours..."):
            timed_df = treatments_in_range.assign(
                period=floor_to_period(treatments_in_range["date_contact"], motif_freq)
            )
            st.session_state["perf_motif"] = performance_comparison_motif_aggregation(timed_df)

    perf = st.session_state.get("perf_motif")
    if perf:
        c1, c2, c3 = st.columns(3)
        c1.metric("Requête directe", f"{perf['time_raw_ms']:.2f} ms")
        c2.metric("Table pré-agrégée", f"{perf['time_cube_ms']:.2f} ms", f"×{perf['speedup']:.1f} plus rapide", delta_color="off")
        c3.metric("Construction de la table", f"{perf['build_cube_ms']:.1f} ms")
        amorti = (
            f"La table est amortie dès {perf['break_even_queries']} requêtes."
            if perf["break_even_queries"] else "La table n'apporte pas de gain sur ce volume."
        )
        st.caption(
            f"{perf['rows_raw']:,} lignes réduites à {perf['rows_cube']:,} ; "
            f"moyenne sur {perf['iterations']} requêtes. {amorti}".replace(",", " ")
        )

# ---------------------------------------------------------------------------
# Onglet 4 — Prévision
# ---------------------------------------------------------------------------
with tab_forecast:
    model_name = settings["forecast_model"]
    horizon = settings["horizon_periods"]
    st.caption(
        f"Canal : **{canal_caption}** · Modèle : **{FORECAST_MODEL_LABELS[model_name]}** · "
        f"Horizon : **{horizon} période(s)** de la granularité choisie ({settings['gran_label'].lower()})."
    )
    st.info(
        "Les modèles à tendance (linéaire, GLM/Bayésien Poisson ou Binomiale négative) "
        "extrapolent la tendance observée sur TOUTE la période d'analyse. En cas de "
        "rupture nette (visible onglet Alertes), restreignez la période d'analyse à la "
        "période post-rupture avant de lancer la prévision.",
        icon="ℹ️",
    )

    numerator_col = settings["rate_definition"]

    def _get_ts(profil_partenaire: str | None) -> pd.Series:
        if profil_partenaire is None:
            s = master.groupby("period")[numerator_col].sum()
        else:
            s = master[master["profil_partenaire"] == profil_partenaire].set_index("period")[numerator_col]
        return s.sort_index()

    def _dispersion_caption(ts: pd.Series):
        ratio = compute_dispersion_ratio(ts)
        if np.isnan(ratio):
            return
        if ratio > OVERDISPERSION_WARNING_RATIO:
            st.warning(
                f"Diagnostic de sur-dispersion : ratio variance/moyenne = **{ratio:.1f}** (> {OVERDISPERSION_WARNING_RATIO}). "
                f"Les données s'écartent d'une hypothèse Poisson (mélange de canaux, campagnes, rupture...) — "
                f"les variantes **Binomiale négative** du menu de modèles donneront des intervalles de confiance "
                f"plus honnêtes que les variantes Poisson.",
                icon="📈",
            )
        else:
            st.caption(f"Diagnostic de sur-dispersion : ratio variance/moyenne = {ratio:.1f} (proche de 1 : Poisson raisonnable).")

    forecast_target = settings["forecast_target"]

    global_fc, global_note = None, None
    if forecast_target in ("global", "both"):
        ts_global = _get_ts(None)
        _dispersion_caption(ts_global)
        try:
            global_fc, global_note = _cached_forecast(ts_global, model_name, horizon)
        except ValueError as e:
            st.error(str(e))

    if forecast_target in ("global", "both") and global_fc is not None:
        st.subheader("Charge globale (volume total de traitements)")
        if global_note:
            st.warning(global_note)
        st.plotly_chart(
            build_forecast_chart(_get_ts(None), global_fc, "Prévision — volume global"),
            width='stretch',
        )

    if forecast_target in ("profile", "both"):
        st.subheader("Comportement par profil")
        profil_for_forecast = st.selectbox("Profil à prévoir", selected, key="forecast_profile_select")
        ts_profile = _get_ts(profil_for_forecast)
        _dispersion_caption(ts_profile)
        try:
            profile_fc, profile_note = _cached_forecast(ts_profile, model_name, horizon)
        except ValueError as e:
            st.error(str(e))
            profile_fc, profile_note = None, None

        if profile_fc is not None:
            if profile_note:
                st.warning(profile_note)
            st.plotly_chart(
                build_forecast_chart(ts_profile, profile_fc, f"Prévision — {profil_for_forecast}"),
                width='stretch',
            )

            # Indice de sur-sollicitation projeté : nécessite la prévision globale ;
            # on la calcule silencieusement si elle n'a pas déjà été demandée.
            if global_fc is None:
                ts_global_bg = _get_ts(None)
                try:
                    global_fc, _ = _cached_forecast(ts_global_bg, model_name, horizon)
                except ValueError:
                    global_fc = None

            if global_fc is not None:
                last_share_row = master[
                    (master["period"] == last_period) & (master["profil_partenaire"] == profil_for_forecast)
                ]
                total_accounts_last = master[master["period"] == last_period]["n_accounts"].sum()
                if not last_share_row.empty and total_accounts_last > 0:
                    portfolio_share = float(last_share_row["n_accounts"].iloc[0]) / total_accounts_last
                    with np.errstate(divide="ignore", invalid="ignore"):
                        projected_pct = profile_fc["yhat"] / global_fc["yhat"].replace(0, np.nan)
                        projected_lift = projected_pct / portfolio_share if portfolio_share > 0 else np.nan

                    proj_df = pd.DataFrame({
                        "Période": profile_fc["ds"].dt.date,
                        "Lift projeté": projected_lift,
                    })
                    st.markdown("**Indice de sur-sollicitation projeté** _(portefeuille supposé stable au dernier poids connu)_")
                    st.dataframe(
                        proj_df.style.format({"Lift projeté": "{:.2f}"}), width='stretch', hide_index=True,
                    )

    # -----------------------------------------------------------------------
    # Validation des modèles (backtesting par repli temporel)
    # -----------------------------------------------------------------------
    st.divider()
    st.subheader("🔍 Validation des modèles (backtesting)")
    st.caption(
        "Compare les modèles du registre par validation glissante (walk-forward) : "
        "chacun est ré-entraîné sur l'historique disponible à plusieurs dates de "
        "coupure successives, puis évalué sur les périodes suivantes réellement "
        "observées — une mesure d'erreur hors-échantillon, pas un simple ajustement. "
        "La couverture de l'IC indique si l'intervalle à 95 % contient vraiment le "
        "réel ~95 % du temps (trop bas = modèle trop confiant, typiquement une "
        "variante Poisson sur une série sur-dispersée). Le gain se lit par rapport au "
        "modèle de référence, sans paramètre à entraîner : un modèle complexe doit le battre."
    )

    backtest_series_options: dict[str, str | None] = {}
    if forecast_target in ("global", "both"):
        backtest_series_options["Charge globale"] = None
    for pid in selected:
        backtest_series_options[f"Profil : {pid}"] = pid

    col_a, col_b = st.columns([2, 1])
    with col_a:
        backtest_choice = st.selectbox(
            "Série à valider", list(backtest_series_options.keys()), key="backtest_series_choice"
        )
    with col_b:
        n_folds = st.slider("Replis glissants", min_value=2, max_value=8, value=5, key="backtest_n_folds")

    if st.button("Lancer la validation croisée", key="run_backtest_button"):
        ts_to_validate = _get_ts(backtest_series_options[backtest_choice])
        with st.spinner(f"Validation croisée en cours ({n_folds} replis) — peut prendre quelques minutes..."):
            st.session_state["backtest_leaderboard"] = backtest_all_models(ts_to_validate, horizon, n_folds)
            st.session_state["backtest_context"] = (
                f"{backtest_choice} · canal {canal_caption} · horizon {horizon} période(s) · {n_folds} replis"
            )

    leaderboard = st.session_state.get("backtest_leaderboard")
    if leaderboard is not None:
        st.caption(f"Résultat pour : {st.session_state.get('backtest_context', '')}")

        display_leaderboard = leaderboard.copy()
        display_leaderboard["Modèle"] = display_leaderboard["model"].map(FORECAST_MODEL_LABELS)
        display_leaderboard["Sélectionné"] = display_leaderboard["model"].eq(model_name).map({True: "⭐", False: ""})
        display_leaderboard = display_leaderboard.rename(columns={
            "mae": "MAE", "wape": "WAPE (%)", "gain_vs_reference": "Gain vs référence (%)",
            "rmse": "RMSE", "bias": "Biais", "coverage": "Couverture IC (%)",
            "n_folds": "Replis OK", "n_failures": "Échecs", "avg_fit_time_s": "Temps moyen (s)",
        })
        cols_order = ["Sélectionné", "Modèle", "MAE", "WAPE (%)", "Gain vs référence (%)", "RMSE", "Biais", "Couverture IC (%)", "Replis OK", "Échecs", "Temps moyen (s)"]
        st.dataframe(
            display_leaderboard[cols_order].style.format({
                "MAE": "{:.2f}", "WAPE (%)": "{:.1f}", "Gain vs référence (%)": "{:+.1f}",
                "RMSE": "{:.2f}", "Biais": "{:+.2f}",
                "Couverture IC (%)": "{:.1f}", "Temps moyen (s)": "{:.2f}",
            }),
            width='stretch', hide_index=True,
        )

        failed = leaderboard[leaderboard["error"].notna()]
        for _, row in failed.iterrows():
            st.caption(f"⚠️ {FORECAST_MODEL_LABELS.get(row['model'], row['model'])} : {row['error']}")

        best_valid = leaderboard[leaderboard["mae"].notna()]
        if not best_valid.empty:
            best_model = best_valid.iloc[0]
            st.success(
                f"Meilleur modèle sur cette série (MAE la plus faible) : "
                f"**{FORECAST_MODEL_LABELS.get(best_model['model'], best_model['model'])}** "
                f"(MAE = {best_model['mae']:.2f}, couverture IC = {best_model['coverage']:.0f} %)."
            )
