"""
ui/charts.py — Fonctions de rendu Plotly.

Plotly est utilisé (plutôt que matplotlib) pour l'interactivité et le
drill-down par profil, comme demandé section 5.1 du cadrage.
"""

from __future__ import annotations

import plotly.express as px
import plotly.graph_objects as go
import pandas as pd


def _hex_to_rgba(hex_color: str, alpha: float = 0.15) -> str:
    hex_color = hex_color.lstrip("#")
    r, g, b = (int(hex_color[i : i + 2], 16) for i in (0, 2, 4))
    return f"rgba({r},{g},{b},{alpha})"


PALETTE = px.colors.qualitative.Plotly


def build_lift_bar_chart(snapshot_df: pd.DataFrame, value_col: str = "lift") -> go.Figure:
    """Barres triées par lift décroissant, avec intervalle de confiance et
    grisage des profils sous le seuil de volume minimum (non fiables)."""
    df = snapshot_df.sort_values(value_col, ascending=False).copy()

    colors = []
    for _, row in df.iterrows():
        if not row.get("is_reliable", True):
            colors.append("#c7c7c7")  # gris : volume insuffisant, non fiable
        elif row[value_col] > 1:
            colors.append("#d62728")  # sur-représenté
        else:
            colors.append("#1f77b4")  # sous-représenté ou conforme

    fig = go.Figure()
    error_y = None
    if "lift_ci_low" in df.columns and "lift_ci_high" in df.columns:
        error_y = dict(
            type="data",
            symmetric=False,
            array=(df["lift_ci_high"] - df[value_col]).clip(lower=0),
            arrayminus=(df[value_col] - df["lift_ci_low"]).clip(lower=0),
        )

    fig.add_bar(
        x=df["profil_partenaire"], y=df[value_col],
        error_y=error_y, marker_color=colors,
        text=[f"{v:.2f}" for v in df[value_col]], textposition="outside",
    )
    fig.add_hline(y=1.0, line_dash="dash", line_color="gray")
    fig.update_layout(
        title="Indice de représentation (lift) par profil — trié décroissant",
        yaxis_title="Lift (1 = conforme au poids du profil)",
        xaxis_title="Profil",
        showlegend=False,
    )
    return fig


def build_time_series_chart(
    df: pd.DataFrame,
    profiles: list[str],
    value_col: str,
    y_title: str,
    ci_low_col: str | None = None,
    ci_high_col: str | None = None,
    alert_col: str | None = None,
    reference_line: float | None = None,
) -> go.Figure:
    """Évolution temporelle du taux/lift pour les profils sélectionnés, avec
    bande d'incertitude optionnelle et marquage des points en alerte."""
    fig = go.Figure()

    for i, pid in enumerate(profiles):
        sub = df[df["profil_partenaire"] == pid].sort_values("period")
        color = PALETTE[i % len(PALETTE)]

        if ci_low_col and ci_high_col and ci_low_col in sub and ci_high_col in sub:
            band = sub.dropna(subset=[ci_low_col, ci_high_col])
            if not band.empty:
                fig.add_scatter(
                    x=pd.concat([band["period"], band["period"][::-1]]),
                    y=pd.concat([band[ci_high_col], band[ci_low_col][::-1]]),
                    fill="toself", fillcolor=_hex_to_rgba(color, 0.12),
                    line=dict(width=0), hoverinfo="skip", showlegend=False, name=f"{pid} (IC)",
                )

        fig.add_scatter(
            x=sub["period"], y=sub[value_col], mode="lines+markers",
            name=pid, line=dict(color=color),
        )

        if alert_col and alert_col in sub.columns:
            alert_pts = sub[sub[alert_col] == True]  # noqa: E712
            if not alert_pts.empty:
                fig.add_scatter(
                    x=alert_pts["period"], y=alert_pts[value_col], mode="markers",
                    marker=dict(size=12, symbol="x", color="black"),
                    name=f"{pid} — alerte", showlegend=False,
                )

    if reference_line is not None:
        fig.add_hline(y=reference_line, line_dash="dash", line_color="gray")

    fig.update_layout(
        title="Évolution temporelle",
        yaxis_title=y_title,
        xaxis_title="Période",
        legend_title="Profil",
        hovermode="x unified",
    )
    return fig


def build_forecast_chart(
    history: pd.Series, forecast_df: pd.DataFrame, title: str, y_title: str = "Volume d'appels"
) -> go.Figure:
    """Historique + prévision avec bande d'incertitude."""
    fig = go.Figure()

    fig.add_scatter(
        x=history.index, y=history.to_numpy(), mode="lines+markers",
        name="Historique", line=dict(color="#1f77b4"),
    )

    band_x = pd.concat([forecast_df["ds"], forecast_df["ds"][::-1]])
    band_y = pd.concat([forecast_df["yhat_upper"], forecast_df["yhat_lower"][::-1]])
    fig.add_scatter(
        x=band_x, y=band_y, fill="toself", fillcolor=_hex_to_rgba("#ff7f0e", 0.15),
        line=dict(width=0), hoverinfo="skip", showlegend=False, name="Intervalle",
    )
    fig.add_scatter(
        x=forecast_df["ds"], y=forecast_df["yhat"], mode="lines+markers",
        name="Prévision", line=dict(color="#ff7f0e", dash="dash"),
    )

    # ligne verticale séparant historique / prévision
    if len(history) > 0:
        fig.add_vline(x=history.index[-1], line_dash="dot", line_color="gray")

    fig.update_layout(title=title, yaxis_title=y_title, xaxis_title="Période", hovermode="x unified")
    return fig
