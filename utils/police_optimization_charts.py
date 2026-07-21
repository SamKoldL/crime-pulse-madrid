"""Visualizaciones Plotly de los escenarios de optimización policial."""

from __future__ import annotations

from html import escape
from typing import Any

import numpy as np
import pandas as pd
import plotly.graph_objects as go

from utils.map_data import MapSource


CYAN = "#42dcff"
BLUE = "#3d8dcc"
RED = "#ff6573"
AMBER = "#f2b85b"
NEUTRAL = "#72899a"
TEXT = "#e8f5ff"
MUTED = "#aec1cf"
GRID = "rgba(91, 162, 194, .14)"

ALIGNMENT_COLORS = {
    "Presión proporcionalmente mayor": AMBER,
    "Equilibrio relativo": NEUTRAL,
    "Dotación proporcionalmente mayor": BLUE,
}
TRANSFER_COLORS = {
    "Recibiría": RED,
    "Sin cambio": NEUTRAL,
    "Cedería": BLUE,
}


def _base_layout(height: int, *, left_margin: int = 70) -> dict[str, Any]:
    return {
        "height": height,
        "margin": {"l": left_margin, "r": 35, "t": 34, "b": 58},
        "paper_bgcolor": "rgba(0,0,0,0)",
        "plot_bgcolor": "rgba(0,0,0,0)",
        "font": {"family": "Inter, Arial, sans-serif", "color": TEXT},
        "hoverlabel": {
            "bgcolor": "#071521",
            "bordercolor": "#238bb5",
            "font": {"color": TEXT, "family": "Inter, Arial, sans-serif"},
        },
    }


def _visual_scope(frame: pd.DataFrame, include_madrid: bool) -> pd.DataFrame:
    if include_madrid:
        return frame.copy()
    return frame.loc[frame["municipality"].ne("Madrid")].copy()


def build_alignment_scatter(
    frame: pd.DataFrame,
    include_madrid: bool = False,
    selected_municipality: str | None = None,
) -> go.Figure:
    """Compara cuota de presión y cuota policial con diagonal de alineación."""
    scope = _visual_scope(frame, include_madrid)
    figure = go.Figure()
    max_axis = float(
        max(scope["pressure_share"].max(), scope["police_share"].max()) * 100 * 1.08
    )
    figure.add_trace(
        go.Scatter(
            x=[0, max_axis],
            y=[0, max_axis],
            mode="lines",
            line={"color": "rgba(175, 205, 219, .42)", "dash": "dash", "width": 1.3},
            hoverinfo="skip",
            showlegend=False,
        )
    )
    for alignment, color in ALIGNMENT_COLORS.items():
        group = scope.loc[scope["alignment"].eq(alignment)].copy()
        if group.empty:
            continue
        sizes = np.clip(np.sqrt(group["current_police"]) * 2.2, 9, 32)
        group["gap_pp"] = group["gap"] * 100
        is_selected = group["municipality"].eq(selected_municipality)
        figure.add_trace(
            go.Scatter(
                x=group["pressure_share"] * 100,
                y=group["police_share"] * 100,
                mode="markers",
                name=alignment,
                marker={
                    "size": sizes,
                    "color": color,
                    "opacity": np.where(is_selected, 1.0, .78),
                    "line": {
                        "color": np.where(is_selected, "#ffffff", "rgba(229, 248, 255, .65)"),
                        "width": np.where(is_selected, 3.0, .8),
                    },
                },
                customdata=group[
                    [
                        "municipality",
                        "pressure_share",
                        "police_share",
                        "gap_pp",
                        "current_police",
                        "proposed_police",
                        "transfer",
                    ]
                ],
                hovertemplate=(
                    "<b>%{customdata[0]}</b>"
                    "<br>Presión criminal: %{customdata[1]:.2%}"
                    "<br>Cuota policial: %{customdata[2]:.2%}"
                    "<br>Brecha: %{customdata[3]:+.2f} pp"
                    "<br>Actuales: %{customdata[4]:,.0f}"
                    "<br>Propuestos: %{customdata[5]:,.0f}"
                    "<br>Transferencia: %{customdata[6]:+,.0f}<extra></extra>"
                ),
            )
        )
    layout = _base_layout(520)
    layout.update(
        legend={
            "orientation": "h",
            "yanchor": "bottom",
            "y": 1.01,
            "xanchor": "left",
            "x": 0,
            "font": {"color": MUTED, "size": 10},
        },
        xaxis={
            "title": {"text": "% DE PRESIÓN CRIMINAL", "font": {"color": MUTED, "size": 10}},
            "ticksuffix": "%",
            "gridcolor": GRID,
            "zeroline": False,
            "rangemode": "tozero",
        },
        yaxis={
            "title": {"text": "% DE POLICÍAS", "font": {"color": MUTED, "size": 10}},
            "ticksuffix": "%",
            "gridcolor": GRID,
            "zeroline": False,
            "rangemode": "tozero",
            "scaleanchor": "x",
            "scaleratio": 1,
        },
    )
    figure.update_layout(**layout)
    return figure


def build_gap_ranking_chart(
    frame: pd.DataFrame,
    count_each: int = 8,
    include_madrid: bool = False,
    selected_municipality: str | None = None,
) -> go.Figure:
    """Muestra los extremos de brecha sin convertir la vista en una tabla masiva."""
    visual_frame = _visual_scope(frame, include_madrid)
    deficit = visual_frame.nsmallest(count_each, "gap")
    surplus = visual_frame.nlargest(count_each, "gap")
    scope = (
        pd.concat([deficit, surplus])
        .drop_duplicates("municipality")
        .sort_values("gap")
    )
    colors = scope["alignment"].map(ALIGNMENT_COLORS).fillna(NEUTRAL)
    selected = scope["municipality"].eq(selected_municipality)
    display_names = np.where(
        selected,
        "◆ " + scope["municipality"].astype(str),
        scope["municipality"].astype(str),
    )
    figure = go.Figure(
        go.Bar(
            x=scope["gap"] * 100,
            y=display_names,
            orientation="h",
            marker={
                "color": colors,
                "line": {
                    "color": np.where(selected, "#ffffff", "rgba(225,245,255,.35)"),
                    "width": np.where(selected, 2.4, .5),
                },
            },
            customdata=scope[
                [
                    "municipality",
                    "pressure_share",
                    "police_share",
                    "current_police",
                    "proposed_police",
                    "transfer",
                ]
            ],
            hovertemplate=(
                "<b>%{customdata[0]}</b><br>Brecha: %{x:+.2f} pp"
                "<br>Presión: %{customdata[1]:.2%}"
                "<br>Policías: %{customdata[2]:.2%}"
                "<br>Actuales: %{customdata[3]:,.0f}"
                "<br>Propuestos: %{customdata[4]:,.0f}"
                "<br>Transferencia: %{customdata[5]:+,.0f}<extra></extra>"
            ),
        )
    )
    layout = _base_layout(560, left_margin=205)
    layout.update(
        showlegend=False,
        bargap=.32,
        xaxis={
            "title": {"text": "BRECHA · PUNTOS PORCENTUALES", "font": {"color": MUTED, "size": 10}},
            "ticksuffix": " pp",
            "gridcolor": GRID,
            "zeroline": True,
            "zerolinecolor": "rgba(225,245,255,.45)",
        },
        yaxis={"automargin": True, "tickfont": {"color": TEXT, "size": 10}},
    )
    figure.update_layout(**layout)
    return figure


def build_transfer_ranking_chart(
    frame: pd.DataFrame,
    count_each: int = 6,
    include_madrid: bool = False,
    selected_municipality: str | None = None,
) -> go.Figure:
    visual_frame = _visual_scope(frame, include_madrid)
    receivers = visual_frame.loc[visual_frame["transfer"].gt(0)].nlargest(count_each, "transfer")
    ceders = visual_frame.loc[visual_frame["transfer"].lt(0)].nsmallest(count_each, "transfer")
    scope = pd.concat([ceders, receivers]).sort_values("transfer")
    colors = [BLUE if value < 0 else RED for value in scope["transfer"]]
    selected = scope["municipality"].eq(selected_municipality)
    display_names = np.where(
        selected,
        "◆ " + scope["municipality"].astype(str),
        scope["municipality"].astype(str),
    )
    figure = go.Figure(
        go.Bar(
            x=scope["transfer"],
            y=display_names,
            orientation="h",
            marker={
                "color": colors,
                "line": {
                    "color": np.where(selected, "#ffffff", "rgba(255,255,255,0)"),
                    "width": np.where(selected, 2.4, 0),
                },
            },
            customdata=scope[
                ["municipality", "current_police", "proposed_police", "pressure_share"]
            ],
            hovertemplate=(
                "<b>%{customdata[0]}</b><br>Transferencia: %{x:+,.0f}"
                "<br>Actuales: %{customdata[1]:,.0f}"
                "<br>Propuestos: %{customdata[2]:,.0f}"
                "<br>Presión: %{customdata[3]:.2%}<extra></extra>"
            ),
        )
    )
    layout = _base_layout(620, left_margin=175)
    layout.update(
        showlegend=False,
        xaxis={
            "title": {"text": "TRANSFERENCIA TEÓRICA · AGENTES", "font": {"color": MUTED, "size": 9}},
            "gridcolor": GRID,
            "zeroline": True,
            "zerolinecolor": "rgba(225,245,255,.45)",
        },
        yaxis={"automargin": True, "tickfont": {"color": TEXT, "size": 9}},
    )
    figure.update_layout(**layout)
    return figure


def build_transfer_map(
    source: MapSource,
    selected_municipality: str | None = None,
) -> go.Figure:
    """Coropleta discreta de recepción, cesión o mantenimiento de efectivos."""
    if not source.available or source.geojson is None:
        raise ValueError("No existe una fuente geográfica poligonal disponible.")
    frame = source.frame.copy()
    frame["transfer_band"] = np.select(
        [frame["transfer"].gt(0), frame["transfer"].lt(0)],
        ["Recibiría", "Cedería"],
        default="Sin cambio",
    )
    transfer_codes = {"Cedería": -1, "Sin cambio": 0, "Recibiría": 1}
    frame["transfer_code"] = frame["transfer_band"].map(transfer_codes).astype(int)
    texts = [
        (
            f"<b>{escape(str(row['municipality']))}</b>"
            f"<br>Policías actuales: {int(row['current_police']):,}"
            f"<br>Policías propuestos: {int(row['proposed_police']):,}"
            f"<br>Transferencia: {int(row['transfer']):+d}"
            f"<br>Cuota de presión: {row['pressure_share']:.2%}"
            f"<br>Cuota policial: {row['police_share']:.2%}"
        )
        for _, row in frame.iterrows()
    ]
    discrete_scale = [
        [0.0, TRANSFER_COLORS["Cedería"]],
        [0.2499, TRANSFER_COLORS["Cedería"]],
        [0.25, TRANSFER_COLORS["Sin cambio"]],
        [0.7499, TRANSFER_COLORS["Sin cambio"]],
        [0.75, TRANSFER_COLORS["Recibiría"]],
        [1.0, TRANSFER_COLORS["Recibiría"]],
    ]
    figure = go.Figure(
        go.Choroplethmapbox(
            geojson=source.geojson,
            featureidkey="properties.feature_id",
            locations=frame["_feature_id"],
            z=frame["transfer_code"],
            zmin=-1,
            zmax=1,
            colorscale=discrete_scale,
            showscale=False,
            marker={"line": {"color": "rgba(81, 211, 240, .66)", "width": .75}},
            text=texts,
            customdata=frame["municipality"],
            hovertemplate="%{text}<extra></extra>",
            showlegend=False,
        )
    )
    for band in ("Recibiría", "Sin cambio", "Cedería"):
        figure.add_trace(
            go.Scattermapbox(
                lon=[None],
                lat=[None],
                mode="markers",
                marker={"size": 9, "color": TRANSFER_COLORS[band]},
                name=band,
                hoverinfo="skip",
                showlegend=True,
            )
        )
    selected = frame.loc[frame["municipality"].eq(selected_municipality)]
    if not selected.empty:
        selected_id = str(selected.iloc[0]["_feature_id"])
        # La capa de foco solo serializa el polÃ­gono seleccionado. Reutilizar el
        # FeatureCollection completo aquÃ­ duplicarÃ­a innecesariamente el GeoJSON.
        selected_geojson = {
            "type": "FeatureCollection",
            "features": [
                feature
                for feature in source.geojson["features"]
                if str(feature["properties"].get("feature_id")) == selected_id
            ],
        }
        figure.add_trace(
            go.Choroplethmapbox(
                geojson=selected_geojson,
                featureidkey="properties.feature_id",
                locations=[selected_id],
                z=[1],
                colorscale=[[0, "rgba(0,0,0,0)"], [1, "rgba(0,0,0,0)"]],
                showscale=False,
                marker={"line": {"color": "#ffffff", "width": 3.2}},
                hoverinfo="skip",
                showlegend=False,
            )
        )
    viewport = source.territorial_viewport or {
        "center": source.center or {"lon": -3.7, "lat": 40.4},
        "zoom": 7.15,
    }
    figure.update_layout(
        height=610,
        margin={"l": 0, "r": 0, "t": 0, "b": 0},
        paper_bgcolor="rgba(0,0,0,0)",
        showlegend=True,
        legend={
            "orientation": "h",
            "yanchor": "bottom",
            "y": .01,
            "xanchor": "center",
            "x": .5,
            "font": {"color": MUTED, "size": 10},
            "bgcolor": "rgba(2,10,16,.76)",
        },
        dragmode=False,
        mapbox={"style": "carto-darkmatter", **viewport},
        hoverlabel={
            "bgcolor": "#071521",
            "bordercolor": "#238bb5",
            "font": {"color": TEXT, "family": "Inter, Arial, sans-serif"},
        },
    )
    return figure


def build_comparison_scatter(
    comparison: pd.DataFrame,
    include_madrid: bool = False,
    selected_municipality: str | None = None,
) -> go.Figure:
    scope = _visual_scope(comparison, include_madrid)
    axis_min = float(min(scope["volume_transfer"].min(), scope["weighted_transfer"].min()))
    axis_max = float(max(scope["volume_transfer"].max(), scope["weighted_transfer"].max()))
    padding = max(4.0, (axis_max - axis_min) * .08)
    axis_range = [axis_min - padding, axis_max + padding]
    colors = np.where(scope["sign_switch"], RED, np.where(scope["significant_change"], AMBER, CYAN))
    selected = scope["municipality"].eq(selected_municipality)
    labels = np.where(
        scope["sign_switch"] | scope["significant_change"] | selected,
        scope["municipality"],
        "",
    )
    figure = go.Figure()
    figure.add_trace(
        go.Scatter(
            x=axis_range,
            y=axis_range,
            mode="lines",
            line={"color": "rgba(195,221,232,.48)", "dash": "dash", "width": 1.3},
            hoverinfo="skip",
            showlegend=False,
        )
    )
    figure.add_trace(
        go.Scatter(
            x=scope["volume_transfer"],
            y=scope["weighted_transfer"],
            mode="markers+text",
            text=labels,
            textposition="top center",
            textfont={"color": MUTED, "size": 9},
            marker={
                "size": np.where(selected, 18, np.where(scope["sign_switch"], 15, 11)),
                "color": colors,
                "opacity": np.where(selected, 1.0, .82),
                "line": {
                    "color": np.where(selected, "#ffffff", "rgba(235,250,255,.7)"),
                    "width": np.where(selected, 3.0, .8),
                },
            },
            customdata=scope[
                ["municipality", "current_police", "transfer_difference", "sign_switch"]
            ],
            hovertemplate=(
                "<b>%{customdata[0]}</b>"
                "<br>Transferencia por volumen: %{x:+,.0f}"
                "<br>Transferencia ponderada: %{y:+,.0f}"
                "<br>Diferencia: %{customdata[2]:+,.0f}"
                "<br>Plantilla actual: %{customdata[1]:,.0f}<extra></extra>"
            ),
            showlegend=False,
        )
    )
    layout = _base_layout(560)
    layout.update(
        xaxis={
            "title": {"text": "TRANSFERENCIA · VOLUMEN", "font": {"color": MUTED, "size": 10}},
            "range": axis_range,
            "gridcolor": GRID,
            "zeroline": True,
            "zerolinecolor": "rgba(225,245,255,.36)",
        },
        yaxis={
            "title": {"text": "TRANSFERENCIA · GRAVEDAD", "font": {"color": MUTED, "size": 10}},
            "range": axis_range,
            "gridcolor": GRID,
            "zeroline": True,
            "zerolinecolor": "rgba(225,245,255,.36)",
            "scaleanchor": "x",
            "scaleratio": 1,
        },
    )
    figure.update_layout(**layout)
    return figure
