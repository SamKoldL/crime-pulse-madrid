"""Visualizaciones Plotly para histórico, forecast y presión territorial."""

from __future__ import annotations

from html import escape
from textwrap import wrap
from typing import Any

import numpy as np
import pandas as pd
import plotly.graph_objects as go

from utils.map_charts import (
    RISK_CODES,
    RISK_COLORS,
    RISK_COLORSCALE,
    RISK_ORDER,
    add_risk_legend_traces,
    classify_relative_risk,
    geojson_subset,
)
from utils.map_data import MapSource
from utils.predictions_data import PREDICTION_KIND, REAL_KIND


CYAN = "#42dcff"
BLUE = "#328fc4"
RED = "#ff6070"
AMBER = "#efb65c"
TEXT = "#e8f5ff"
MUTED = "#aec1cf"
GRID = "rgba(91, 162, 194, .14)"


def _base_layout(height: int, *, left_margin: int = 70) -> dict[str, Any]:
    return {
        "height": height,
        "margin": {"l": left_margin, "r": 38, "t": 36, "b": 62},
        "paper_bgcolor": "rgba(0,0,0,0)",
        "plot_bgcolor": "rgba(0,0,0,0)",
        "font": {"family": "Inter, Arial, sans-serif", "color": TEXT},
        "hoverlabel": {
            "bgcolor": "#071521",
            "bordercolor": "#238bb5",
            "font": {"color": TEXT, "family": "Inter, Arial, sans-serif"},
        },
    }


def _wrapped(value: object, width: int = 34) -> str:
    return "<br>".join(wrap(str(value), width=width))


def build_history_forecast_chart(series: pd.DataFrame) -> go.Figure:
    """Línea real continua y forecast discontinuo conectado al último dato real."""
    real = series.loc[series["tipo_dato"].eq(REAL_KIND)].copy()
    predicted = series.loc[series["tipo_dato"].eq(PREDICTION_KIND)].copy()
    figure = go.Figure()
    figure.add_trace(
        go.Scatter(
            x=real["period_label"],
            y=real["valor"],
            mode="lines+markers",
            name="REAL",
            line={"color": CYAN, "width": 2.8},
            marker={"size": 7, "color": "#dff9ff", "line": {"color": CYAN, "width": 1.4}},
            customdata=real[["tipo_dato", "modelo", "nivel_confianza"]],
            hovertemplate=(
                "<b>%{x}</b><br>Valor: %{y:,.1f}"
                "<br>Tipo: %{customdata[0]}"
                "<br>Modelo: %{customdata[1]}"
                "<br>Confianza: %{customdata[2]}<extra></extra>"
            ),
        )
    )
    if not predicted.empty and not real.empty:
        connector = pd.concat([real.tail(1), predicted]).sort_values("period_index")
        figure.add_trace(
            go.Scatter(
                x=connector["period_label"],
                y=connector["valor"],
                mode="lines+markers",
                name="PREDICCIÓN",
                line={"color": RED, "width": 2.8, "dash": "dash"},
                marker={"size": 8, "color": RED, "line": {"color": "#ffd8dd", "width": 1}},
                customdata=connector[["tipo_dato", "modelo", "nivel_confianza"]],
                hovertemplate=(
                    "<b>%{x}</b><br>Valor: %{y:,.1f}"
                    "<br>Tipo: %{customdata[0]}"
                    "<br>Modelo: %{customdata[1]}"
                    "<br>Confianza: %{customdata[2]}<extra></extra>"
                ),
            )
        )
    layout = _base_layout(460)
    layout.update(
        legend={
            "orientation": "h",
            "yanchor": "bottom",
            "y": 1.02,
            "xanchor": "right",
            "x": 1,
            "font": {"color": MUTED, "size": 10},
        },
        xaxis={
            "title": {"text": "PERIODO", "font": {"color": MUTED, "size": 10}},
            "tickangle": -35,
            "tickfont": {"color": MUTED, "size": 9},
            "showgrid": False,
            "categoryorder": "trace",
        },
        yaxis={
            "title": {"text": "CONTEO", "font": {"color": MUTED, "size": 10}},
            "gridcolor": GRID,
            "zeroline": False,
            "rangemode": "tozero",
        },
    )
    figure.update_layout(**layout)
    return figure


def build_annual_comparison_chart(comparison: pd.DataFrame) -> go.Figure:
    variation_labels = [
        f"{value:+.1f}%" if pd.notna(value) else "Sin base"
        for value in comparison["change_percent"]
    ]
    hover_data = comparison[["actual_2025", "change_absolute"]].copy()
    hover_data["variation_text"] = variation_labels
    figure = go.Figure()
    figure.add_trace(
        go.Bar(
            x=comparison["trimestre"],
            y=comparison["actual_2025"],
            name="REAL 2025",
            marker={"color": BLUE},
            hovertemplate="<b>%{x} 2025</b><br>Real: %{y:,.1f}<extra></extra>",
        )
    )
    figure.add_trace(
        go.Bar(
            x=comparison["trimestre"],
            y=comparison["predicted_2026"],
            name="PREDICHO 2026",
            marker={"color": RED},
            text=variation_labels,
            textposition="outside",
            cliponaxis=False,
            customdata=hover_data,
            hovertemplate=(
                "<b>%{x} 2026</b><br>Predicho: %{y:,.1f}"
                "<br>Real 2025: %{customdata[0]:,.1f}"
                "<br>Diferencia: %{customdata[1]:+,.1f}"
                "<br>Variación: %{customdata[2]}<extra></extra>"
            ),
        )
    )
    layout = _base_layout(390)
    layout.update(
        barmode="group",
        bargap=.28,
        legend={
            "orientation": "h",
            "yanchor": "bottom",
            "y": 1.03,
            "xanchor": "right",
            "x": 1,
            "font": {"color": MUTED, "size": 10},
        },
        xaxis={"tickfont": {"color": TEXT}, "showgrid": False},
        yaxis={"gridcolor": GRID, "zeroline": False, "rangemode": "tozero"},
    )
    figure.update_layout(**layout)
    return figure


def build_type_trend_chart(trends: pd.DataFrame) -> go.Figure:
    scope = trends.sort_values("change_percent")
    colors = np.where(scope["change_percent"].ge(0), RED, BLUE)
    figure = go.Figure(
        go.Bar(
            x=scope["change_percent"],
            y=scope["tipo de crimen"].map(_wrapped),
            orientation="h",
            marker={"color": colors},
            customdata=scope[
                ["tipo de crimen", "real_2025", "predicted_2026", "change_absolute", "modelo", "nivel_confianza"]
            ],
            hovertemplate=(
                "<b>%{customdata[0]}</b><br>Variación: %{x:+.1f}%"
                "<br>Q1–Q3 2025: %{customdata[1]:,.1f}"
                "<br>Q1–Q3 2026: %{customdata[2]:,.1f}"
                "<br>Diferencia: %{customdata[3]:+,.1f}"
                "<br>Modelo: %{customdata[4]}"
                "<br>Confianza: %{customdata[5]}<extra></extra>"
            ),
        )
    )
    layout = _base_layout(690, left_margin=315)
    layout.update(
        showlegend=False,
        bargap=.30,
        xaxis={
            "title": {"text": "VARIACIÓN Q1–Q3 2025 → 2026", "font": {"color": MUTED, "size": 10}},
            "ticksuffix": "%",
            "gridcolor": GRID,
            "zeroline": True,
            "zerolinecolor": "rgba(230,247,255,.45)",
        },
        yaxis={"automargin": True, "tickfont": {"color": TEXT, "size": 9}},
    )
    figure.update_layout(**layout)
    return figure


def build_territorial_ranking_chart(
    territorial: pd.DataFrame,
    top_n: int = 12,
    selected_municipality: str | None = None,
) -> go.Figure:
    scope = territorial.nlargest(top_n, "predicted_count").sort_values("predicted_count")
    selected = scope["municipio"].eq(selected_municipality)
    display_names = np.where(
        selected,
        "◆ " + scope["municipio"].astype(str),
        scope["municipio"].astype(str),
    )
    hover_data = scope[["actual_2025", "change_absolute", "nivel_confianza"]].copy()
    hover_data["variation_text"] = [
        f"{value:+.1f}%" if pd.notna(value) else "Sin base comparable"
        for value in scope["change_percent"]
    ]
    figure = go.Figure(
        go.Bar(
            x=scope["predicted_count"],
            y=display_names,
            orientation="h",
            marker={
                "color": scope["predicted_count"],
                "colorscale": [[0, "#17618b"], [.65, CYAN], [1, RED]],
                "showscale": False,
                "line": {
                    "color": np.where(selected, "#ffffff", "rgba(255,255,255,0)"),
                    "width": np.where(selected, 2.4, 0),
                },
            },
            customdata=pd.concat(
                [scope[["municipio"]].reset_index(drop=True), hover_data.reset_index(drop=True)],
                axis=1,
            ),
            hovertemplate=(
                "<b>%{customdata[0]}</b><br>Predicho: %{x:,.1f}"
                "<br>Real 2025: %{customdata[1]:,.1f}"
                "<br>Diferencia: %{customdata[2]:+,.1f}"
                "<br>Variación: %{customdata[4]}"
                "<br>Confianza: %{customdata[3]}<extra></extra>"
            ),
        )
    )
    layout = _base_layout(500, left_margin=185)
    layout.update(
        showlegend=False,
        bargap=.30,
        xaxis={
            "title": {"text": "CONTEO PREVISTO", "font": {"color": MUTED, "size": 10}},
            "gridcolor": GRID,
            "zeroline": False,
            "rangemode": "tozero",
        },
        yaxis={"automargin": True, "tickfont": {"color": TEXT, "size": 9}},
    )
    figure.update_layout(**layout)
    return figure


def build_prediction_map(
    source: MapSource,
    crime_type: str,
    quarter: str,
    selected_municipality: str | None = None,
    *,
    include_madrid: bool = True,
) -> go.Figure:
    if not source.available or source.geojson is None:
        raise ValueError("No existe una fuente geográfica predictiva disponible.")
    frame = source.frame.copy()
    madrid_mask = frame["municipio"].eq("Madrid")
    active_frame = frame.loc[~madrid_mask].copy() if not include_madrid else frame.copy()
    active_frame["_eligible"] = True
    active_frame = classify_relative_risk(
        active_frame,
        value_column="predicted_count",
        eligibility_column="_eligible",
        output_column="_prediction_band",
    )
    active_frame["_prediction_code"] = active_frame["_prediction_band"].map(RISK_CODES).astype(float)

    # Reincorporamos Madrid solo como referencia cartográfica cuando está excluido
    # de la comparación territorial. Sus valores no participan en los quintiles.
    if not include_madrid and madrid_mask.any():
        madrid_frame = frame.loc[madrid_mask].copy()
        madrid_frame["_prediction_band"] = "No incluido"
        madrid_frame["_prediction_code"] = np.nan
        frame = pd.concat([active_frame, madrid_frame], ignore_index=True)
    else:
        frame = active_frame
    change_text = [
        f"{value:+.1f}%" if pd.notna(value) else "Sin base comparable"
        for value in frame["change_percent"]
    ]
    texts = [
        (
            f"<b>{escape(str(row['municipio']))}</b>"
            f"<br>Tipo: {escape(crime_type)}"
            f"<br>Horizonte: {quarter} 2026"
            f"<br>Conteo previsto: {row['predicted_count']:,.1f}"
            f"<br>Variación vs {quarter} 2025: {variation}"
            f"<br>Segmento visual: {escape(str(row['_prediction_band']))}"
            f"<br>Confianza: {escape(str(row['nivel_confianza']))}"
        )
        for (_, row), variation in zip(frame.iterrows(), change_text)
    ]
    active_plot_frame = frame.loc[~frame["_prediction_band"].eq("No incluido")].copy()
    active_texts = [
        text
        for text, band in zip(texts, frame["_prediction_band"])
        if band != "No incluido"
    ]
    figure = go.Figure(
        go.Choroplethmapbox(
            geojson=source.geojson,
            featureidkey="properties.feature_id",
            locations=active_plot_frame["_feature_id"],
            z=active_plot_frame["_prediction_code"],
            zmin=0,
            zmax=len(RISK_ORDER) - 1,
            colorscale=RISK_COLORSCALE,
            showscale=False,
            marker={"line": {"color": "rgba(78, 210, 241, .67)", "width": .75}},
            text=active_texts,
            hovertemplate="%{text}<extra></extra>",
            showlegend=False,
        )
    )
    add_risk_legend_traces(figure, active_plot_frame["_prediction_band"])

    if not include_madrid:
        madrid_plot = frame.loc[frame["_prediction_band"].eq("No incluido")].copy()
        if not madrid_plot.empty:
            madrid_text = [
                (
                    f"<b>{escape(str(row['municipio']))}</b>"
                    f"<br>Excluido de la comparación territorial"
                    f"<br>Activa «Incluir Madrid» para incorporarlo al análisis."
                )
                for _, row in madrid_plot.iterrows()
            ]
            figure.add_trace(
                go.Choroplethmapbox(
                    geojson=source.geojson,
                    featureidkey="properties.feature_id",
                    locations=madrid_plot["_feature_id"],
                    z=[0] * len(madrid_plot),
                    zmin=0,
                    zmax=1,
                    colorscale=[[0, "#586875"], [1, "#586875"]],
                    showscale=False,
                    marker={"line": {"color": "rgba(180, 200, 214, .72)", "width": 1.2}},
                    text=madrid_text,
                    hovertemplate="%{text}<extra></extra>",
                    showlegend=False,
                )
            )
    selected = frame.loc[frame["municipio"].eq(str(selected_municipality))]
    if not selected.empty:
        selected_color = RISK_COLORS[str(selected.iloc[0]["_prediction_band"])]
        selected_position = int(frame.index.get_loc(selected.index[0]))
        figure.add_trace(
            go.Choroplethmapbox(
                geojson=geojson_subset(source.geojson, selected["_feature_id"]),
                featureidkey="properties.feature_id",
                locations=selected["_feature_id"],
                z=[1],
                colorscale=[[0, selected_color], [1, selected_color]],
                showscale=False,
                marker={"line": {"color": "#f4fdff", "width": 3}},
                text=[texts[selected_position]],
                hovertemplate="%{text}<extra></extra>",
                showlegend=False,
            )
        )
    base_viewport = source.territorial_viewport or source.viewport or {
        "center": source.center or {"lon": -3.7, "lat": 40.4},
        "zoom": 7.15,
    }
    # El encuadre parte del bounding box compartido, pero aprovecha el formato
    # más compacto de esta columna para reducir contexto geográfico exterior.
    viewport = {
        "center": dict(base_viewport["center"]),
        "zoom": min(float(base_viewport["zoom"]) + 0.05, 20.0),
    }
    figure.update_layout(
        height=560,
        margin={"l": 0, "r": 0, "t": 0, "b": 0},
        paper_bgcolor="rgba(0,0,0,0)",
        uirevision=f"prediction-map-tight-{crime_type}-{quarter}",
        dragmode=False,
        showlegend=True,
        legend={
            "orientation": "h",
            "yanchor": "bottom",
            "y": .01,
            "xanchor": "center",
            "x": .5,
            "title": {"text": "Quintil visual de volumen"},
            "font": {"color": MUTED, "size": 10},
            "bgcolor": "rgba(2,10,16,.76)",
        },
        mapbox={"style": "carto-darkmatter", **viewport},
        hoverlabel={
            "bgcolor": "#071521",
            "bordercolor": "#238bb5",
            "font": {"color": TEXT, "family": "Inter, Arial, sans-serif"},
        },
    )
    return figure


def build_emerging_trend_chart(comparison: pd.DataFrame) -> go.Figure:
    figure = go.Figure()
    figure.add_trace(
        go.Bar(
            x=comparison["trimestre"],
            y=comparison["actual_2025"],
            name="REAL 2025",
            marker={"color": BLUE},
            hovertemplate="<b>%{x} 2025</b><br>Conteo: %{y:,.0f}<extra></extra>",
        )
    )
    figure.add_trace(
        go.Bar(
            x=comparison["trimestre"],
            y=comparison["predicted_2026"],
            name="PREDICHO 2026",
            marker={"color": RED},
            hovertemplate="<b>%{x} 2026</b><br>Conteo previsto: %{y:,.1f}<extra></extra>",
        )
    )
    layout = _base_layout(320)
    layout.update(
        barmode="group",
        bargap=.25,
        legend={
            "orientation": "h",
            "yanchor": "bottom",
            "y": 1.03,
            "xanchor": "right",
            "x": 1,
            "font": {"color": MUTED, "size": 10},
        },
        xaxis={"showgrid": False},
        yaxis={"gridcolor": GRID, "zeroline": False, "rangemode": "tozero"},
    )
    figure.update_layout(**layout)
    return figure
