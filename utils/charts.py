"""Plotly figures for the Home dashboard."""

from __future__ import annotations

from html import escape

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
    mapbox_viewport_from_geojson,
)
from utils.map_data import MapSource
from utils.metrics import CRIME_INDEX


PLOT_BG = "rgba(0,0,0,0)"
TEXT = "#dcecff"
MUTED = "#7890aa"
GRID = "rgba(79, 147, 194, 0.12)"
CYAN = "#38d9ff"


def _base_layout() -> dict:
    return {
        "paper_bgcolor": PLOT_BG,
        "plot_bgcolor": PLOT_BG,
        "font": {"family": "Inter, Arial, sans-serif", "color": TEXT},
        "margin": {"l": 10, "r": 20, "t": 18, "b": 28},
        "hoverlabel": {
            "bgcolor": "#081725",
            "bordercolor": "#1b6d91",
            "font": {"color": "#eef9ff", "family": "Inter, Arial, sans-serif"},
        },
        "showlegend": False,
    }


def build_top_five_chart(year_df: pd.DataFrame) -> go.Figure:
    """Horizontal ranking of the five highest existing annual index values."""
    top_five = (
        year_df.dropna(subset=[CRIME_INDEX])
        .nlargest(5, CRIME_INDEX)
        .sort_values(CRIME_INDEX, ascending=True)
    )
    values = top_five[CRIME_INDEX]
    colors = ["#177da8", "#1d91bd", "#24a7d0", "#2bc1e7", "#45e4ff"]

    figure = go.Figure(
        go.Bar(
            x=values,
            y=top_five["Municipio"],
            orientation="h",
            marker={
                "color": colors,
                "line": {"color": "rgba(94, 232, 255, 0.50)", "width": 1},
            },
            text=[f"{value:,.1f}".replace(",", "X").replace(".", ",").replace("X", ".") for value in values],
            textposition="outside",
            textfont={"color": "#dff8ff", "size": 12},
            customdata=top_five[["Ranking_criminal_anual", "Delitos_totales"]],
            hovertemplate=(
                "<b>%{y}</b><br>Índice: %{x:,.2f}"
                "<br>Ranking anual: #%{customdata[0]:.0f}"
                "<br>Delitos registrados: %{customdata[1]:,.0f}<extra></extra>"
            ),
        )
    )
    layout = _base_layout()
    layout.update(
        height=365,
        bargap=0.42,
        margin={"l": 12, "r": 70, "t": 18, "b": 35},
        xaxis={
            "title": {"text": "ÍNDICE PONDERADO / 10.000 HAB.", "font": {"size": 10, "color": MUTED}},
            "showgrid": True,
            "gridcolor": GRID,
            "zeroline": False,
            "tickfont": {"size": 10, "color": MUTED},
        },
        yaxis={
            "showgrid": False,
            "tickfont": {"size": 12, "color": TEXT},
            "automargin": True,
        },
    )
    figure.update_layout(**layout)
    return figure


def build_distribution_chart(year_df: pd.DataFrame) -> go.Figure:
    """Show the annual municipal index distribution and its median."""
    distribution = year_df.dropna(subset=[CRIME_INDEX]).copy()
    median = float(distribution[CRIME_INDEX].median())

    figure = go.Figure(
        go.Histogram(
            x=distribution[CRIME_INDEX],
            nbinsx=10,
            marker={
                "color": "rgba(33, 177, 222, 0.62)",
                "line": {"color": "rgba(84, 225, 255, 0.68)", "width": 1},
            },
            hovertemplate="Índice: %{x:,.1f}<br>Municipios: %{y}<extra></extra>",
        )
    )
    figure.add_vline(
        x=median,
        line_width=1.5,
        line_dash="dot",
        line_color="#ff5b68",
        annotation_text=f"Mediana {median:,.1f}".replace(",", "X").replace(".", ",").replace("X", "."),
        annotation_position="top right",
        annotation_font={"color": "#ff8b94", "size": 10},
    )
    layout = _base_layout()
    layout.update(
        height=365,
        bargap=0.08,
        xaxis={
            "title": {"text": "ÍNDICE PONDERADO / 10.000 HAB.", "font": {"size": 10, "color": MUTED}},
            "showgrid": False,
            "zeroline": False,
            "tickfont": {"size": 10, "color": MUTED},
        },
        yaxis={
            "title": {"text": "MUNICIPIOS", "font": {"size": 10, "color": MUTED}},
            "showgrid": True,
            "gridcolor": GRID,
            "zeroline": False,
            "dtick": 1,
            "tickfont": {"size": 10, "color": MUTED},
        },
    )
    figure.update_layout(**layout)
    return figure


def build_home_ranking_chart(
    territorial: pd.DataFrame,
    selected_municipality: str,
) -> go.Figure:
    """Top 5 regional o entorno inmediato del municipio seleccionado."""
    eligible = territorial.loc[territorial["eligible"]].sort_values("computed_rank")
    if selected_municipality == "TODOS LOS MUNICIPIOS":
        scope = eligible.head(5).sort_values("relative_metric")
    else:
        selected = eligible.loc[eligible["municipality"].eq(selected_municipality)]
        if selected.empty:
            scope = eligible.head(5).sort_values("relative_metric")
        else:
            rank = int(selected.iloc[0]["computed_rank"])
            scope = eligible.loc[
                eligible["computed_rank"].between(max(1, rank - 2), rank + 2)
            ].sort_values("relative_metric")
    colors = [
        "#ff6573" if name == selected_municipality else "#2bc1e7"
        for name in scope["municipality"]
    ]
    figure = go.Figure(
        go.Bar(
            x=scope["relative_metric"],
            y=scope["municipality"],
            orientation="h",
            marker={
                "color": colors,
                "line": {"color": "rgba(94, 232, 255, 0.50)", "width": 1},
            },
            text=[f"{value:,.1f}".replace(",", "X").replace(".", ",").replace("X", ".") for value in scope["relative_metric"]],
            textposition="outside",
            textfont={"color": "#dff8ff", "size": 11},
            cliponaxis=False,
            customdata=scope[["computed_rank", "crime_count", "population"]],
            hovertemplate=(
                "<b>%{y}</b><br>Valor relativo: %{x:,.2f}"
                "<br>Posición: #%{customdata[0]:.0f}"
                "<br>Delitos: %{customdata[1]:,.0f}"
                "<br>Población: %{customdata[2]:,.0f}<extra></extra>"
            ),
        )
    )
    layout = _base_layout()
    layout.update(
        height=410,
        bargap=.40,
        margin={"l": 12, "r": 72, "t": 18, "b": 42},
        xaxis={
            "title": {"text": "ÍNDICE CRIMINAL PONDERADO / 10.000 HAB.", "font": {"size": 9, "color": MUTED}},
            "gridcolor": GRID,
            "zeroline": False,
            "tickfont": {"size": 9, "color": MUTED},
        },
        yaxis={"automargin": True, "tickfont": {"size": 11, "color": TEXT}},
    )
    figure.update_layout(**layout)
    return figure



def build_home_bottom_ranking_chart(
    territorial: pd.DataFrame,
    selected_municipality: str,
) -> go.Figure:
    """Bottom 5 regional del índice criminal."""
    eligible = territorial.loc[territorial["eligible"]].copy()
    scope = (
        eligible.nsmallest(5, "relative_metric")
        .sort_values("relative_metric", ascending=False)
    )

    colors = [
        "#ff6573" if name == selected_municipality else "#f2b85b"
        for name in scope["municipality"]
    ]

    figure = go.Figure(
        go.Bar(
            x=scope["relative_metric"],
            y=scope["municipality"],
            orientation="h",
            marker={
                "color": colors,
                "line": {"color": "rgba(255, 214, 133, 0.55)", "width": 1},
            },
            text=[f"{value:,.1f}".replace(",", "X").replace(".", ",").replace("X", ".") for value in scope["relative_metric"]],
            textposition="outside",
            textfont={"color": "#fff0cf", "size": 11},
            cliponaxis=False,
            customdata=scope[["computed_rank", "crime_count", "population"]],
            hovertemplate=(
                "<b>%{y}</b><br>Índice criminal: %{x:,.2f}"
                "<br>Posición: #%{customdata[0]:.0f}"
                "<br>Delitos: %{customdata[1]:,.0f}"
                "<br>Población: %{customdata[2]:,.0f}<extra></extra>"
            ),
        )
    )

    regional_max = float(eligible["relative_metric"].max()) if not eligible.empty else 1.0
    layout = _base_layout()
    layout.update(
        height=410,
        bargap=.40,
        margin={"l": 12, "r": 72, "t": 18, "b": 42},
        xaxis={
            "title": {"text": "ÍNDICE CRIMINAL PONDERADO / 10.000 HAB.", "font": {"size": 9, "color": MUTED}},
            "gridcolor": GRID,
            "zeroline": False,
            "range": [0, regional_max * 1.08],
            "tickfont": {"size": 9, "color": MUTED},
        },
        yaxis={"automargin": True, "tickfont": {"size": 11, "color": TEXT}},
    )
    figure.update_layout(**layout)
    return figure

def _home_map_text(frame: pd.DataFrame) -> list[str]:
    texts: list[str] = []
    for _, row in frame.iterrows():
        if bool(row["eligible"]):
            texts.append(
                f"<b>{escape(str(row['municipality']))}</b>"
                f"<br>Índice territorial: {row['relative_metric']:,.2f} / 10.000 hab."
                f"<br>Delitos registrados: {row['crime_count']:,.0f}"
                f"<br>Segmento relativo: {escape(str(row['_risk_band']))}"
            )
        else:
            texts.append(
                f"<b>{escape(str(row['municipality']))}</b>"
                "<br>Fuera del universo criminal válido del periodo"
            )
    return texts


def build_home_mini_map(
    source: MapSource,
    selected_municipality: str,
) -> go.Figure:
    """Mapa ejecutivo de focos con los 37 polígonos y sin controles analíticos internos."""
    if not source.available or source.geojson is None:
        raise ValueError("No existe una fuente cartográfica disponible.")
    frame = classify_relative_risk(
        source.frame,
        value_column="relative_metric",
        eligibility_column="eligible",
    )
    frame["hover_text"] = _home_map_text(frame)
    eligible = frame.loc[frame["eligible"]].copy()
    ineligible = frame.loc[~frame["eligible"]].copy()
    figure = go.Figure()
    if not ineligible.empty:
        figure.add_trace(
            go.Choroplethmapbox(
                geojson=geojson_subset(source.geojson, ineligible["_feature_id"]),
                featureidkey="properties.feature_id",
                locations=ineligible["_feature_id"],
                z=np.zeros(len(ineligible)),
                colorscale=[[0, "#18232d"], [1, "#18232d"]],
                showscale=False,
                marker={"line": {"color": "rgba(68, 194, 226, .52)", "width": .7}},
                text=ineligible["hover_text"],
                hovertemplate="%{text}<extra></extra>",
                showlegend=False,
            )
        )
    if not eligible.empty:
        eligible["_risk_code"] = eligible["_risk_band"].map(RISK_CODES).astype(float)
        figure.add_trace(
            go.Choroplethmapbox(
                geojson=source.geojson,
                featureidkey="properties.feature_id",
                locations=eligible["_feature_id"],
                z=eligible["_risk_code"],
                zmin=0,
                zmax=len(RISK_ORDER) - 1,
                colorscale=RISK_COLORSCALE,
                showscale=False,
                marker={"line": {"color": "rgba(70, 218, 248, .70)", "width": .75}},
                text=eligible["hover_text"],
                hovertemplate="%{text}<extra></extra>",
                name="Municipios elegibles",
                showlegend=False,
            )
        )
        add_risk_legend_traces(figure, eligible["_risk_band"])
    if selected_municipality != "TODOS LOS MUNICIPIOS":
        selected = eligible.loc[eligible["municipality"].eq(selected_municipality)]
        if not selected.empty:
            selected_color = RISK_COLORS[str(selected.iloc[0]["_risk_band"])]
            figure.add_trace(
                go.Choroplethmapbox(
                    geojson=geojson_subset(source.geojson, selected["_feature_id"]),
                    featureidkey="properties.feature_id",
                    locations=selected["_feature_id"],
                    z=np.ones(len(selected)),
                    colorscale=[[0, selected_color], [1, selected_color]],
                    showscale=False,
                    marker={"line": {"color": "#f4fdff", "width": 3}},
                    text=selected["hover_text"],
                    hovertemplate="%{text}<extra></extra>",
                    showlegend=False,
                )
            )
    viewport = source.viewport or mapbox_viewport_from_geojson(
        source.geojson,
        width=720,
        height=410,
        padding_ratio=.09,
    )
    figure.update_layout(
        height=410,
        margin={"l": 0, "r": 0, "t": 0, "b": 0},
        paper_bgcolor="rgba(0,0,0,0)",
        dragmode=False,
        showlegend=True,
        legend={
            "orientation": "h",
            "yanchor": "bottom",
            "y": .01,
            "xanchor": "center",
            "x": .5,
            "title": {"text": "Nivel relativo"},
            "font": {"color": "#93adbd", "size": 9},
            "bgcolor": "rgba(2, 10, 16, .78)",
        },
        mapbox={"style": "carto-darkmatter", **viewport},
        hoverlabel={
            "bgcolor": "#071521",
            "bordercolor": "#238bb5",
            "font": {"color": TEXT, "family": "Inter, Arial, sans-serif"},
        },
    )
    return figure


def build_quarterly_evolution_chart(
    quarterly: pd.DataFrame,
    selected_year: int,
) -> go.Figure:
    """Serie histórica 2023–2025 con escala dinámica y sin forecast."""
    scope = quarterly.copy()

    if scope.empty:
        figure = go.Figure()
        figure.add_annotation(
            text="Sin serie histórica disponible para el filtro seleccionado.",
            x=.5, y=.5, xref="paper", yref="paper",
            showarrow=False,
            font={"color": MUTED, "size": 12},
        )
        layout = _base_layout()
        layout.update(height=440)
        figure.update_layout(**layout)
        return figure

    scope["variation"] = scope["value"].pct_change(fill_method=None) * 100
    variation_text = [
        f"{value:+.1f}%" if pd.notna(value) else "Sin periodo anterior comparable"
        for value in scope["variation"]
    ]
    customdata = np.column_stack([scope["quarter"], variation_text])

    valid_values = pd.to_numeric(scope["value"], errors="coerce").dropna()
    y_range = None
    if not valid_values.empty:
        y_min = float(valid_values.min())
        y_max = float(valid_values.max())
        spread = y_max - y_min
        padding = max(spread * 0.12, abs(y_max) * 0.03, 1.0)
        lower = max(0.0, y_min - padding)
        upper = y_max + padding
        if upper <= lower:
            upper = lower + max(abs(y_max) * 0.05, 1.0)
        y_range = [lower, upper]

    figure = go.Figure(
        go.Scatter(
            x=scope["period_label"],
            y=scope["value"],
            mode="lines+markers",
            connectgaps=False,
            line={"color": CYAN, "width": 3},
            marker={
                "size": 7,
                "color": "#dff8ff",
                "line": {"color": CYAN, "width": 1.5},
            },
            customdata=customdata,
            hovertemplate=(
                "<b>%{x}</b><br>Delitos registrados: %{y:,.0f}"
                "<br>Variación trimestral: %{customdata[1]}<extra></extra>"
            ),
            showlegend=False,
        )
    )

    layout = _base_layout()
    layout.update(
        height=440,
        margin={"l": 64, "r": 28, "t": 28, "b": 72},
        xaxis={
            "title": {"text": "PERIODO", "font": {"size": 9, "color": MUTED}},
            "tickangle": -35,
            "showgrid": False,
            "tickfont": {"size": 9, "color": MUTED},
            "categoryorder": "array",
            "categoryarray": scope["period_label"].tolist(),
        },
        yaxis={
            "title": {"text": "DELITOS REGISTRADOS", "font": {"size": 9, "color": MUTED}},
            "gridcolor": GRID,
            "zeroline": False,
            "range": y_range,
            "tickfont": {"size": 10, "color": TEXT},
        },
    )
    figure.update_layout(**layout)
    return figure

def build_change_drivers_chart(drivers: pd.DataFrame) -> go.Figure:
    """Contribuciones absolutas principales, evitando rankings porcentuales inestables."""
    figure = go.Figure()
    if drivers.empty:
        figure.add_annotation(
            text="Sin histórico comparable para calcular contribuciones.",
            x=.5,
            y=.5,
            xref="paper",
            yref="paper",
            showarrow=False,
            font={"color": MUTED, "size": 12},
        )
    else:
        scope = drivers.sort_values("change_absolute")
        colors = np.where(scope["change_absolute"].ge(0), CYAN, "#ff6573")
        figure.add_trace(
            go.Bar(
                x=scope["change_absolute"],
                y=scope["crime_type"],
                orientation="h",
                marker={"color": colors},
                customdata=scope[["previous_count", "current_count", "combined_volume"]],
                hovertemplate=(
                    "<b>%{y}</b><br>Contribución absoluta: %{x:+,.0f}"
                    "<br>Año anterior: %{customdata[0]:,.0f}"
                    "<br>Año seleccionado: %{customdata[1]:,.0f}<extra></extra>"
                ),
            )
        )
    layout = _base_layout()
    layout.update(
        height=330,
        margin={"l": 270, "r": 30, "t": 15, "b": 48},
        xaxis={
            "title": {"text": "CONTRIBUCIÓN ABSOLUTA AL CAMBIO", "font": {"size": 9, "color": MUTED}},
            "gridcolor": GRID,
            "zeroline": True,
            "zerolinecolor": "rgba(225,245,255,.45)",
        },
        yaxis={"automargin": True, "tickfont": {"size": 9, "color": TEXT}},
    )
    figure.update_layout(**layout)
    return figure
