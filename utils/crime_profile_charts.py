"""Visualizaciones Plotly del Perfil Delictivo."""

from __future__ import annotations

from textwrap import wrap

import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from utils.crime_profile_data import (
    ANNUAL_AVERAGE_VIEW,
    ANNUAL_VIEW,
    GROUP_LEVEL,
    IMPACT_VIEW,
    PERIOD_VARIATION_VIEW,
    QUARTERLY_VIEW,
    TYPE_LEVEL,
    DrugCase,
)


PAPER_BG = "rgba(0,0,0,0)"
TEXT = "#e8f5ff"
MUTED = "#aec1cf"
GRID = "rgba(79, 147, 194, 0.15)"
CYAN = "#42dcff"
BLUE = "#168fc4"
RED = "#ff6170"
AMBER = "#f2b84b"
PALETTE = ["#42dcff", "#ff6170", "#9d8cff", "#42debc", "#f2b84b", "#5ca8ff"]


def _base_layout(height: int, left_margin: int = 42) -> dict:
    return {
        "height": height,
        "paper_bgcolor": PAPER_BG,
        "plot_bgcolor": PAPER_BG,
        "font": {"family": "Inter, Arial, sans-serif", "color": TEXT},
        "margin": {"l": left_margin, "r": 42, "t": 34, "b": 54},
        "hoverlabel": {
            "bgcolor": "#071521",
            "bordercolor": "#238bb5",
            "font": {"color": "#f1fbff", "family": "Inter, Arial, sans-serif"},
        },
    }


def _wrapped_label(value: object, width: int = 38) -> str:
    return "<br>".join(wrap(str(value), width=width))


def _metric(view: str) -> tuple[str, str, str]:
    if view == IMPACT_VIEW:
        return "weighted_count", "CARGA DELICTIVA PONDERADA", RED
    return "count", "CONTEO", CYAN


def build_crime_ranking_chart(
    summary: pd.DataFrame,
    view: str,
    selected_crime_id: str | None = None,
) -> go.Figure:
    """Ranking de los 16 IDs con etiquetas completas y tooltip metodológico."""
    metric, axis_title, accent = _metric(view)
    ranked = summary.sort_values(metric, ascending=True).copy()
    ranked["display_label"] = ranked.apply(
        lambda row: _wrapped_label(f"{row['crime_id']} · {row['crime_type']}", 44), axis=1
    )
    colors = [
        RED if selected_crime_id and str(crime_id) == str(selected_crime_id) else accent
        for crime_id in ranked["crime_id"]
    ]
    if colors and selected_crime_id is None:
        colors[-1] = RED if view != IMPACT_VIEW else "#ff9aa3"

    figure = go.Figure(
        go.Bar(
            x=ranked[metric],
            y=ranked["display_label"],
            orientation="h",
            marker={
                "color": colors,
                "line": {"color": "rgba(132, 230, 255, .42)", "width": 1},
            },
            text=[f"{value:,.0f}" for value in ranked[metric]],
            textposition="outside",
            textfont={"color": "#e2f7ff", "size": 10},
            cliponaxis=False,
            customdata=ranked[
                [
                    "crime_id",
                    "crime_type",
                    "group",
                    "count",
                    "weight",
                    "weighted_count",
                    "share_total",
                ]
            ],
            hovertemplate=(
                "<b>%{customdata[1]}</b><br>ID: %{customdata[0]}"
                "<br>Grupo: %{customdata[2]}"
                "<br>Conteo: %{customdata[3]:,.0f}"
                "<br>Peso gravedad: %{customdata[4]:.1f}"
                "<br>Conteo ponderado: %{customdata[5]:,.0f}"
                "<br>% del total: %{customdata[6]:.2f}%<extra></extra>"
            ),
        )
    )
    layout = _base_layout(max(520, 31 * len(ranked) + 105), left_margin=330)
    layout.update(
        showlegend=False,
        bargap=0.32,
        xaxis={
            "title": {"text": axis_title, "font": {"color": MUTED, "size": 10}},
            "tickfont": {"color": MUTED, "size": 10},
            "gridcolor": GRID,
            "zeroline": False,
        },
        yaxis={"tickfont": {"color": TEXT, "size": 10}, "automargin": True},
    )
    figure.update_layout(**layout)
    return figure


def build_group_composition_chart(
    summary: pd.DataFrame,
    view: str,
    selected_group: str | None = None,
) -> go.Figure:
    """Composición por los grupos exactos de la hoja Peso Crimen."""
    metric, axis_title, accent = _metric(view)
    ranked = summary.sort_values(metric, ascending=True).copy()
    ranked["display_label"] = ranked["group"].map(lambda value: _wrapped_label(value, 30))
    figure = go.Figure(
        go.Bar(
            x=ranked[metric],
            y=ranked["display_label"],
            orientation="h",
            marker={
                "color": [
                    RED if selected_group and group == selected_group else accent
                    for group in ranked["group"]
                ],
                "line": {"color": "rgba(115, 222, 251, .42)", "width": 1},
            },
            text=[f"{value:,.0f}" for value in ranked[metric]],
            textposition="outside",
            textfont={"color": "#e2f7ff", "size": 11},
            cliponaxis=False,
            customdata=ranked[["group", "count", "weighted_count", "share_total"]],
            hovertemplate=(
                "<b>%{customdata[0]}</b><br>Conteo: %{customdata[1]:,.0f}"
                "<br>Carga ponderada: %{customdata[2]:,.0f}"
                "<br>% del total: %{customdata[3]:.2f}%<extra></extra>"
            ),
        )
    )
    layout = _base_layout(460, left_margin=250)
    layout.update(
        showlegend=False,
        bargap=0.38,
        xaxis={
            "title": {"text": axis_title, "font": {"color": MUTED, "size": 10}},
            "tickfont": {"color": MUTED, "size": 10},
            "gridcolor": GRID,
            "zeroline": False,
        },
        yaxis={"tickfont": {"color": TEXT, "size": 11}, "automargin": True},
    )
    figure.update_layout(**layout)
    return figure


def build_time_series_chart(
    series: pd.DataFrame,
    level: str,
    cadence: str,
) -> go.Figure:
    """Evolución observada trimestral o anual para entidades seleccionadas."""
    label_column = "type_label" if level == TYPE_LEVEL else "group"
    x_column = "period_label" if cadence == QUARTERLY_VIEW else "year"
    figure = go.Figure()
    for position, (label, entity) in enumerate(series.groupby(label_column, sort=False)):
        figure.add_trace(
            go.Scatter(
                x=entity[x_column],
                y=entity["count"],
                mode="lines+markers",
                name=str(label),
                line={"color": PALETTE[position % len(PALETTE)], "width": 2.4},
                marker={"size": 7, "line": {"color": "#06121d", "width": 1}},
                customdata=entity[["weighted_count"]],
                hovertemplate=(
                    f"<b>{label}</b><br>Periodo: %{{x}}"
                    "<br>Conteo: %{y:,.0f}"
                    "<br>Carga ponderada: %{customdata[0]:,.0f}<extra></extra>"
                ),
            )
        )
    layout = _base_layout(430, left_margin=72)
    layout.update(
        showlegend=True,
        legend={
            "orientation": "h",
            "yanchor": "bottom",
            "y": 1.04,
            "xanchor": "left",
            "x": 0,
            "font": {"color": MUTED, "size": 9},
        },
        xaxis={
            "title": {"text": "PERIODO", "font": {"color": MUTED, "size": 10}},
            "tickfont": {"color": MUTED, "size": 10},
            "tickangle": -35 if cadence == QUARTERLY_VIEW else 0,
            "showgrid": False,
            "categoryorder": "trace" if cadence == QUARTERLY_VIEW else None,
        },
        yaxis={
            "title": {"text": "CONTEO", "font": {"color": MUTED, "size": 10}},
            "tickfont": {"color": MUTED, "size": 10},
            "gridcolor": GRID,
            "zeroline": False,
            "rangemode": "tozero",
        },
    )
    figure.update_layout(**layout)
    return figure


def build_trend_diverging_chart(
    trends: pd.DataFrame,
    change_column: str,
    period_label: str,
    selected_entity: str | None = None,
    level: str = TYPE_LEVEL,
    metric_view: str = PERIOD_VARIATION_VIEW,
) -> go.Figure:
    """Compara variación acumulada o CAGR sobre la cohorte ya preparada."""
    period_columns = {
        "cumulative_change": ("count_2023", "count_2025", 2),
        "change_23_24": ("count_2023", "count_2024", 1),
        "change_24_25": ("count_2024", "count_2025", 1),
    }
    start_column, end_column, year_span = period_columns[change_column]
    plotted = trends.dropna(subset=[change_column]).copy()
    if metric_view == ANNUAL_AVERAGE_VIEW:
        valid = plotted[start_column].gt(0) & plotted[end_column].gt(0)
        plotted = plotted.loc[valid].copy()
        plotted["display_change"] = (
            (plotted[end_column].div(plotted[start_column])).pow(1 / year_span) - 1
        ).mul(100)
        value_suffix = "% anual"
        hover_metric = "Variación media anual"
        axis_title = f"VARIACIÓN MEDIA ANUAL {period_label} (%)"
    else:
        plotted["display_change"] = plotted[change_column]
        value_suffix = "%"
        hover_metric = "Variación descriptiva"
        axis_title = f"VARIACIÓN {period_label} (%)"

    if level == GROUP_LEVEL:
        entity_column = "group"
        plotted["display_label"] = plotted["group"].map(
            lambda value: _wrapped_label(value, 34)
        )
        left_margin = 260
    else:
        entity_column = "crime_id"
        plotted["display_label"] = plotted.apply(
            lambda row: _wrapped_label(f"{row['crime_id']} · {row['crime_type']}", 42),
            axis=1,
        )
        left_margin = 330

    plotted = plotted.sort_values("display_change")
    colors = ["#42debc" if value >= 0 else RED for value in plotted["display_change"]]
    border_colors = [
        "#f4fdff"
        if selected_entity and str(entity) == str(selected_entity)
        else AMBER
        if (level == TYPE_LEVEL and str(entity) == "10")
        or (level == GROUP_LEVEL and str(entity) == "Drogas")
        else "rgba(110,222,249,.32)"
        for entity in plotted[entity_column]
    ]
    border_widths = [
        2.5
        if (selected_entity and str(entity) == str(selected_entity))
        or (level == TYPE_LEVEL and str(entity) == "10")
        or (level == GROUP_LEVEL and str(entity) == "Drogas")
        else 0.7
        for entity in plotted[entity_column]
    ]
    plotted["entity_name"] = (
        plotted["group"] if level == GROUP_LEVEL else plotted["crime_type"]
    )
    plotted["entity_id"] = (
        plotted["group"] if level == GROUP_LEVEL else plotted["crime_id"]
    )
    id_line = "" if level == GROUP_LEVEL else "<br>ID: %{customdata[0]}"
    figure = go.Figure(
        go.Bar(
            x=plotted["display_change"],
            y=plotted["display_label"],
            orientation="h",
            marker={
                "color": colors,
                "line": {"color": border_colors, "width": border_widths},
            },
            text=[f"{value:+.1f}{value_suffix}" for value in plotted["display_change"]],
            textposition="outside",
            textfont={"color": TEXT, "size": 10},
            cliponaxis=False,
            customdata=plotted[
                [
                    "entity_id",
                    "entity_name",
                    start_column,
                    end_column,
                    "display_change",
                    "classification",
                ]
            ],
            hovertemplate=(
                f"<b>%{{customdata[1]}}</b>{id_line}"
                f"<br>Periodo: {period_label}"
                "<br>Valor inicial: %{customdata[2]:,.0f}"
                "<br>Valor final: %{customdata[3]:,.0f}"
                f"<br>{hover_metric}: %{{customdata[4]:+.2f}}%"
                "<br>Clasificación: %{customdata[5]}<extra></extra>"
            ),
        )
    )
    figure.add_vline(x=0, line_color="rgba(215,238,247,.55)", line_width=1)
    layout = _base_layout(max(430, 30 * len(plotted) + 105), left_margin=left_margin)
    layout.update(
        showlegend=False,
        bargap=.32,
        xaxis={
            "title": {
                "text": axis_title,
                "font": {"color": MUTED, "size": 10},
            },
            "ticksuffix": "%",
            "tickfont": {"color": MUTED, "size": 10},
            "gridcolor": GRID,
            "zeroline": False,
        },
        yaxis={"tickfont": {"color": TEXT, "size": 10}, "automargin": True},
    )
    figure.update_layout(**layout)
    return figure


def build_night_profile_chart(
    summary: pd.DataFrame,
    selected_crime_id: str | None = None,
) -> go.Figure:
    """Relaciona proporción nocturna con volumen absoluto por tipología."""
    plotted = summary.loc[summary["count"].gt(0) & summary["night_share"].notna()].copy()
    plotted = plotted.sort_values("night_share", ascending=True)
    plotted["display_label"] = plotted.apply(
        lambda row: _wrapped_label(f"{row['crime_id']} · {row['crime_type']}", 42), axis=1
    )
    max_count = max(float(plotted["count"].max()), 1.0)
    sizeref = 2 * max_count / (36**2)
    figure = go.Figure(
        go.Scatter(
            x=plotted["night_share"],
            y=plotted["display_label"],
            mode="markers",
            opacity=.48 if selected_crime_id else .9,
            marker={
                "size": plotted["count"],
                "sizemode": "area",
                "sizeref": sizeref,
                "sizemin": 7,
                "color": plotted["night_count"],
                "colorscale": [[0, "#1b6686"], [.55, "#f2b84b"], [1, "#ff6170"]],
                "line": {"color": "rgba(235, 250, 255, .55)", "width": 1},
                "showscale": False,
            },
            customdata=plotted[
                ["crime_id", "crime_type", "count", "night_count", "night_share"]
            ],
            hovertemplate=(
                "<b>%{customdata[1]}</b><br>ID: %{customdata[0]}"
                "<br>Conteo total: %{customdata[2]:,.0f}"
                "<br>Conteo noche: %{customdata[3]:,.0f}"
                "<br>Componente nocturno: %{customdata[4]:.2f}%<extra></extra>"
            ),
        )
    )
    selected = plotted.loc[plotted["crime_id"].eq(str(selected_crime_id))]
    if not selected.empty:
        figure.add_trace(
            go.Scatter(
                x=selected["night_share"],
                y=selected["display_label"],
                mode="markers",
                marker={
                    "size": selected["count"],
                    "sizemode": "area",
                    "sizeref": sizeref * .68,
                    "sizemin": 12,
                    "color": RED,
                    "line": {"color": "#f4fdff", "width": 2.5},
                },
                customdata=selected[
                    ["crime_id", "crime_type", "count", "night_count", "night_share"]
                ],
                hovertemplate=(
                    "<b>%{customdata[1]}</b><br>ID: %{customdata[0]}"
                    "<br>Conteo total: %{customdata[2]:,.0f}"
                    "<br>Conteo noche: %{customdata[3]:,.0f}"
                    "<br>Componente nocturno: %{customdata[4]:.2f}%<extra></extra>"
                ),
                showlegend=False,
            )
        )
    median = float(plotted["night_share"].median()) if not plotted.empty else 0.0
    figure.add_vline(
        x=median,
        line_color="rgba(174, 193, 207, .55)",
        line_dash="dot",
        annotation_text=f"Mediana {median:.1f}%",
        annotation_font={"color": MUTED, "size": 9},
    )
    layout = _base_layout(max(520, 30 * len(plotted) + 110), left_margin=330)
    layout.update(
        showlegend=False,
        xaxis={
            "title": {"text": "% NOCTURNO · TAMAÑO = VOLUMEN", "font": {"color": MUTED, "size": 10}},
            "ticksuffix": "%",
            "range": [0, min(100, max(10, float(plotted["night_share"].max()) * 1.08))],
            "tickfont": {"color": MUTED, "size": 10},
            "gridcolor": GRID,
            "zeroline": False,
        },
        yaxis={"tickfont": {"color": TEXT, "size": 10}, "automargin": True},
    )
    figure.update_layout(**layout)
    return figure


def build_frequency_gravity_matrix(
    summary: pd.DataFrame,
    selected_crime_id: str | None = None,
) -> go.Figure:
    """Matriz de frecuencia y peso con cuadrantes definidos por medianas."""
    plotted = summary.loc[summary["count"].gt(0)].copy()
    median_count = float(plotted["count"].median())
    median_weight = float(plotted["weight"].median())
    max_weighted = max(float(plotted["weighted_count"].max()), 1.0)
    sizeref = 2 * max_weighted / (44**2)
    figure = go.Figure(
        go.Scatter(
            x=plotted["count"],
            y=plotted["weight"],
            mode="markers+text",
            opacity=.52 if selected_crime_id else .9,
            text=plotted["crime_id"],
            textposition="top center",
            textfont={"color": "#eaf9ff", "size": 10},
            marker={
                "size": plotted["weighted_count"],
                "sizemode": "area",
                "sizeref": sizeref,
                "sizemin": 9,
                "color": plotted["night_share"].fillna(0),
                "colorscale": [[0, "#167ca5"], [.62, CYAN], [1, RED]],
                "line": {"color": "rgba(238, 252, 255, .64)", "width": 1},
                "showscale": False,
            },
            customdata=plotted[
                ["crime_id", "crime_type", "group", "count", "weight", "weighted_count"]
            ],
            hovertemplate=(
                "<b>%{customdata[1]}</b><br>ID: %{customdata[0]}"
                "<br>Grupo: %{customdata[2]}"
                "<br>Frecuencia: %{customdata[3]:,.0f}"
                "<br>Peso gravedad: %{customdata[4]:.1f}"
                "<br>Conteo ponderado: %{customdata[5]:,.0f}<extra></extra>"
            ),
        )
    )
    selected = plotted.loc[plotted["crime_id"].eq(str(selected_crime_id))]
    if not selected.empty:
        figure.add_trace(
            go.Scatter(
                x=selected["count"],
                y=selected["weight"],
                mode="markers+text",
                text=selected["crime_id"],
                textposition="top center",
                textfont={"color": "#f4fdff", "size": 11},
                marker={
                    "size": selected["weighted_count"],
                    "sizemode": "area",
                    "sizeref": sizeref * .68,
                    "sizemin": 13,
                    "color": RED,
                    "line": {"color": "#f4fdff", "width": 2.5},
                },
                customdata=selected[
                    ["crime_id", "crime_type", "group", "count", "weight", "weighted_count"]
                ],
                hovertemplate=(
                    "<b>%{customdata[1]}</b><br>ID: %{customdata[0]}"
                    "<br>Grupo: %{customdata[2]}"
                    "<br>Frecuencia: %{customdata[3]:,.0f}"
                    "<br>Peso gravedad: %{customdata[4]:.1f}"
                    "<br>Conteo ponderado: %{customdata[5]:,.0f}<extra></extra>"
                ),
                showlegend=False,
            )
        )
    figure.add_vline(x=median_count, line_dash="dot", line_color="rgba(174,193,207,.55)")
    figure.add_hline(y=median_weight, line_dash="dot", line_color="rgba(174,193,207,.55)")
    annotations = [
        (.98, .96, "ALTA FRECUENCIA · ALTA GRAVEDAD", "right"),
        (.02, .96, "BAJA FRECUENCIA · ALTA GRAVEDAD", "left"),
        (.98, .04, "ALTA FRECUENCIA · BAJA GRAVEDAD", "right"),
        (.02, .04, "BAJA FRECUENCIA · BAJA GRAVEDAD", "left"),
    ]
    for x, y, text, anchor in annotations:
        figure.add_annotation(
            x=x,
            y=y,
            xref="paper",
            yref="paper",
            text=text,
            showarrow=False,
            xanchor=anchor,
            font={"color": "rgba(174,193,207,.72)", "size": 8},
        )
    layout = _base_layout(540, left_margin=78)
    layout.update(
        showlegend=False,
        xaxis={
            "title": {"text": "FRECUENCIA · CONTEO", "font": {"color": MUTED, "size": 10}},
            "tickfont": {"color": MUTED, "size": 10},
            "gridcolor": GRID,
            "zeroline": False,
            "rangemode": "tozero",
        },
        yaxis={
            "title": {"text": "PESO DE GRAVEDAD", "font": {"color": MUTED, "size": 10}},
            "tickfont": {"color": MUTED, "size": 10},
            "gridcolor": GRID,
            "zeroline": False,
            "range": [0, max(10.8, float(plotted["weight"].max()) + .8)],
        },
    )
    figure.update_layout(**layout)
    return figure


def build_territorial_ranking_chart(
    summary: pd.DataFrame,
    selected_municipality: str | None = None,
    metric_column: str = "count",
) -> go.Figure:
    """Ranking territorial del tipo seleccionado; muestra hasta 15 municipios."""
    ordered = summary.sort_values(metric_column, ascending=False).copy()
    ranked = ordered.head(15).copy()
    selected_row = ordered.loc[
        ordered["municipality"].eq(str(selected_municipality))
    ]
    if (
        not selected_row.empty
        and selected_row.iloc[0]["municipality"] not in set(ranked["municipality"])
    ):
        ranked = pd.concat([ranked.head(14), selected_row.head(1)], ignore_index=True)
    ranked = ranked.sort_values(metric_column, ascending=True)
    colors = [
        RED if selected_municipality and name == selected_municipality else CYAN
        for name in ranked["municipality"]
    ]
    figure = go.Figure(
        go.Bar(
            x=ranked[metric_column],
            y=ranked["municipality"],
            orientation="h",
            marker={"color": colors, "line": {"color": "rgba(110,222,249,.45)", "width": 1}},
            text=[
                f"{value:.1f}%"
                if metric_column == "share_municipality"
                else f"{value:,.0f}"
                for value in ranked[metric_column]
            ],
            textposition="outside",
            textfont={"color": "#e4f8ff", "size": 10},
            cliponaxis=False,
            customdata=ranked[
                [
                    "municipality",
                    "count",
                    "night_count",
                    "weighted_count",
                    "share_total",
                    "municipality_total",
                    "share_municipality",
                ]
            ],
            hovertemplate=(
                "<b>%{customdata[0]}</b><br>Conteo: %{customdata[1]:,.0f}"
                "<br>Conteo noche: %{customdata[2]:,.0f}"
                "<br>Carga ponderada: %{customdata[3]:,.0f}"
                "<br>% del ámbito: %{customdata[4]:.2f}%"
                "<br>Total del municipio: %{customdata[5]:,.0f}"
                "<br>Peso dentro del municipio: %{customdata[6]:.2f}%<extra></extra>"
            ),
        )
    )
    layout = _base_layout(500, left_margin=185)
    layout.update(
        showlegend=False,
        xaxis={
            "title": {
                "text": (
                    "PESO DENTRO DEL MUNICIPIO (%)"
                    if metric_column == "share_municipality"
                    else "CONTEO ABSOLUTO"
                ),
                "font": {"color": MUTED, "size": 10},
            },
            "ticksuffix": "%" if metric_column == "share_municipality" else "",
            "tickfont": {"color": MUTED, "size": 10},
            "gridcolor": GRID,
            "zeroline": False,
        },
        yaxis={"tickfont": {"color": TEXT, "size": 10}, "automargin": True},
    )
    figure.update_layout(**layout)
    return figure


def build_drug_quarterly_chart(case: DrugCase) -> go.Figure:
    figure = go.Figure(
        go.Scatter(
            x=case.quarterly["period_label"],
            y=case.quarterly["count"],
            mode="lines+markers",
            line={"color": RED, "width": 3},
            marker={"size": 8, "color": "#ffd0d5", "line": {"color": RED, "width": 2}},
            fill="tozeroy",
            fillcolor="rgba(255, 97, 112, .08)",
            hovertemplate="<b>%{x}</b><br>Conteo: %{y:,.0f}<extra></extra>",
        )
    )
    layout = _base_layout(350, left_margin=70)
    layout.update(
        showlegend=False,
        xaxis={
            "tickangle": -35,
            "tickfont": {"color": MUTED, "size": 9},
            "showgrid": False,
        },
        yaxis={
            "title": {"text": "CONTEO", "font": {"color": MUTED, "size": 10}},
            "tickfont": {"color": MUTED, "size": 10},
            "gridcolor": GRID,
            "zeroline": False,
            "rangemode": "tozero",
        },
    )
    figure.update_layout(**layout)
    return figure


def build_drug_territorial_chart(case: DrugCase) -> go.Figure:
    """Dos paneles: volumen 2025 y crecimiento absoluto 2023-2025."""
    volume = case.top_volume.head(8).sort_values("count", ascending=True)
    growth = case.top_growth.head(8).sort_values("change_abs", ascending=True)
    figure = make_subplots(
        rows=1,
        cols=2,
        horizontal_spacing=.22,
        subplot_titles=("MAYOR VOLUMEN · 2025", "MAYOR AUMENTO ABSOLUTO · 2023–2025"),
    )
    figure.add_trace(
        go.Bar(
            x=volume["count"],
            y=volume["municipality"],
            orientation="h",
            marker={"color": RED},
            customdata=volume[["municipality", "count"]],
            hovertemplate="<b>%{customdata[0]}</b><br>Conteo 2025: %{customdata[1]:,.0f}<extra></extra>",
        ),
        row=1,
        col=1,
    )
    figure.add_trace(
        go.Bar(
            x=growth["change_abs"],
            y=growth["municipality"],
            orientation="h",
            marker={"color": AMBER},
            customdata=growth[["municipality", 2023, 2025, "change_abs", "change_pct"]],
            hovertemplate=(
                "<b>%{customdata[0]}</b><br>2023: %{customdata[1]:,.0f}"
                "<br>2025: %{customdata[2]:,.0f}"
                "<br>Aumento absoluto: %{customdata[3]:+,.0f}"
                "<br>Variación: %{customdata[4]:+.1f}%<extra></extra>"
            ),
        ),
        row=1,
        col=2,
    )
    figure.update_layout(**_base_layout(420, left_margin=155))
    figure.update_layout(showlegend=False, bargap=.34)
    figure.update_xaxes(gridcolor=GRID, zeroline=False, tickfont={"color": MUTED, "size": 9})
    figure.update_yaxes(automargin=True, tickfont={"color": TEXT, "size": 9})
    for annotation in figure.layout.annotations:
        annotation.font = {"color": MUTED, "size": 10}
    return figure
