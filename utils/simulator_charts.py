"""Visualizaciones Plotly del simulador de redistribución policial."""

from __future__ import annotations

from html import escape
from typing import Any, Iterator, Mapping, Sequence

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


def _iter_positions(coordinates: Any) -> Iterator[tuple[float, float]]:
    if (
        isinstance(coordinates, list)
        and len(coordinates) >= 2
        and isinstance(coordinates[0], (int, float))
        and isinstance(coordinates[1], (int, float))
    ):
        yield float(coordinates[0]), float(coordinates[1])
    elif isinstance(coordinates, list):
        for child in coordinates:
            yield from _iter_positions(child)


def _feature_centres(source: MapSource) -> dict[str, tuple[float, float]]:
    """Centros de bounding box suficientes para dibujar rutas sobre polígonos."""
    if source.geojson is None:
        return {}
    centres: dict[str, tuple[float, float]] = {}
    for feature in source.geojson.get("features", []):
        feature_id = str(feature.get("properties", {}).get("feature_id", ""))
        positions = list(_iter_positions(feature.get("geometry", {}).get("coordinates", [])))
        if not feature_id or not positions:
            continue
        longitudes, latitudes = zip(*positions)
        centres[feature_id] = (
            (min(longitudes) + max(longitudes)) / 2,
            (min(latitudes) + max(latitudes)) / 2,
        )
    return centres


def _municipality_centres(source: MapSource) -> dict[str, tuple[float, float]]:
    centres_by_feature = _feature_centres(source)
    return {
        str(row["municipality"]): centres_by_feature[str(row["_feature_id"])]
        for _, row in source.frame.iterrows()
        if str(row["_feature_id"]) in centres_by_feature
    }


def _map_layout(source: MapSource, *, height: int = 610) -> dict[str, Any]:
    viewport = source.territorial_viewport or source.viewport
    return {
        "height": height,
        "margin": {"l": 0, "r": 0, "t": 0, "b": 0},
        "paper_bgcolor": "rgba(0,0,0,0)",
        "mapbox": {
            "style": "carto-darkmatter",
            **(viewport or {"center": source.center or {"lon": -3.65, "lat": 40.42}, "zoom": 7.15}),
        },
        "hoverlabel": {
            "bgcolor": "#071521",
            "bordercolor": "#238bb5",
            "font": {"color": TEXT, "family": "Inter, Arial, sans-serif"},
        },
        "showlegend": False,
        "uirevision": "simulator-territorial-view",
    }


def _hover_text(frame: pd.DataFrame) -> list[str]:
    return [
        (
            f"<b>{escape(str(row['municipality']))}</b>"
            f"<br>Forecast: {row['predicted_crime']:,.1f}"
            f"<br>Agentes: {int(row['agents_before']):,} → {int(row['agents_after']):,}"
            f"<br>Presión: {row['pressure_before']:,.2f} → {row['pressure_after']:,.2f}"
            f"<br>Variación agentes: {int(row['agent_change']):+,}"
            f"<br>Mejora cobertura: {row['coverage_improvement_pct']:+.1f}%"
        )
        for _, row in frame.iterrows()
    ]


def _route_traces(
    source: MapSource,
    movements: Sequence[Mapping[str, object]],
) -> tuple[list[go.Scattermapbox], list[tuple[float, float, str]]]:
    centres = _municipality_centres(source)
    maximum = max((int(item["agents"]) for item in movements), default=1)
    traces: list[go.Scattermapbox] = []
    midpoints: list[tuple[float, float, str]] = []
    for movement in movements:
        origin = str(movement["origin"])
        destination = str(movement["destination"])
        agents = int(movement["agents"])
        if origin not in centres or destination not in centres:
            continue
        origin_lon, origin_lat = centres[origin]
        destination_lon, destination_lat = centres[destination]
        label = f"{escape(origin)} → {escape(destination)} · {agents} agentes"
        traces.append(
            go.Scattermapbox(
                lon=[origin_lon, destination_lon],
                lat=[origin_lat, destination_lat],
                mode="lines",
                line={"color": CYAN, "width": 1.4 + 4.6 * agents / maximum},
                opacity=0.76,
                text=[label, label],
                hovertemplate="%{text}<extra></extra>",
                showlegend=False,
            )
        )
        midpoints.append(((origin_lon + destination_lon) / 2, (origin_lat + destination_lat) / 2, label))
    return traces, midpoints


def build_simulator_map(
    source: MapSource,
    view: str,
    movements: Sequence[Mapping[str, object]] = (),
) -> go.Figure:
    """Mapa Antes/Después/Variación/Movimientos con una geometría compartida."""
    if not source.available or source.geojson is None:
        raise ValueError("No existe una fuente geográfica poligonal disponible.")
    if view not in {"ANTES", "DESPUÉS", "VARIACIÓN", "MOVIMIENTOS"}:
        raise ValueError(f"Vista de mapa no reconocida: {view}.")
    frame = source.frame.copy()
    texts = _hover_text(frame)
    figure = go.Figure()

    if view in {"ANTES", "DESPUÉS"}:
        pressure_column = "pressure_before" if view == "ANTES" else "pressure_after"
        pressure_min = float(min(frame["pressure_before"].min(), frame["pressure_after"].min()))
        pressure_max = float(max(frame["pressure_before"].max(), frame["pressure_after"].max()))
        figure.add_trace(
            go.Choroplethmapbox(
                geojson=source.geojson,
                featureidkey="properties.feature_id",
                locations=frame["_feature_id"],
                z=frame[pressure_column],
                zmin=pressure_min,
                zmax=pressure_max,
                colorscale=[[0.0, "#183b55"], [0.34, "#2b8eb3"], [0.66, AMBER], [1.0, RED]],
                marker={"line": {"color": "rgba(81, 211, 240, .68)", "width": 0.75}},
                text=texts,
                hovertemplate="%{text}<extra></extra>",
                colorbar={
                    "title": {"text": "PRESIÓN<br>POR AGENTE", "font": {"color": MUTED, "size": 9}},
                    "tickfont": {"color": MUTED, "size": 9},
                    "thickness": 10,
                    "len": 0.46,
                    "x": 0.985,
                    "outlinewidth": 0,
                    "bgcolor": "rgba(2,10,16,.68)",
                },
            )
        )
    elif view == "VARIACIÓN":
        maximum = max(abs(float(frame["coverage_improvement_pct"].min())), abs(float(frame["coverage_improvement_pct"].max())), 0.01)
        figure.add_trace(
            go.Choroplethmapbox(
                geojson=source.geojson,
                featureidkey="properties.feature_id",
                locations=frame["_feature_id"],
                z=frame["coverage_improvement_pct"],
                zmin=-maximum,
                zmax=maximum,
                zmid=0,
                colorscale=[[0.0, RED], [0.5, "#233340"], [1.0, CYAN]],
                marker={"line": {"color": "rgba(81, 211, 240, .62)", "width": 0.75}},
                text=texts,
                hovertemplate="%{text}<extra></extra>",
                colorbar={
                    "title": {"text": "MEJORA DE<br>COBERTURA %", "font": {"color": MUTED, "size": 9}},
                    "ticksuffix": "%",
                    "tickfont": {"color": MUTED, "size": 9},
                    "thickness": 10,
                    "len": 0.46,
                    "x": 0.985,
                    "outlinewidth": 0,
                    "bgcolor": "rgba(2,10,16,.68)",
                },
            )
        )
    else:
        figure.add_trace(
            go.Choroplethmapbox(
                geojson=source.geojson,
                featureidkey="properties.feature_id",
                locations=frame["_feature_id"],
                z=np.zeros(len(frame)),
                colorscale=[[0, "#102635"], [1, "#102635"]],
                showscale=False,
                marker={"line": {"color": "rgba(81, 211, 240, .52)", "width": 0.7}},
                text=texts,
                hovertemplate="%{text}<extra></extra>",
            )
        )
        routes, midpoints = _route_traces(source, movements)
        for trace in routes:
            figure.add_trace(trace)
        if midpoints:
            figure.add_trace(
                go.Scattermapbox(
                    lon=[item[0] for item in midpoints],
                    lat=[item[1] for item in midpoints],
                    text=[item[2] for item in midpoints],
                    mode="markers",
                    marker={"size": 7, "color": AMBER},
                    hovertemplate="%{text}<extra></extra>",
                    showlegend=False,
                )
            )

    figure.update_layout(**_map_layout(source))
    return figure


def build_pressure_map(source: MapSource, view: str) -> go.Figure:
    """Compatibilidad con llamadas anteriores del simulador."""
    return build_simulator_map(source, view)


def build_movement_animation(
    source: MapSource,
    movements: Sequence[Mapping[str, object]],
) -> go.Figure:
    """Reproduce rutas con frames Plotly; no comunica eventos al servidor."""
    figure = build_simulator_map(source, "MOVIMIENTOS", movements)
    _, midpoints = _route_traces(source, movements)
    if not midpoints:
        return figure
    animated_trace_index = len(figure.data)
    first_lon, first_lat, first_label = midpoints[0]
    figure.add_trace(
        go.Scattermapbox(
            lon=[first_lon],
            lat=[first_lat],
            text=[first_label],
            mode="markers",
            marker={"size": 17, "color": RED, "opacity": 0.92},
            hovertemplate="%{text}<extra></extra>",
            showlegend=False,
        )
    )
    figure.frames = tuple(
        go.Frame(
            name=str(index),
            data=[
                go.Scattermapbox(
                    lon=[longitude],
                    lat=[latitude],
                    text=[label],
                    mode="markers",
                    marker={"size": 17, "color": RED, "opacity": 0.92},
                    hovertemplate="%{text}<extra></extra>",
                )
            ],
            traces=[animated_trace_index],
        )
        for index, (longitude, latitude, label) in enumerate(midpoints)
    )
    figure.update_layout(
        updatemenus=[
            {
                "type": "buttons",
                "direction": "left",
                "x": 0.02,
                "y": 0.04,
                "showactive": False,
                "bgcolor": "rgba(3,17,28,.88)",
                "bordercolor": CYAN,
                "font": {"color": TEXT},
                "buttons": [
                    {
                        "label": "▶ REPRODUCIR MOVIMIENTOS",
                        "method": "animate",
                        "args": [None, {"frame": {"duration": 420, "redraw": True}, "transition": {"duration": 180}, "fromcurrent": True}],
                    }
                ],
            }
        ]
    )
    return figure


def build_coverage_change_chart(frame: pd.DataFrame, count_each: int = 7) -> go.Figure:
    """Extremos de mejora y deterioro de cobertura relativa."""
    changed = frame.loc[frame["agent_change"].ne(0)].copy()
    figure = go.Figure()
    if changed.empty:
        return figure
    scope = (
        pd.concat([
            changed.nlargest(count_each, "coverage_improvement_pct"),
            changed.nsmallest(count_each, "coverage_improvement_pct"),
        ])
        .drop_duplicates("municipality")
        .sort_values("coverage_improvement_pct")
    )
    figure.add_trace(
        go.Bar(
            x=scope["coverage_improvement_pct"],
            y=scope["municipality"],
            orientation="h",
            marker={"color": np.where(scope["coverage_improvement_pct"].ge(0), CYAN, RED)},
            customdata=scope[["agents_before", "agents_after", "pressure_before", "pressure_after", "agent_change"]],
            hovertemplate=(
                "<b>%{y}</b><br>Mejora de cobertura: %{x:+.1f}%"
                "<br>Agentes: %{customdata[0]:,.0f} → %{customdata[1]:,.0f}"
                "<br>Variación agentes: %{customdata[4]:+,.0f}"
                "<br>Presión: %{customdata[2]:,.2f} → %{customdata[3]:,.2f}<extra></extra>"
            ),
        )
    )
    layout = _base_layout(480, left_margin=190)
    layout.update(
        showlegend=False,
        xaxis={
            "title": {"text": "MEJORA DE COBERTURA RELATIVA", "font": {"color": MUTED, "size": 10}},
            "ticksuffix": "%",
            "gridcolor": GRID,
            "zeroline": True,
            "zerolinecolor": "rgba(225,245,255,.45)",
        },
        yaxis={"automargin": True, "tickfont": {"color": TEXT, "size": 10}},
    )
    figure.update_layout(**layout)
    return figure


def build_scenario_comparison_chart(comparison: pd.DataFrame) -> go.Figure:
    """Compara la brecha de presión de actual, simulación y benchmark."""
    figure = go.Figure(
        go.Bar(
            x=comparison["gap"],
            y=comparison["scenario"],
            orientation="h",
            marker={"color": [NEUTRAL, CYAN, AMBER]},
            text=comparison["gap"].map(lambda value: f"{value:.2f}"),
            textposition="outside",
            customdata=comparison[["redistributed", "affected", "above_mean", "max_pressure"]],
            hovertemplate=(
                "<b>%{y}</b><br>Brecha: %{x:.2f}<br>Agentes redistribuidos: %{customdata[0]:,.0f}"
                "<br>Municipios afectados: %{customdata[1]:,.0f}<br>Sobre referencia: %{customdata[2]:,.0f}"
                "<br>Mayor presión: %{customdata[3]:.2f}<extra></extra>"
            ),
        )
    )
    layout = _base_layout(330, left_margin=175)
    layout.update(
        showlegend=False,
        xaxis={"title": {"text": "BRECHA MÁXIMA DE PRESIÓN", "font": {"color": MUTED, "size": 10}}, "gridcolor": GRID, "rangemode": "tozero"},
        yaxis={"autorange": "reversed", "tickfont": {"color": TEXT, "size": 10}},
    )
    figure.update_layout(**layout)
    return figure
