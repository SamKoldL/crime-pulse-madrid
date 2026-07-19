"""Plotly choropleth visualization for Crime Pulse Madrid."""

from __future__ import annotations

from html import escape
from typing import Any

import pandas as pd
import plotly.graph_objects as go

from utils.map_data import (
    CRIME_INDEX,
    MapSource,
    geojson_bounds,
    map_center_from_bounds,
    mapbox_viewport_from_geojson,
)


NIGHT_RATE = "Delitos_noche_por_10000_hab"
RISK_COLORS = {
    "Muy bajo": "#145476",
    "Bajo": "#168db5",
    "Medio": "#35d2ed",
    "Alto": "#f1ab4c",
    "Muy alto": "#ff5262",
}
RISK_ORDER = ["Muy bajo", "Bajo", "Medio", "Alto", "Muy alto"]
RISK_BINS = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
RISK_CODES = {risk_band: index for index, risk_band in enumerate(RISK_ORDER)}
RISK_COLORSCALE = [
    [0.0, RISK_COLORS["Muy bajo"]],
    [0.124999, RISK_COLORS["Muy bajo"]],
    [0.125, RISK_COLORS["Bajo"]],
    [0.374999, RISK_COLORS["Bajo"]],
    [0.375, RISK_COLORS["Medio"]],
    [0.624999, RISK_COLORS["Medio"]],
    [0.625, RISK_COLORS["Alto"]],
    [0.874999, RISK_COLORS["Alto"]],
    [0.875, RISK_COLORS["Muy alto"]],
    [1.0, RISK_COLORS["Muy alto"]],
]


def _spanish_number(value: object, decimals: int = 1) -> str:
    if pd.isna(value):
        return "No disponible"
    formatted = f"{float(value):,.{decimals}f}"
    return formatted.replace(",", "X").replace(".", ",").replace("X", ".")


def classify_relative_risk(
    frame: pd.DataFrame,
    *,
    value_column: str = CRIME_INDEX,
    eligibility_column: str = "_eligible",
    output_column: str = "_risk_band",
) -> pd.DataFrame:
    """Aplica los mismos quintiles relativos y categorías del Mapa Criminal."""
    classified = frame.copy()
    eligible_mask = (
        classified[eligibility_column].fillna(False).astype(bool)
        & classified[value_column].notna()
    )
    classified[output_column] = pd.Series(pd.NA, index=classified.index, dtype="object")
    if eligible_mask.any():
        ranked = classified.loc[eligible_mask, value_column].rank(
            method="first",
            pct=True,
        )
        bands = pd.cut(
            ranked,
            bins=RISK_BINS,
            labels=RISK_ORDER,
            include_lowest=True,
        )
        classified.loc[eligible_mask, output_column] = bands.astype("object")
    classified[output_column] = pd.Categorical(
        classified[output_column],
        categories=RISK_ORDER,
        ordered=True,
    )
    return classified


def geojson_subset(
    geojson: dict[str, Any], feature_ids: pd.Series
) -> dict[str, Any]:
    """Create a lightweight collection referencing, not copying, its polygons."""
    requested = set(feature_ids.astype(str))
    return {
        "type": "FeatureCollection",
        "features": [
            feature
            for feature in geojson["features"]
            if str(feature["properties"]["feature_id"]) in requested
        ],
    }


def add_risk_legend_traces(figure: go.Figure, risk_bands: pd.Series) -> None:
    """Keep the categorical legend without repeating GeoJSON per category."""
    present = set(risk_bands.dropna().astype(str))
    for risk_band in RISK_ORDER:
        if risk_band not in present:
            continue
        figure.add_trace(
            go.Scattermapbox(
                lat=[None],
                lon=[None],
                mode="markers",
                marker={"size": 9, "color": RISK_COLORS[risk_band]},
                name=risk_band,
                hoverinfo="skip",
                showlegend=True,
            )
        )


def _eligible_hover(frame: pd.DataFrame) -> list[str]:
    night_values = (
        frame[NIGHT_RATE] if NIGHT_RATE in frame.columns else pd.Series(pd.NA, index=frame.index)
    )
    return [
        (
            f"<b>{escape(str(row['Municipio']))}</b>"
            f"<br>Índice ponderado / 10.000 hab.: {_spanish_number(row[CRIME_INDEX], 1)}"
            f"<br>Ranking anual: #{_spanish_number(row['Ranking_criminal_anual'], 0)}"
            f"<br>Delitos totales: {_spanish_number(row['Delitos_totales'], 0)}"
            f"<br>Población: {_spanish_number(row['Población'], 0)}"
            f"<br>Delitos nocturnos / 10.000 hab.: {_spanish_number(night, 1)}"
        )
        for (_, row), night in zip(frame.iterrows(), night_values)
    ]


def build_territorial_map(
    source: MapSource,
    year: int,
    classified_frame: pd.DataFrame | None = None,
    selected_municipality: str | None = None,
) -> go.Figure:
    """Render all 37 polygons, separating eligible and excluded municipalities."""
    if not source.available or source.geojson is None:
        raise ValueError("No hay una fuente geográfica poligonal disponible.")

    frame = (
        classified_frame.copy()
        if classified_frame is not None
        else classify_relative_risk(source.frame)
    )
    eligible = frame.loc[frame["_eligible"]].copy()
    excluded = frame.loc[~frame["_eligible"]].copy()
    figure = go.Figure()

    if not excluded.empty:
        figure.add_trace(
            go.Choroplethmapbox(
                geojson=geojson_subset(source.geojson, excluded["_feature_id"]),
                featureidkey="properties.feature_id",
                locations=excluded["_feature_id"],
                z=[0] * len(excluded),
                colorscale=[[0, "#18232d"], [1, "#18232d"]],
                showscale=False,
                marker={"line": {"color": "rgba(68, 194, 226, .52)", "width": 0.7}},
                text=[
                    f"<b>{escape(str(name))}</b><br>Fuera del universo de análisis en {year}"
                    for name in excluded["Municipio"]
                ],
                customdata=excluded["Municipio"].astype(str),
                hovertemplate="%{text}<extra></extra>",
                name="No elegible",
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
                marker={"line": {"color": "rgba(70, 218, 248, .70)", "width": 0.75}},
                text=_eligible_hover(eligible),
                customdata=eligible["Municipio"].astype(str),
                hovertemplate="%{text}<extra></extra>",
                name="Municipios elegibles",
                showlegend=False,
                selectedpoints=[],
                selected={"marker": {"opacity": 1}},
                unselected={"marker": {"opacity": 1}},
            )
        )
        add_risk_legend_traces(figure, eligible["_risk_band"])

        selected = eligible.loc[
            eligible["Municipio"].eq(str(selected_municipality))
        ]
        if not selected.empty:
            selected_color = RISK_COLORS[str(selected.iloc[0]["_risk_band"])]
            figure.add_trace(
                go.Choroplethmapbox(
                    geojson=geojson_subset(
                        source.geojson,
                        selected["_feature_id"],
                    ),
                    featureidkey="properties.feature_id",
                    locations=selected["_feature_id"],
                    z=[1] * len(selected),
                    colorscale=[[0, selected_color], [1, selected_color]],
                    showscale=False,
                    marker={"line": {"color": "#f4fdff", "width": 3}},
                    text=_eligible_hover(selected),
                    customdata=selected["Municipio"].astype(str),
                    hovertemplate="%{text}<extra></extra>",
                    name="Municipio seleccionado",
                    showlegend=False,
                    selectedpoints=[],
                    selected={"marker": {"opacity": 1}},
                    unselected={"marker": {"opacity": 1}},
                )
            )

    figure.update_layout(
        height=620,
        margin={"l": 0, "r": 0, "t": 0, "b": 0},
        paper_bgcolor="rgba(0,0,0,0)",
        uirevision=f"territorial-map-{year}",
        showlegend=True,
        legend={
            "orientation": "h",
            "yanchor": "bottom",
            "y": 0.01,
            "xanchor": "center",
            "x": 0.5,
            "title": {"text": "Nivel relativo anual"},
            "font": {"color": "#93adbd", "size": 10},
            "bgcolor": "rgba(2, 10, 16, .72)",
        },
        mapbox={
            "style": "carto-darkmatter",
            **(
                source.territorial_viewport
                or {
                    "center": source.center
                    or map_center_from_bounds(geojson_bounds(source.geojson)),
                    "zoom": 7.15,
                }
            ),
        },
        hoverlabel={
            "bgcolor": "#071521",
            "bordercolor": "#238bb5",
            "font": {"color": "#e9f8ff", "family": "Inter, Arial, sans-serif"},
        },
    )
    return figure
