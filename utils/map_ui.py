"""Presentación compacta de la visión regional y la ficha territorial."""

from __future__ import annotations

from html import escape

import pandas as pd
import streamlit as st

from utils.map_charts import RISK_COLORS
from utils.map_data import CRIME_INDEX
from utils.map_metrics import MapRegionalSnapshot


ALL_MAP_MUNICIPALITIES = "TODOS LOS MUNICIPIOS"
NIGHT_RATE = "Delitos_noche_por_10000_hab"
CONTEXT_INDICES = (
    ("Ocio", "Indice_Ocio", "Indice_Ocio_Nivel"),
    ("Movilidad", "Indice_Movilidad", "Indice_Movilidad_Nivel"),
    ("Urbano", "Indice_Urbano", "Indice_Urbano_Nivel"),
    ("Servicios", "Indice_Servicios", "Indice_Servicios_Nivel"),
    (
        "Presión socioeconómica",
        "Indice_Socioeconomico",
        "Indice_Socioeconomico_Nivel",
    ),
)


def _format_number(value: object, decimals: int = 1) -> str:
    if pd.isna(value):
        return "No disponible"
    formatted = f"{float(value):,.{decimals}f}"
    return formatted.replace(",", "X").replace(".", ",").replace("X", ".")


def _metric_card(label: str, value: str, detail: str = "") -> str:
    detail_html = (
        f'<span class="map-panel-metric-detail">{escape(detail)}</span>'
        if detail
        else ""
    )
    return (
        '<article class="map-panel-metric">'
        f'<span>{escape(label)}</span><strong>{escape(value)}</strong>{detail_html}'
        "</article>"
    )


def _risk_color(level: object) -> str:
    normalized = str(level).strip().casefold()
    for risk, color in RISK_COLORS.items():
        if risk.casefold() == normalized:
            return color
    return "#35d2ed"


def _feminine_pressure_level(level: object) -> str:
    """Adapt only the socioeconomic-pressure label without changing its value."""
    normalized = str(level).strip().casefold()
    feminine_levels = {
        "muy bajo": "Muy baja",
        "bajo": "Baja",
        "medio": "Media",
        "alto": "Alta",
        "muy alto": "Muy alta",
    }
    return feminine_levels.get(normalized, str(level))


def _render_regional(snapshot: MapRegionalSnapshot) -> str:
    metrics = "".join(
        (
            _metric_card(
                "Cobertura territorial",
                f"{snapshot.eligible_count} municipios",
                "Universo elegible del periodo",
            ),
            _metric_card(
                "Índice ponderado regional",
                _format_number(snapshot.regional_index),
                "/ 10.000 hab.",
            ),
            _metric_card(
                "Mediana municipal",
                _format_number(snapshot.municipal_median),
                "Índice ponderado / 10.000 hab.",
            ),
            _metric_card(
                "Mayor índice territorial",
                snapshot.top_municipality or "No disponible",
                (
                    f"{_format_number(snapshot.top_index)} / 10.000 hab."
                    if snapshot.top_index is not None
                    else ""
                ),
            ),
        )
    )
    distribution = "".join(
        '<div class="map-risk-row">'
        f'<i style="--risk-color:{RISK_COLORS[risk]}"></i>'
        f'<span>{escape(risk.upper())}</span><b>{count} municipios</b>'
        "</div>"
        for risk, count in snapshot.risk_counts
    )
    return (
        '<div class="map-panel-heading"><span>VISIÓN REGIONAL</span>'
        '<h2>LECTURA TERRITORIAL</h2>'
        '<p>Contexto agregado del universo municipal válido para el periodo.</p></div>'
        f'<div class="map-panel-metric-grid">{metrics}</div>'
        '<div class="map-panel-subheading">DISTRIBUCIÓN TERRITORIAL</div>'
        f'<div class="map-risk-distribution">{distribution}</div>'
        '<p class="map-panel-hint">Selecciona un municipio en el mapa o utiliza el filtro superior para consultar su ficha territorial.</p>'
    )


def _render_municipality(
    snapshot: MapRegionalSnapshot,
    municipality: str,
) -> str:
    selected = snapshot.eligible.loc[
        snapshot.eligible["Municipio"].eq(municipality)
    ]
    if selected.empty:
        return (
            '<div class="map-panel-heading"><span>FICHA TERRITORIAL</span>'
            f'<h2>{escape(municipality.upper())}</h2>'
            '<p>Fuera del universo de análisis para el periodo seleccionado.</p></div>'
        )

    row = selected.iloc[0]
    risk_band = str(row["_risk_band"])
    median = snapshot.municipal_median
    index_value = float(row[CRIME_INDEX])
    versus_median = (
        (index_value - median) / median * 100
        if median is not None and median != 0
        else None
    )
    versus_text = (
        f"{versus_median:+.1f}%".replace(".", ",")
        if versus_median is not None
        else "No disponible"
    )
    metrics = "".join(
        (
            _metric_card(
                "Ranking territorial",
                f"#{int(row['Ranking_criminal_anual'])} de {snapshot.eligible_count}",
            ),
            _metric_card(
                "Índice ponderado / 10.000 hab.",
                _format_number(index_value),
            ),
            _metric_card("Nivel relativo", risk_band.upper()),
            _metric_card(
                "Delitos registrados",
                _format_number(row["Delitos_totales"], 0),
            ),
            _metric_card(
                "Nocturnos / 10.000 hab.",
                _format_number(row[NIGHT_RATE]) if NIGHT_RATE in row.index else "No disponible",
            ),
            _metric_card("Vs. mediana regional", versus_text),
        )
    )

    profiles: list[str] = []
    for label, value_column, level_column in CONTEXT_INDICES:
        value = row[value_column] if value_column in row.index else pd.NA
        level = row[level_column] if level_column in row.index else "No disponible"
        numeric_value = float(value) if pd.notna(value) else 0.0
        width = min(100.0, max(0.0, numeric_value))
        color = _risk_color(level)
        display_level = (
            _feminine_pressure_level(level)
            if value_column == "Indice_Socioeconomico"
            else str(level)
        )
        profiles.append(
            '<div class="map-profile-row" '
            f'style="--profile-width:{width:.2f}%;--profile-color:{color}">'
            f'<div><span>{escape(label)}</span><b>{escape(display_level.upper())}</b></div>'
            '<div class="map-profile-track"><i></i></div>'
            f'<strong>{_format_number(value)}</strong></div>'
        )

    return (
        '<div class="map-panel-heading"><span>FICHA TERRITORIAL</span>'
        f'<h2>{escape(municipality.upper())}</h2>'
        '<p>Posición criminal y contexto territorial del periodo seleccionado.</p></div>'
        '<div class="map-panel-subheading">POSICIÓN CRIMINAL</div>'
        f'<div class="map-panel-metric-grid">{metrics}</div>'
        '<div class="map-panel-subheading">PERFIL TERRITORIAL</div>'
        f'<div class="map-profile-list">{"".join(profiles)}</div>'
        '<p class="map-profile-note">Índices contextuales normalizados 0–100 por posición relativa entre municipios. No implican causalidad con la criminalidad.</p>'
    )


def render_map_context_panel(
    snapshot: MapRegionalSnapshot,
    selected_municipality: str,
) -> None:
    """Render regional context or the selected municipality in one stable panel."""
    html = (
        _render_regional(snapshot)
        if selected_municipality == ALL_MAP_MUNICIPALITIES
        else _render_municipality(snapshot, selected_municipality)
    )
    with st.container(border=True, key="map_detail"):
        st.markdown(html, unsafe_allow_html=True)
