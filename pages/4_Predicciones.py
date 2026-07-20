"""Forecast criminal municipal y por tipología para Q1–Q3 2026."""

from __future__ import annotations

from html import escape
from pathlib import Path

import pandas as pd
import streamlit as st

from utils.navigation import render_top_navigation
from utils.predictions_charts import (
    build_annual_comparison_chart,
    build_emerging_trend_chart,
    build_history_forecast_chart,
    build_prediction_map,
    build_territorial_ranking_chart,
    build_type_trend_chart,
)
from utils.predictions_data import (
    ALL_CRIME_GROUPS,
    ALL_CRIME_TYPES,
    ALL_MUNICIPALITIES,
    CRIME_GROUPS,
    FORECAST_QUARTERS,
    GROUP_LEVEL,
    HIGH_UNCERTAINTY,
    HISTORY_FORECAST_PATH,
    PREDICTIONS_PATH,
    RARE_CRIME_TYPES,
    TYPE_LEVEL,
    WAPE_BY_HORIZON,
    annual_comparison,
    emerging_quarterly_comparison,
    entity_trends_for_scope,
    individual_series,
    load_prediction_model,
    prediction_snapshot,
    prepare_prediction_map,
    territorial_summary,
    type_trends_for_scope,
)
from utils.ui import inject_global_styles


PAGE_ROOT = Path(__file__).resolve().parents[1]
STYLES_PATH = PAGE_ROOT / "assets" / "predictions_styles.css"


def _inject_styles() -> None:
    st.markdown(
        f"<style>{STYLES_PATH.read_text(encoding='utf-8')}</style>",
        unsafe_allow_html=True,
    )


def _number(value: object, digits: int = 1) -> str:
    output = f"{float(value):,.{digits}f}"
    return output.replace(",", "X").replace(".", ",").replace("X", ".")


def _integer(value: object) -> str:
    return f"{int(round(float(value))):,}".replace(",", ".")


def _change(value: float | None) -> str:
    if value is None or pd.isna(value):
        return "Sin base comparable"
    sign = "+" if value >= 0 else ""
    return f"{sign}{_number(value, 1)}%"


def _section(eyebrow: str, title: str, copy: str) -> None:
    st.markdown(
        f'<header class="prediction-section"><div><span>{escape(eyebrow)}</span>'
        f'<h2>{escape(title)}</h2></div><p>{escape(copy)}</p></header>',
        unsafe_allow_html=True,
    )


def _render_kpi(column, key: str, label: str, value: str, detail: str) -> None:
    with column:
        with st.container(border=True, key=f"prediction_kpi_{key}"):
            st.metric(label, value)
            st.caption(detail)


def _territorial_table(frame: pd.DataFrame) -> pd.DataFrame:
    output = frame.copy()
    output["Variación vs 2025"] = output["change_percent"].map(_change)
    return output[
        [
            "municipio",
            "predicted_count",
            "actual_2025",
            "change_absolute",
            "Variación vs 2025",
            "nivel_confianza",
        ]
    ].rename(
        columns={
            "municipio": "Municipio",
            "predicted_count": "Previsto 2026",
            "actual_2025": "Real 2025",
            "change_absolute": "Diferencia",
            "nivel_confianza": "Confianza",
        }
    )


def _show_territorial_table(frame: pd.DataFrame, key: str) -> None:
    st.dataframe(
        _territorial_table(frame),
        hide_index=True,
        width="stretch",
        key=key,
        column_config={
            "Previsto 2026": st.column_config.NumberColumn(format="%.1f"),
            "Real 2025": st.column_config.NumberColumn(format="%.0f"),
            "Diferencia": st.column_config.NumberColumn(format="%+.1f"),
        },
    )


def _reset_prediction_filters() -> None:
    st.session_state["prediction_analysis_level"] = TYPE_LEVEL
    st.session_state["prediction_municipality"] = ALL_MUNICIPALITIES
    st.session_state["prediction_crime_type"] = ALL_CRIME_TYPES
    st.session_state["prediction_quarter"] = FORECAST_QUARTERS[0]
    st.session_state["prediction_include_madrid"] = False


def _trend_card(label: str, row: pd.Series) -> str:
    uncertainty = (
        '<small class="uncertainty-tag">ALTA INCERTIDUMBRE · BAJA FRECUENCIA</small>'
        if str(row["nivel_confianza"]) == HIGH_UNCERTAINTY
        else '<small>PREDICCIÓN ESTÁNDAR</small>'
    )
    return (
        f'<article><span>{escape(label)}</span><h3>{escape(str(row["tipo de crimen"]))}</h3>'
        f'<strong>{escape(_change(float(row["change_percent"])))}</strong>{uncertainty}</article>'
    )


st.set_page_config(
    page_title="Predicciones 2026 | Crime Pulse Madrid",
    page_icon=str(PAGE_ROOT / "assets" / "icono.png"),
    layout="wide",
    initial_sidebar_state="collapsed",
)
inject_global_styles()
_inject_styles()
render_top_navigation("predictions")

try:
    model = load_prediction_model()
except (FileNotFoundError, ValueError, OSError) as exc:
    st.error(f"No se ha podido cargar el laboratorio predictivo: {exc}")
    st.stop()

municipalities = [ALL_MUNICIPALITIES, *sorted(model.predictions["municipio"].unique())]
analysis_levels = (TYPE_LEVEL, GROUP_LEVEL)
if st.session_state.get("prediction_analysis_level") not in analysis_levels:
    st.session_state["prediction_analysis_level"] = TYPE_LEVEL

active_analysis_level = st.session_state["prediction_analysis_level"]
crime_scope_options = (
    [ALL_CRIME_TYPES, *sorted(model.predictions["tipo de crimen"].unique())]
    if active_analysis_level == TYPE_LEVEL
    else [ALL_CRIME_GROUPS, *CRIME_GROUPS]
)
if st.session_state.get("prediction_municipality") not in municipalities:
    st.session_state["prediction_municipality"] = ALL_MUNICIPALITIES
if st.session_state.get("prediction_crime_type") not in crime_scope_options:
    st.session_state["prediction_crime_type"] = (
        ALL_CRIME_TYPES if active_analysis_level == TYPE_LEVEL else ALL_CRIME_GROUPS
    )
if st.session_state.get("prediction_quarter") not in FORECAST_QUARTERS:
    st.session_state["prediction_quarter"] = FORECAST_QUARTERS[0]
if "prediction_include_madrid" not in st.session_state:
    st.session_state["prediction_include_madrid"] = False

st.markdown(
    '<section class="prediction-hero"><div class="prediction-eyebrow">PREDICTIVE ANALYTICS · Q1–Q3 2026</div>'
    '<h1>PREDICCIONES <span>2026</span></h1>'
    '<h2>Forecast criminal por municipio y tipología delictiva</h2>'
    '<p>Estimaciones trimestrales construidas a partir de patrones observados entre 2023 y 2025 mediante un modelo predictivo híbrido. '
    'Son orientativas: no representan valores deterministas ni predicciones causales.</p>'
    '<div class="prediction-horizon"><span>HISTÓRICO</span><i></i><strong>12 TRIMESTRES</strong><b>→</b>'
    '<span>FORECAST</span><i></i><strong>Q1 · Q2 · Q3 2026</strong></div></section>',
    unsafe_allow_html=True,
)
st.markdown(
    f'<div class="prediction-source"><i></i>{_integer(model.audit.prediction_rows)} PREDICCIONES · '
    f'{model.audit.municipality_count} MUNICIPIOS · {model.audit.crime_type_count} TIPOLOGÍAS · '
    'DOS FUENTES RECONCILIADAS · 0 DUPLICADOS</div>',
    unsafe_allow_html=True,
)

st.radio(
    "NIVEL DE ANÁLISIS",
    options=analysis_levels,
    horizontal=True,
    key="prediction_analysis_level",
)

# Recalcular opciones tras un posible cambio del nivel de análisis.
active_analysis_level = st.session_state["prediction_analysis_level"]
crime_scope_options = (
    [ALL_CRIME_TYPES, *sorted(model.predictions["tipo de crimen"].unique())]
    if active_analysis_level == TYPE_LEVEL
    else [ALL_CRIME_GROUPS, *CRIME_GROUPS]
)
if st.session_state.get("prediction_crime_type") not in crime_scope_options:
    st.session_state["prediction_crime_type"] = (
        ALL_CRIME_TYPES if active_analysis_level == TYPE_LEVEL else ALL_CRIME_GROUPS
    )

with st.container(key="home_filter_bar"):
    filter_columns = st.columns([1.15, 1.55, .68, .82, .76], gap="small")
    with filter_columns[0]:
        selected_municipality = st.selectbox(
            "MUNICIPIO",
            options=municipalities,
            key="prediction_municipality",
            label_visibility="collapsed",
        )
    with filter_columns[1]:
        selected_type = st.selectbox(
            "TIPO DE CRIMEN O AGRUPACIÓN",
            options=crime_scope_options,
            key="prediction_crime_type",
            label_visibility="collapsed",
        )
    with filter_columns[2]:
        selected_quarter = st.selectbox(
            "HORIZONTE",
            options=FORECAST_QUARTERS,
            format_func=lambda quarter: f"{quarter} 2026",
            key="prediction_quarter",
            label_visibility="collapsed",
        )
    with filter_columns[3]:
        include_madrid = st.toggle(
            "INCLUIR MADRID",
            key="prediction_include_madrid",
            help=(
                "Afecta solo al mapa, ranking y tablas territoriales. "
                "No modifica las predicciones, KPIs ni agregados globales."
            ),
        )
    with filter_columns[4]:
        st.button(
            "RESTABLECER FILTROS",
            width="stretch",
            key="prediction_reset_filters",
            on_click=_reset_prediction_filters,
        )

selection_level = str(st.session_state["prediction_analysis_level"])
snapshot = prediction_snapshot(
    model,
    selected_municipality,
    selected_type,
    selected_quarter,
    selection_level,
)
series = individual_series(
    model,
    selected_municipality,
    selected_type,
    selection_level,
)
comparison = annual_comparison(series)
territorial = territorial_summary(
    model,
    selected_type,
    selected_quarter,
    selection_level,
)
type_trends = entity_trends_for_scope(
    model,
    selected_municipality,
    selection_level,
)

# "Incluir Madrid" afecta exclusivamente a comparaciones territoriales.
# Si el usuario selecciona Madrid explícitamente, esa selección tiene prioridad.
explicit_madrid = str(selected_municipality).strip().casefold() == "madrid"
territorial_include_madrid = include_madrid or explicit_madrid
territorial_view = (
    territorial.copy()
    if territorial_include_madrid
    else territorial.loc[~territorial["municipio"].eq("Madrid")].copy()
)
if territorial_view.empty:
    territorial_view = territorial.copy()

all_municipalities = selected_municipality == ALL_MUNICIPALITIES
all_crime_scope = (
    selected_type == ALL_CRIME_TYPES
    if selection_level == TYPE_LEVEL
    else selected_type == ALL_CRIME_GROUPS
)
scope_unit = "tipologías" if selection_level == TYPE_LEVEL else "agrupaciones"
if all_municipalities and all_crime_scope:
    scope_label = (
        "Visión global agregada · 37 municipios · 16 tipologías"
        if selection_level == TYPE_LEVEL
        else "Visión global agregada · 37 municipios · 8 agrupaciones"
    )
elif all_municipalities:
    scope_label = f"Agregado de todos los municipios · {selected_type}"
elif all_crime_scope:
    scope_label = f"{selected_municipality} · agregado de todas las {scope_unit}"
else:
    scope_label = f"{selected_municipality} · {selected_type}"
st.markdown(
    f'<div class="prediction-context"><span>ÁMBITO ACTIVO</span><strong>{escape(scope_label)}</strong>'
    f'<i></i><b>{escape(selected_quarter)} 2026</b></div>',
    unsafe_allow_html=True,
)

kpi_columns = st.columns(4, gap="small")
_render_kpi(
    kpi_columns[0],
    "forecast",
    (
        f"Criminalidad prevista · {selected_quarter} 2026"
        if all_crime_scope
        else f"Conteo previsto · {selected_quarter} 2026"
    ),
    _number(snapshot.predicted_count, 1),
    f"Real {selected_quarter} 2025: {_integer(snapshot.actual_2025)}",
)
_render_kpi(
    kpi_columns[1],
    "change",
    f"Variación vs {selected_quarter} 2025",
    _change(snapshot.change_percent),
    f"Diferencia absoluta: {_number(snapshot.change_absolute, 1)}",
)
_render_kpi(
    kpi_columns[2],
    "model",
    "Tratamiento predictivo",
    "13 estándar · 3 alta incertidumbre" if all_crime_scope else snapshot.model,
    (
        "Modelo híbrido · tratamientos diferenciados"
        if all_crime_scope
        else "Asignación metodológica de la tipología"
    ),
)
if all_crime_scope:
    _render_kpi(
        kpi_columns[3],
        "coverage",
        "Cobertura y confianza",
        f"{snapshot.municipality_count} municipio{'s' if snapshot.municipality_count != 1 else ''} / {snapshot.crime_type_count} tipologías",
        "Confianza heterogénea · no existe una etiqueta única",
    )
else:
    _render_kpi(
        kpi_columns[3],
        "confidence",
        "Nivel de confianza",
        snapshot.confidence,
        f"Cobertura: {snapshot.municipality_count} municipio{'s' if snapshot.municipality_count != 1 else ''}",
    )

if not all_crime_scope and snapshot.includes_high_uncertainty:
    st.markdown(
        '<aside class="prediction-alert"><span>ALTA INCERTIDUMBRE</span>'
        '<h3>La selección incluye eventos de muy baja frecuencia</h3>'
        '<p>Las tipologías de baja frecuencia incluidas utilizan mediana móvil de cuatro trimestres. '
        'La agregación del grupo conserva esa incertidumbre y debe interpretarse con especial cautela.</p></aside>',
        unsafe_allow_html=True,
    )

_section(
    "HISTÓRICO + FORECAST",
    "Del patrón observado a la estimación",
    f"{scope_label}. La línea continua representa datos reales; el tramo discontinuo conecta el último dato observado con Q1–Q3 2026.",
)
st.plotly_chart(
    build_history_forecast_chart(series),
    width="stretch",
    config={"displayModeBar": False, "responsive": True},
    key=f"prediction-history-{selection_level}-{selected_municipality}-{selected_type}",
)

_section(
    "COMPARACIÓN 2025 VS 2026",
    "Mismo trimestre, distinto horizonte",
    "La comparación utiliza Q1, Q2 y Q3 reales de 2025 frente al trimestre equivalente predicho para 2026.",
)
st.plotly_chart(
    build_annual_comparison_chart(comparison),
    width="stretch",
    config={"displayModeBar": False, "responsive": True},
    key=f"prediction-year-comparison-{selection_level}-{selected_municipality}-{selected_type}",
)

_section(
    "TENDENCIAS PREVISTAS",
    (
        "Qué tipologías crecen y cuáles descienden"
        if selection_level == TYPE_LEVEL
        else "Qué agrupaciones crecen y cuáles descienden"
    ),
    (
        (
            "Ranking por tipología para todos los municipios: Q1–Q3 2025 real frente a Q1–Q3 2026 predicho."
            if selection_level == TYPE_LEVEL
            else "Ranking por agrupación para todos los municipios: Q1–Q3 2025 real frente a Q1–Q3 2026 predicho."
        )
        if all_municipalities
        else (
            f"Ranking por tipología de {selected_municipality}: Q1–Q3 2025 real frente a Q1–Q3 2026 predicho."
            if selection_level == TYPE_LEVEL
            else f"Ranking por agrupación de {selected_municipality}: Q1–Q3 2025 real frente a Q1–Q3 2026 predicho."
        )
    ),
)
comparable_trends = type_trends.dropna(subset=["change_percent"])
strongest_growth = comparable_trends.sort_values("change_percent", ascending=False).iloc[0]
strongest_decline = comparable_trends.sort_values("change_percent", ascending=True).iloc[0]
st.markdown(
    '<div class="prediction-trend-summary">'
    + _trend_card("MAYOR CRECIMIENTO PREVISTO", strongest_growth)
    + _trend_card("MAYOR DESCENSO PREVISTO", strongest_decline)
    + '</div>',
    unsafe_allow_html=True,
)
st.plotly_chart(
    build_type_trend_chart(type_trends),
    width="stretch",
    config={"displayModeBar": False, "responsive": True},
    key=f"prediction-type-trends-{selection_level}-{selected_municipality}",
)

_section(
    "MAPA PREDICTIVO",
    (
        "Dónde se concentra la criminalidad total prevista"
        if all_crime_scope
        else "Dónde se concentra la selección delictiva"
    ),
    (
        f"Conteo previsto por municipio para {selected_type.lower()} en {selected_quarter} 2026. "
        f"La escala representa volumen absoluto, no una tasa por población. "
        f"{'Madrid incluido en la comparación territorial.' if territorial_include_madrid else 'Madrid excluido de la comparación territorial.'}"
    ),
)
try:
    map_source = prepare_prediction_map(territorial)
except ValueError as exc:
    map_source = None
    st.warning(f"No se ha podido preparar la cartografía predictiva: {exc}")

map_column, ranking_column = st.columns([1.5, .82], gap="large")
with map_column:
    with st.container(key="prediction_map_shell"):
        if map_source is not None and map_source.available:
            st.plotly_chart(
                build_prediction_map(
                    map_source,
                    selected_type,
                    selected_quarter,
                    None if all_municipalities else selected_municipality,
                    include_madrid=territorial_include_madrid,
                ),
                width="stretch",
                config={
                    "displayModeBar": False,
                    "scrollZoom": False,
                    "doubleClick": False,
                    "staticPlot": True,
                    "responsive": True,
                },
                key=f"prediction-map-{selection_level}-{selected_type}-{selected_quarter}-{selected_municipality}",
            )
        else:
            st.warning(
                map_source.message
                if map_source is not None and map_source.message
                else "El mapa no está disponible; el ranking territorial permanece operativo."
            )
with ranking_column:
    st.plotly_chart(
        build_territorial_ranking_chart(
            territorial_view,
            selected_municipality=(
                None if all_municipalities else selected_municipality
            ),
        ),
        width="stretch",
        config={"displayModeBar": False, "responsive": True},
        key=f"prediction-territorial-ranking-{selection_level}-{selected_type}-{selected_quarter}",
    )

with st.expander("Explorar ranking territorial completo"):
    st.caption(
        "El crecimiento y descenso se ordenan por diferencia absoluta para evitar que porcentajes sobre bases cercanas a cero dominen la lectura."
    )
    volume_tab, growth_tab, decline_tab = st.tabs(
        ["MAYOR CONTEO PREVISTO", "MAYOR CRECIMIENTO", "MAYOR DESCENSO"]
    )
    with volume_tab:
        _show_territorial_table(
            territorial_view.nlargest(10, "predicted_count"),
            f"prediction-volume-table-{selected_type}-{selected_quarter}",
        )
    with growth_tab:
        _show_territorial_table(
            territorial_view.nlargest(10, "change_absolute"),
            f"prediction-growth-table-{selected_type}-{selected_quarter}",
        )
    with decline_tab:
        _show_territorial_table(
            territorial_view.nsmallest(10, "change_absolute"),
            f"prediction-decline-table-{selected_type}-{selected_quarter}",
        )

emerging = model.emerging_trend
emerging_comparison = emerging_quarterly_comparison(model)
_section(
    "TENDENCIA EMERGENTE",
    str(emerging["tipo de crimen"]),
    "Selección automática entre tipologías con volumen real suficiente; describe una señal estadística, no una causa ni una certeza futura.",
)
emerging_left, emerging_right = st.columns([.8, 1.35], gap="large")
with emerging_left:
    st.markdown(
        '<section class="emerging-case"><span>SEÑAL Q1–Q3 · 2025 → 2026</span>'
        f'<h3>{escape(str(emerging["tipo de crimen"]))}</h3>'
        f'<strong>{_change(float(emerging["change_percent"]))}</strong>'
        f'<div><b>REAL 2025</b><p>{_number(emerging["real_2025"], 0)}</p></div>'
        f'<div><b>PREDICHO 2026</b><p>{_number(emerging["predicted_2026"], 1)}</p></div>'
        f'<small>Umbral mínimo dinámico: {_number(model.emerging_threshold, 2)} casos reales.</small></section>',
        unsafe_allow_html=True,
    )
with emerging_right:
    st.plotly_chart(
        build_emerging_trend_chart(emerging_comparison),
        width="stretch",
        config={"displayModeBar": False, "responsive": True},
        key="prediction-emerging-chart",
    )

top_territory = territorial_view.iloc[0]
principal_signal = (
    f"{top_territory['municipio']} encabeza el ranking territorial del filtro con {_number(top_territory['predicted_count'], 1)} casos previstos."
    if all_municipalities
    else f"{strongest_growth['tipo de crimen']} registra la mayor variación prevista en {selected_municipality} ({_change(float(strongest_growth['change_percent']))})."
)
insights = (
    f"El ámbito activo proyecta {_number(snapshot.predicted_count, 1)} casos para {selected_quarter} 2026 frente a {_integer(snapshot.actual_2025)} observados en 2025.",
    f"La diferencia interanual es {_number(snapshot.change_absolute, 1)} casos ({_change(snapshot.change_percent)}).",
    principal_signal,
)
insight_html = "".join(
    f'<article><b>{index:02d}</b><p>{escape(text)}</p></article>'
    for index, text in enumerate(insights, start=1)
)
st.markdown(
    '<section class="prediction-insights"><span>LECTURAS AUTOMÁTICAS · SIN CAUSALIDAD</span>'
    f'<h2>Señales del filtro activo</h2><div>{insight_html}</div></section>',
    unsafe_allow_html=True,
)

wape_cards = "".join(
    f'<article><b>{quarter}</b><strong>{_number(value, 2)}%</strong></article>'
    for quarter, value in WAPE_BY_HORIZON.items()
)
rare_items = "".join(f"<li>{escape(name)}</li>" for name in sorted(RARE_CRIME_TYPES))
with st.expander("Metodología, validación e incertidumbre"):
    st.markdown(
        '<section class="prediction-methodology-panel"><span>MODELO HÍBRIDO · SIN CAUSALIDAD</span>'
        '<div class="uncertainty-grid compact"><article class="standard"><b>PREDICCIÓN ESTÁNDAR</b>'
        '<h3>Gradient Boosting</h3><strong>13 tipologías · 1.443 predicciones</strong>'
        '<p>Variables temporales: lag 1–4, media móvil, tendencia reciente, variación interanual, trimestre, municipio y tipología.</p></article>'
        '<article class="uncertain"><b>ALTA INCERTIDUMBRE</b><h3>Mediana móvil de 4 trimestres</h3>'
        f'<strong>3 tipologías · {model.audit.rare_prediction_rows} predicciones</strong><ul>{rare_items}</ul></article></div>'
        '<div class="wape-heading"><b>ERROR DE VALIDACIÓN WAPE</b><p>Menor WAPE implica menor error relativo agregado en la validación; no representa un intervalo de confianza.</p></div>'
        f'<div class="wape-strip">{wape_cards}</div>'
        '<ul class="method-notes"><li><strong>Validación:</strong> predicción temporal recursiva sobre 2025; el rendimiento varía por horizonte y tipología.</li>'
        f'<li><strong>Cobertura histórica:</strong> según municipios incluidos en el ámbito del proyecto: {model.audit.historical_municipalities_by_year[2023]} en 2023, '
        f'{model.audit.historical_municipalities_by_year[2024]} en 2024 y {model.audit.historical_municipalities_by_year[2025]} en 2025. '
        'Las ausencias corresponden a municipios que no cumplían el umbral poblacional definido y no se imputan como datos faltantes.</li>'
        '<li><strong>Límites:</strong> no incorpora eventos extraordinarios, cambios normativos ni factores futuros no observados.</li></ul>'
        '<p class="method-conclusion">Las predicciones representan estimaciones estadísticas basadas en patrones históricos y no valores deterministas.</p>'
        f'<small>Fuentes en solo lectura: {escape(PREDICTIONS_PATH.name)} · {escape(HISTORY_FORECAST_PATH.name)}</small></section>',
        unsafe_allow_html=True,
    )
