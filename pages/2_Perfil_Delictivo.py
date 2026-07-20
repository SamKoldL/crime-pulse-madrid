"""Perfil delictivo: composición, evolución y lectura territorial por tipología."""

from __future__ import annotations

from html import escape
from pathlib import Path

import pandas as pd
import streamlit as st

from utils.crime_profile_charts import (
    build_crime_ranking_chart,
    build_frequency_gravity_matrix,
    build_group_composition_chart,
    build_night_profile_chart,
    build_territorial_ranking_chart,
    build_trend_diverging_chart,
)
from utils.crime_profile_data import (
    ALL_CRIME_TYPES_LABEL,
    ALL_MUNICIPALITIES_LABEL,
    ALL_YEARS_LABEL,
    ANNUAL_AVERAGE_VIEW,
    GROUP_LEVEL,
    IMPACT_VIEW,
    PERIOD_VARIATION_VIEW,
    PROFILE_DATA_PATH,
    TYPE_LEVEL,
    VOLUME_VIEW,
    YEARS,
    build_comparable_scope,
    build_drug_case,
    build_entity_trends,
    build_global_relevance_reference,
    build_profile_kpis,
    filter_profile_data,
    load_profile_workbook,
    summarize_crime_types,
    summarize_groups,
    summarize_territory,
)
from utils.navigation import render_top_navigation
from utils.ui import inject_global_styles, render_footer


PROJECT_ROOT = Path(__file__).resolve().parents[1]
PROFILE_STYLES_PATH = PROJECT_ROOT / "assets" / "crime_profile_styles.css"
WITH_MADRID = "CON MADRID"
WITHOUT_MADRID = "SIN MADRID"
ABSOLUTE_COUNT = "CONTEO ABSOLUTO"
MUNICIPAL_SHARE = "PESO DENTRO DEL MUNICIPIO (%)"
DRUG_CRIME_ID = "10"


def _inject_profile_styles() -> None:
    st.markdown(
        f"<style>{PROFILE_STYLES_PATH.read_text(encoding='utf-8')}</style>",
        unsafe_allow_html=True,
    )


def _format_integer(value: float | int) -> str:
    return f"{float(value):,.0f}".replace(",", ".")


def _format_decimal(value: float, digits: int = 1) -> str:
    formatted = f"{value:,.{digits}f}"
    return formatted.replace(",", "X").replace(".", ",").replace("X", ".")


def _format_change(value: object) -> str:
    if value is None or pd.isna(value):
        return "—"
    sign = "+" if float(value) > 0 else ""
    return f"{sign}{_format_decimal(float(value))}%"


def _section_heading(eyebrow: str, title: str, description: str) -> None:
    st.markdown(
        f'<header class="profile-section-heading"><div><span>{escape(eyebrow)}</span>'
        f'<h2>{escape(title)}</h2></div><p>{escape(description)}</p></header>',
        unsafe_allow_html=True,
    )


def _clean_section_heading(title: str) -> None:
    """Título único de sección para la versión ejecutiva y depurada."""
    st.markdown(
        f'<header class="profile-clean-section-heading"><h2>{escape(title)}</h2></header>',
        unsafe_allow_html=True,
    )


def _render_profile_kpi(
    position: int,
    label: str,
    value: str,
    detail: str,
    *,
    crime_id: str | None = None,
) -> None:
    """Renderiza una tarjeta KPI con una estructura visual única y estable."""
    icons = ("∑", "◫", "◇", "◐", "↗")
    identifier = (
        f'<small class="profile-kpi-card__id">ID {escape(crime_id)}</small>'
        if crime_id is not None
        else '<small class="profile-kpi-card__id" aria-hidden="true">&nbsp;</small>'
    )
    st.html(
        f'<article class="profile-kpi-card profile-kpi-card--{position}">'
        '<header class="profile-kpi-card__header">'
        f'<span class="profile-kpi-card__label">{escape(label)}</span>'
        f'<span class="profile-kpi-card__icon" aria-hidden="true">{icons[position]}</span>'
        '</header>'
        '<div class="profile-kpi-card__main">'
        f'{identifier}<strong>{escape(value)}</strong>'
        '</div>'
        '<div class="profile-kpi-card__divider" aria-hidden="true"></div>'
        f'<p class="profile-kpi-card__detail">{escape(detail)}</p>'
        '</article>'
    )


def _comparison_context(selected_year: int | None) -> tuple[tuple[int, ...], str | None, str]:
    if selected_year is None:
        return YEARS, "cumulative_change", "2023→2025"
    if selected_year == 2025:
        return (2024, 2025), "change_24_25", "2024→2025"
    if selected_year == 2024:
        return (2023, 2024), "change_23_24", "2023→2024"
    return (2023,), None, "Sin año anterior"


st.set_page_config(
    page_title="Perfil Delictivo | Crime Pulse Madrid",
    page_icon=str(PROJECT_ROOT / "assets" / "icono.png"),
    layout="wide",
    initial_sidebar_state="collapsed",
)
inject_global_styles()
_inject_profile_styles()
render_top_navigation("profile")

try:
    profile_df, _weight_catalog, audit = load_profile_workbook(PROFILE_DATA_PATH)
except (FileNotFoundError, ValueError, OSError) as exc:
    st.error(f"No se ha podido cargar el perfil delictivo: {exc}")
    st.stop()

type_catalog = (
    profile_df[["crime_id", "crime_type", "type_label", "group"]]
    .drop_duplicates()
    .assign(order=lambda frame: frame["crime_id"].astype(float))
    .sort_values("order")
    .reset_index(drop=True)
)
type_labels = type_catalog["type_label"].tolist()
type_id_by_label = dict(zip(type_catalog["type_label"], type_catalog["crime_id"]))
type_group_by_id = dict(zip(type_catalog["crime_id"], type_catalog["group"]))
drug_type_label = str(
    type_catalog.loc[type_catalog["crime_id"].eq(DRUG_CRIME_ID), "type_label"].iloc[0]
)
drug_case = build_drug_case(profile_df)


def _reset_profile_filters() -> None:
    st.session_state["profile_selected_year"] = ALL_YEARS_LABEL
    st.session_state["profile_selected_municipality"] = ALL_MUNICIPALITIES_LABEL
    st.session_state["profile_selected_crime_type"] = ALL_CRIME_TYPES_LABEL
    st.session_state["profile_evolution_type"] = drug_type_label
    st.session_state["profile_territorial_type"] = drug_type_label


year_options = [ALL_YEARS_LABEL, *reversed(YEARS)]
crime_type_options = [ALL_CRIME_TYPES_LABEL, *type_labels]
if st.session_state.get("profile_selected_year") not in year_options:
    st.session_state["profile_selected_year"] = ALL_YEARS_LABEL
if st.session_state.get("profile_selected_crime_type") not in crime_type_options:
    st.session_state["profile_selected_crime_type"] = ALL_CRIME_TYPES_LABEL

current_year_option = st.session_state["profile_selected_year"]
current_year = None if current_year_option == ALL_YEARS_LABEL else int(current_year_option)
municipality_scope = (
    profile_df
    if current_year is None
    else profile_df.loc[profile_df["year"].eq(current_year)]
)
municipality_options = [
    ALL_MUNICIPALITIES_LABEL,
    *sorted(municipality_scope["municipality"].astype(str).unique()),
]
if st.session_state.get("profile_selected_municipality") not in municipality_options:
    st.session_state["profile_selected_municipality"] = ALL_MUNICIPALITIES_LABEL

st.markdown(
    '<section class="profile-hero"><div class="profile-eyebrow">INTELIGENCIA ANALÍTICA · ESTRUCTURA DELICTIVA</div>'
    '<h1>PERFIL <span>DELICTIVO</span></h1>'
    '<h2>Composición, frecuencia e impacto de la criminalidad analizada</h2>'
    '<p>Qué delitos componen la criminalidad, cómo evolucionan y qué patrones temporales y territoriales presentan.</p>'
    '<div class="profile-bridge"><span>MAPA CRIMINAL</span><i></i><strong>DÓNDE</strong><b>→</b>'
    '<span>PERFIL DELICTIVO</span><i></i><strong>QUÉ Y CÓMO</strong><b>→</b>'
    '<span>OPTIMIZACIÓN</span><i></i><strong>RESPUESTA</strong></div></section>',
    unsafe_allow_html=True,
)

with st.container(key="home_filter_bar"):
    filter_columns = st.columns([.72, 1.25, 1.65, .72], gap="small")
    with filter_columns[0]:
        year_option = st.selectbox(
            "AÑO",
            options=year_options,
            key="profile_selected_year",
            label_visibility="collapsed",
        )
    with filter_columns[1]:
        selected_municipality = st.selectbox(
            "MUNICIPIO",
            options=municipality_options,
            key="profile_selected_municipality",
            label_visibility="collapsed",
        )
    with filter_columns[2]:
        selected_type_label = st.selectbox(
            "TIPO DE DELITO",
            options=crime_type_options,
            key="profile_selected_crime_type",
            label_visibility="collapsed",
        )
    with filter_columns[3]:
        st.button(
            "RESTABLECER FILTROS",
            width="stretch",
            key="profile_reset_filters",
            on_click=_reset_profile_filters,
        )

selected_year = None if year_option == ALL_YEARS_LABEL else int(year_option)
selected_crime_id = (
    None
    if selected_type_label == ALL_CRIME_TYPES_LABEL
    else str(type_id_by_label[selected_type_label])
)
selected_group = (
    None if selected_crime_id is None else str(type_group_by_id[selected_crime_id])
)
year_label = str(selected_year) if selected_year is not None else "2023–2025"
municipality_label = (
    "ÁMBITO COMPLETO"
    if selected_municipality == ALL_MUNICIPALITIES_LABEL
    else selected_municipality.upper()
)

# One load, one active context and explicit comparative universes.
comparative_scope = filter_profile_data(
    profile_df,
    selected_year,
    selected_municipality,
)
active_scope = filter_profile_data(
    profile_df,
    selected_year,
    selected_municipality,
    selected_crime_id,
)
comparison_years, growth_column, growth_period = _comparison_context(selected_year)
growth_scope, growth_cohort_size = build_comparable_scope(
    profile_df,
    selected_municipality,
    comparison_years,
)
comparative_type_summary = summarize_crime_types(comparative_scope)
group_summary = summarize_groups(comparative_scope)
growth_trends = build_entity_trends(growth_scope, TYPE_LEVEL)
group_growth_trends = build_entity_trends(growth_scope, GROUP_LEVEL)
relevant_crime_ids, relevance_threshold = build_global_relevance_reference(profile_df)
kpis = build_profile_kpis(
    active_scope,
    comparative_scope,
    growth_scope,
    selected_year,
    selected_crime_id,
    trends=growth_trends,
)

st.markdown(
    f'<div class="profile-source-status"><i></i>{escape(year_label)} · {escape(municipality_label)} · '
    f'{escape(selected_type_label.upper())} · {len(active_scope):,} OBSERVACIONES · '
    '16 IDS INDEPENDIENTES · PESOS 16/16</div>',
    unsafe_allow_html=True,
)

if selected_crime_id is None:
    composition_label = "Tipología más frecuente"
    composition_value = f"ID {kpis.top_type_id} · {kpis.top_type}"
    composition_detail = (
        f"{_format_integer(kpis.top_type_count)} · "
        f"{_format_decimal(kpis.top_type_share)}% del total"
    )
else:
    composition_label = "Peso dentro del total"
    composition_value = f"{_format_decimal(kpis.top_type_share)}%"
    composition_detail = f"ID {kpis.top_type_id} · {kpis.top_type}"

if kpis.growth_value is None:
    growth_label = "Variación comparable"
    growth_value = "SIN HISTÓRICO"
    growth_detail = "No existe un periodo anterior comparable"
elif selected_crime_id is None:
    growth_label = "Mayor crecimiento relevante"
    growth_value = f"ID {kpis.growth_id} · {kpis.growth_type}"
    growth_detail = (
        f"{_format_change(kpis.growth_value)} · {kpis.growth_period} · "
        f"{growth_cohort_size} municipios comunes"
    )
else:
    growth_label = "Variación comparable"
    growth_value = _format_change(kpis.growth_value)
    growth_detail = (
        f"{kpis.growth_period} · {growth_cohort_size} municipio"
        f"{'s' if growth_cohort_size != 1 else ''} común"
        f"{'es' if growth_cohort_size != 1 else ''}"
    )

kpi_content = (
    (
        "Total de delitos",
        _format_integer(kpis.total_crimes),
        f"Suma de conteos · {year_label}",
    ),
    (composition_label, composition_value, composition_detail),
    (
        "Carga delictiva ponderada",
        _format_integer(kpis.weighted_load),
        "Conteo × peso del delito · magnitud analítica",
    ),
    (
        "% de delitos nocturnos",
        f"{_format_decimal(kpis.night_share)}%",
        f"{_format_integer(kpis.total_night)} registros nocturnos",
    ),
    (growth_label, growth_value, growth_detail),
)
with st.container(key="profile_kpi_grid"):
    kpi_columns = st.columns(5, gap="small")
    for column, content, position in zip(kpi_columns, kpi_content, range(5)):
        label, value, detail = content
        entity_id: str | None = None
        if position == 1 and selected_crime_id is None:
            value = str(kpis.top_type)
            entity_id = str(kpis.top_type_id)
        elif position == 4 and selected_crime_id is None and kpis.growth_value is not None:
            value = str(kpis.growth_type)
            entity_id = str(kpis.growth_id)
        with column:
            _render_profile_kpi(
                position,
                label,
                value,
                detail,
                crime_id=entity_id,
            )

_clean_section_heading("Qué delitos predominan")
ranking_view = st.radio(
    "Métrica de radiografía",
    options=(VOLUME_VIEW, IMPACT_VIEW),
    horizontal=True,
    key="profile_ranking_view",
)
composition_tab, ranking_tab = st.tabs(["COMPOSICIÓN POR GRUPOS", "TIPOS DE DELITO"])
with composition_tab:
    st.plotly_chart(
        build_group_composition_chart(group_summary, ranking_view, selected_group),
        width="stretch",
        config={"displayModeBar": False, "responsive": True},
        key=f"profile-groups-{ranking_view}-{year_label}-{selected_municipality}-{selected_group}",
    )
with ranking_tab:
    st.plotly_chart(
        build_crime_ranking_chart(
            comparative_type_summary,
            ranking_view,
            selected_crime_id,
        ),
        width="stretch",
        config={"displayModeBar": False, "responsive": True},
        key=f"profile-ranking-{ranking_view}-{year_label}-{selected_municipality}-{selected_crime_id}",
    )

_clean_section_heading("Qué crece y qué disminuye")
trend_metric_view = st.radio(
    "Métrica de tendencias",
    options=(PERIOD_VARIATION_VIEW, ANNUAL_AVERAGE_VIEW),
    horizontal=True,
    key="profile_trend_metric_view",
)
if growth_column is None:
    st.info("2023 no dispone de un año anterior dentro de la fuente para construir una variación comparable.")
else:
    trend_type_tab, trend_group_tab = st.tabs(
        ["TIPOS DE DELITO", "COMPOSICIÓN POR GRUPOS"]
    )
    with trend_type_tab:
        st.plotly_chart(
            build_trend_diverging_chart(
                growth_trends,
                growth_column,
                growth_period,
                selected_crime_id,
                TYPE_LEVEL,
                trend_metric_view,
            ),
            width="stretch",
            config={"displayModeBar": False, "responsive": True},
            key=(
                f"profile-trends-types-{trend_metric_view}-{growth_period}-"
                f"{selected_municipality}-{selected_crime_id}"
            ),
        )
    with trend_group_tab:
        st.plotly_chart(
            build_trend_diverging_chart(
                group_growth_trends,
                growth_column,
                growth_period,
                selected_group,
                GROUP_LEVEL,
                trend_metric_view,
            ),
            width="stretch",
            config={"displayModeBar": False, "responsive": True},
            key=(
                f"profile-trends-groups-{trend_metric_view}-{growth_period}-"
                f"{selected_municipality}-{selected_group}"
            ),
        )
_clean_section_heading("Cuándo ocurre y cuánto pesa")
night_tab, matrix_tab = st.tabs(["DÍA VS NOCHE", "MATRIZ FRECUENCIA × GRAVEDAD"])
with night_tab:
    st.plotly_chart(
        build_night_profile_chart(comparative_type_summary, selected_crime_id),
        width="stretch",
        config={"displayModeBar": False, "responsive": True},
        key=f"profile-night-{year_label}-{selected_municipality}-{selected_crime_id}",
    )
    st.caption(
        f"% nocturno = Conteo Noche / Conteo. Día+noche difiere del total por ±1 en "
        f"{audit.day_night_difference_rows} filas; la fuente no se recalcula."
    )
with matrix_tab:
    st.plotly_chart(
        build_frequency_gravity_matrix(comparative_type_summary, selected_crime_id),
        width="stretch",
        config={"displayModeBar": False, "responsive": True},
        key=f"profile-matrix-{year_label}-{selected_municipality}-{selected_crime_id}",
    )
    st.caption(
        "Cuadrantes definidos con las medianas del contexto activo. Tamaño = conteo ponderado; los pesos originales no se modifican."
    )

_clean_section_heading("Cómo cambia el patrón entre municipios")
if selected_crime_id is None:
    if st.session_state.get("profile_territorial_type") not in type_labels:
        st.session_state["profile_territorial_type"] = drug_type_label
    territorial_type_label = st.selectbox(
        "Tipología territorial",
        options=type_labels,
        key="profile_territorial_type",
    )
    territorial_crime_id = str(type_id_by_label[territorial_type_label])
else:
    territorial_type_label = selected_type_label
    territorial_crime_id = selected_crime_id

territorial_control_columns = st.columns(2, gap="large")
with territorial_control_columns[0]:
    territorial_scope_option = st.radio(
        "Ámbito territorial",
        options=(WITH_MADRID, WITHOUT_MADRID),
        horizontal=True,
        key="profile_territorial_scope",
    )
with territorial_control_columns[1]:
    territorial_metric = st.radio(
        "Métrica territorial",
        options=(ABSOLUTE_COUNT, MUNICIPAL_SHARE),
        horizontal=True,
        key="profile_territorial_metric",
    )

territorial_summary = summarize_territory(
    profile_df,
    territorial_crime_id,
    selected_year,
)
if territorial_scope_option == WITHOUT_MADRID:
    territorial_summary = territorial_summary.loc[
        territorial_summary["municipality"].ne("Madrid")
    ].copy()
metric_column = "count" if territorial_metric == ABSOLUTE_COUNT else "share_municipality"
highlighted_municipality = (
    None
    if selected_municipality == ALL_MUNICIPALITIES_LABEL
    else selected_municipality
)
st.plotly_chart(
    build_territorial_ranking_chart(
        territorial_summary,
        highlighted_municipality,
        metric_column,
    ),
    width="stretch",
    config={"displayModeBar": False, "responsive": True},
    key=(
        f"profile-territory-{territorial_crime_id}-{year_label}-"
        f"{territorial_scope_option}-{metric_column}-{selected_municipality}"
    ),
)
night_candidates = comparative_type_summary.loc[
    comparative_type_summary["count"].gt(0)
    & comparative_type_summary["night_share"].notna()
].sort_values("night_share", ascending=False)
top_night = night_candidates.iloc[0]
if selected_crime_id is None:
    composition_insight = (
        f"ID {kpis.top_type_id} · {kpis.top_type} concentra {_format_decimal(kpis.top_type_share)}% "
        f"del volumen del contexto activo."
    )
else:
    composition_insight = (
        f"{selected_type_label} representa {_format_decimal(kpis.top_type_share)}% del total "
        f"del contexto activo."
    )

if kpis.growth_value is None:
    trend_insight = "El contexto activo no dispone de un periodo anterior comparable para calcular variación."
elif selected_crime_id is None:
    trend_insight = (
        f"ID {kpis.growth_id} · {kpis.growth_type} muestra el mayor crecimiento comparable del periodo "
        f"({_format_change(kpis.growth_value)})."
    )
else:
    trend_insight = (
        f"{selected_type_label} registra una variación comparable de "
        f"{_format_change(kpis.growth_value)} en {kpis.growth_period}."
    )

territorial_leader = territorial_summary.sort_values(metric_column, ascending=False).iloc[0]
if selected_crime_id is None:
    context_insight = (
        f"ID {top_night['crime_id']} · {top_night['crime_type']} presenta el mayor componente "
        f"nocturno ({_format_decimal(top_night['night_share'])}%) sobre "
        f"{_format_integer(top_night['count'])} casos."
    )
else:
    metric_text = (
        f"{_format_decimal(territorial_leader['share_municipality'])}% de su criminalidad interna"
        if metric_column == "share_municipality"
        else f"{_format_integer(territorial_leader['count'])} casos"
    )
    context_insight = (
        f"Para {territorial_type_label}, {territorial_leader['municipality']} encabeza la comparación "
        f"territorial seleccionada con {metric_text}."
    )

insights = (composition_insight, trend_insight, context_insight)
insight_html = "".join(
    f'<article><b>{position:02d}</b><p>{escape(text)}</p></article>'
    for position, text in enumerate(insights, start=1)
)
st.markdown(
    '<section class="profile-insights"><span>LECTURAS DESCRIPTIVAS · SIN INFERENCIA CAUSAL</span>'
    f'<h2>Contexto activo</h2><div class="profile-insight-list">{insight_html}</div></section>',
    unsafe_allow_html=True,
)

with st.expander("METODOLOGÍA Y LÍMITES"):
    st.markdown(
        f"""
- **Fuente:** hojas `Tabla Maestra` y `Peso Crimen` del Excel del proyecto.
- **Carga delictiva ponderada:** suma de `Conteo × Puntuación crimen`. Es una magnitud analítica, no una unidad oficial.
- **IDs independientes:** 5, 5.1, 5.2, 7 y 7.1 nunca se colapsan.
- **Cobertura descriptiva:** {audit.municipalities_by_year[2023]} municipios en 2023, {audit.municipalities_by_year[2024]} en 2024 y {audit.municipalities_by_year[2025]} en 2025. Las ausencias responden al universo elegible y no se imputan.
- **Comparaciones temporales:** utilizan municipios presentes en todos los años comparados. Para 2023→2025 son {drug_case.comparable_municipality_count} municipios comunes.
- **Variación descriptiva:** el gráfico muestra el cambio porcentual real del periodo para las 16 tipologías o los 8 grupos existentes, siempre sobre la misma cohorte territorial comparable.
- **Media anual:** se calcula como CAGR entre el valor inicial y final cuando ambos son positivos; no se divide linealmente la variación acumulada.
- **Crecimiento relevante:** el titular utiliza el mayor crecimiento sostenido calculado sobre la cohorte territorial comparable, la misma métrica del hallazgo validado. Los descensos relevantes conservan el umbral de volumen acumulado global 2023–2025 ({_format_integer(relevance_threshold)} casos).
- **Día/noche:** {audit.day_night_difference_rows} filas difieren en ±{audit.day_night_max_abs_difference}; se conserva `Conteo` como denominador sin modificar la fuente.
- **Interpretación:** todas las lecturas son descriptivas; no implican predicción ni causalidad.
        """
    )

render_footer(selected_year if selected_year is not None else "2023–2025")
