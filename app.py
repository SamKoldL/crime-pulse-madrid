"""Centro de mando ejecutivo de Crime Pulse Madrid."""

from __future__ import annotations

import streamlit as st

from utils.charts import (
    build_quarterly_evolution_chart,
    build_home_bottom_ranking_chart,
    build_home_ranking_chart,
)
from utils.home_data import (
    ALL_HOME_CRIME_TYPES,
    ALL_HOME_MUNICIPALITIES,
    build_home_snapshot,
    load_home_model,
)
from utils.navigation import render_top_navigation
from utils.ui import (
    inject_global_styles,
    render_footer,
    render_hero,
    render_home_kpis,
    render_home_signals,
    render_section_heading,
)


st.set_page_config(
    page_title="Crime Pulse Madrid | Centro de mando ejecutivo",
    page_icon="assets/icono.png",
    layout="wide",
    initial_sidebar_state="collapsed",
)
inject_global_styles()

render_top_navigation("home")

# Ajustes visuales específicos de la Home para la limpieza final.
st.markdown(
    """
    <style>
    .live-status {
        font-size: 0.82rem !important;
    }
    .live-status span {
        width: 9px !important;
        height: 9px !important;
    }
    .home-clean-heading {
        margin: 1.15rem 0 1.05rem 0;
        padding-bottom: 0.85rem;
        border-bottom: 1px solid rgba(90, 205, 240, 0.16);
        color: #43d7f4;
        font-size: 0.92rem;
        font-weight: 800;
        letter-spacing: 0.16em;
        text-transform: uppercase;
    }
    [data-testid="stCaptionContainer"] {
        font-size: 0.9rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


def _render_home_clean_heading(title: str) -> None:
    st.markdown(
        f'<div class="home-clean-heading">{title}</div>',
        unsafe_allow_html=True,
    )

try:
    home_model = load_home_model()
except (FileNotFoundError, ValueError, OSError) as exc:
    st.error(f"No se ha podido cargar el centro de mando: {exc}")
    st.stop()

latest_year = int(home_model.years[0])
municipality_options = [ALL_HOME_MUNICIPALITIES, *home_model.municipalities]
crime_type_options = [ALL_HOME_CRIME_TYPES, *home_model.crime_types]


def _reset_home_filters() -> None:
    st.session_state["home_context_year"] = latest_year
    st.session_state["home_context_municipality"] = ALL_HOME_MUNICIPALITIES
    st.session_state["home_context_crime_type"] = ALL_HOME_CRIME_TYPES


if st.session_state.get("home_context_year") not in home_model.years:
    st.session_state["home_context_year"] = latest_year
if st.session_state.get("home_context_municipality") not in municipality_options:
    st.session_state["home_context_municipality"] = ALL_HOME_MUNICIPALITIES
if st.session_state.get("home_context_crime_type") not in crime_type_options:
    st.session_state["home_context_crime_type"] = ALL_HOME_CRIME_TYPES

render_hero()

with st.container(key="home_filter_bar"):
    filter_columns = st.columns([.72, 1.25, 1.65, .72], gap="small")
    with filter_columns[0]:
        selected_year = st.selectbox(
            "Año",
            options=home_model.years,
            key="home_context_year",
            label_visibility="collapsed",
        )
    with filter_columns[1]:
        selected_municipality = st.selectbox(
            "Municipio",
            options=municipality_options,
            key="home_context_municipality",
            label_visibility="collapsed",
        )
    with filter_columns[2]:
        selected_crime_type = st.selectbox(
            "Tipo de delito",
            options=crime_type_options,
            key="home_context_crime_type",
            label_visibility="collapsed",
        )
    with filter_columns[3]:
        st.button(
            "RESTABLECER FILTROS",
            width="stretch",
            on_click=_reset_home_filters,
            key="home_reset_filters",
        )

# Contexto preparado para compartirse con otras páginas en una futura iteración.
st.session_state["crime_pulse_global_context"] = {
    "year": int(selected_year),
    "municipality": selected_municipality,
    "crime_type": selected_crime_type,
}

try:
    snapshot = build_home_snapshot(
        home_model,
        selected_year,
        selected_municipality,
        selected_crime_type,
    )
except ValueError as exc:
    st.error(f"No se ha podido construir el contexto seleccionado: {exc}")
    st.stop()

st.markdown(
    f'<div class="live-status"><span></span> PANORAMA · {snapshot.year} · '
    f'{snapshot.municipality} · {snapshot.crime_type}</div>',
    unsafe_allow_html=True,
)
_render_home_clean_heading("PULSO GENERAL")
render_home_kpis(snapshot)

_render_home_clean_heading("¿DÓNDE SE CONCENTRA?")
territory_left, territory_right = st.columns([1, 1], gap="large")

with territory_left:
    if selected_municipality == ALL_HOME_MUNICIPALITIES:
        ranking_title = "TOP 5 ÍNDICE CRIMINAL POR MUNICIPIO"
    elif snapshot.selected_rank is None:
        ranking_title = f"REFERENCIA REGIONAL · {selected_municipality} SIN DATO COMPARABLE"
    else:
        ranking_title = f"ENTORNO DE RANKING · {selected_municipality}"

    st.markdown(
        f'<div class="chart-kicker">{ranking_title}</div>',
        unsafe_allow_html=True,
    )
    st.plotly_chart(
        build_home_ranking_chart(snapshot.territorial, selected_municipality),
        width="stretch",
        config={"displayModeBar": False, "responsive": True},
        key=f"home-ranking-{selected_year}-{selected_municipality}-{selected_crime_type}",
    )
    if selected_municipality != ALL_HOME_MUNICIPALITIES and snapshot.selected_rank is None:
        st.caption(
            f"{selected_municipality} no pertenece al universo criminal válido de {selected_year}; se mantiene el Top 5 regional como referencia."
        )

with territory_right:
    st.markdown(
        '<div class="chart-kicker">BOTTOM 5 ÍNDICE CRIMINAL POR MUNICIPIO</div>',
        unsafe_allow_html=True,
    )
    st.plotly_chart(
        build_home_bottom_ranking_chart(snapshot.territorial, selected_municipality),
        width="stretch",
        config={"displayModeBar": False, "responsive": True},
        key=f"home-bottom-ranking-{selected_year}-{selected_municipality}-{selected_crime_type}",
    )

_render_home_clean_heading("¿CÓMO EVOLUCIONA?")
st.plotly_chart(
    build_quarterly_evolution_chart(snapshot.quarterly, selected_year),
    width="stretch",
    config={"displayModeBar": False, "responsive": True},
    key=f"home-quarterly-{selected_year}-{selected_municipality}-{selected_crime_type}",
)

render_home_signals()
render_footer(selected_year, home_compact=True)
