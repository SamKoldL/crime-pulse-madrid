"""Interactive territorial view of eligible Crime Pulse Madrid municipalities."""

from __future__ import annotations

from pathlib import Path

import streamlit as st

from utils.data import DATA_PATH, get_available_years, load_master_data
from utils.map_charts import build_territorial_map
from utils.map_data import (
    CRIME_INDEX,
    get_eligible_municipalities,
    municipality_from_map_selection,
    prepare_map_source,
)
from utils.map_metrics import build_map_regional_snapshot
from utils.map_ui import ALL_MAP_MUNICIPALITIES, render_map_context_panel
from utils.ui import inject_global_styles, render_footer
from utils.navigation import render_top_navigation


PAGE_ROOT = Path(__file__).resolve().parents[1]
MAP_STYLES_PATH = PAGE_ROOT / "assets" / "map_styles.css"


def _inject_map_styles() -> None:
    st.markdown(
        f"<style>{MAP_STYLES_PATH.read_text(encoding='utf-8')}</style>",
        unsafe_allow_html=True,
    )


def _on_year_filter_change() -> None:
    st.session_state["map_selected_municipality"] = ALL_MAP_MUNICIPALITIES


def _reset_map_filters() -> None:
    st.session_state["map_selected_year"] = st.session_state["map_default_year"]
    st.session_state["map_selected_municipality"] = ALL_MAP_MUNICIPALITIES


def _on_map_selection() -> None:
    """Synchronize Plotly selection before the filter widgets are rendered."""
    chart_key = st.session_state.get("map_active_chart_key")
    event = st.session_state.get(chart_key, {}) if chart_key else {}
    clicked = municipality_from_map_selection(
        event,
        tuple(st.session_state.get("map_active_eligible_names", ())),
    )
    if clicked:
        st.session_state["map_selected_municipality"] = clicked


st.set_page_config(
    page_title="Mapa Criminal | Crime Pulse Madrid",
    page_icon="⌖",
    layout="wide",
    initial_sidebar_state="expanded",
)
inject_global_styles()
_inject_map_styles()
render_top_navigation("map")

try:
    master_df = load_master_data(DATA_PATH)
except (FileNotFoundError, ValueError) as exc:
    st.error(f"No se ha podido cargar la base de datos: {exc}")
    st.stop()

years = get_available_years(master_df)
st.markdown(
    '<section class="map-page-hero"><div class="map-page-eyebrow">INTELIGENCIA TERRITORIAL · VISTA GEOGRÁFICA</div><h1>MAPA <span>CRIMINAL</span></h1><p>Exploración municipal de la criminalidad ponderada, su distribución territorial y los principales indicadores asociados.</p></section>',
    unsafe_allow_html=True,
)

default_year = int(years[0])
st.session_state["map_default_year"] = default_year
if st.session_state.get("map_selected_year") not in years:
    st.session_state["map_selected_year"] = default_year

eligible_for_options = get_eligible_municipalities(
    master_df, st.session_state["map_selected_year"]
)
municipality_options = [
    ALL_MAP_MUNICIPALITIES,
    *sorted(eligible_for_options["Municipio"].astype(str).tolist()),
]
if st.session_state.get("map_selected_municipality") not in municipality_options:
    st.session_state["map_selected_municipality"] = ALL_MAP_MUNICIPALITIES

with st.container(key="home_filter_bar"):
    filter_columns = st.columns(
        [0.72, 1.65, 0.72],
        gap="small",
    )
    with filter_columns[0]:
        selected_year = st.selectbox(
            "AÑO",
            options=years,
            key="map_selected_year",
            on_change=_on_year_filter_change,
            label_visibility="collapsed",
        )
    with filter_columns[1]:
        selected_municipality = st.selectbox(
            "MUNICIPIO",
            options=municipality_options,
            key="map_selected_municipality",
            label_visibility="collapsed",
        )
    with filter_columns[2]:
        st.button(
            "RESTABLECER FILTROS",
            width="stretch",
            key="map_reset_filters",
            on_click=_reset_map_filters,
        )

eligible = eligible_for_options
st.markdown(
    f'<div class="map-eligible-status"><i></i>{len(eligible)} MUNICIPIOS ELEGIBLES · ÍNDICE DISPONIBLE · {selected_year}</div>',
    unsafe_allow_html=True,
)

try:
    map_source = prepare_map_source(master_df, year=selected_year)
except ValueError as exc:
    st.error(f"No se puede utilizar la cartografía municipal: {exc}")
    map_source = None

if map_source is not None and map_source.available:
    context_frame = map_source.frame
else:
    context_frame = (
        master_df.loc[master_df["Año"].eq(int(selected_year))]
        .copy()
        .sort_values("Municipio")
        .reset_index(drop=True)
    )
    context_frame["_eligible"] = context_frame[CRIME_INDEX].notna()
regional_snapshot = build_map_regional_snapshot(context_frame)

map_column, detail_column = st.columns([1.75, 0.8], gap="large")
with map_column:
    st.subheader("Panorama geográfico")
    if map_source is not None and map_source.available:
        chart_key = f"criminal_map_{selected_year}"
        st.session_state["map_active_chart_key"] = chart_key
        st.session_state["map_active_eligible_names"] = tuple(
            eligible["Municipio"].astype(str)
        )
        with st.container(key="criminal_map_frame"):
            st.plotly_chart(
                build_territorial_map(
                    map_source,
                    selected_year,
                    classified_frame=regional_snapshot.classified,
                    selected_municipality=(
                        None
                        if selected_municipality == ALL_MAP_MUNICIPALITIES
                        else selected_municipality
                    ),
                ),
                width="stretch",
                config={
                    "displayModeBar": False,
                    "displaylogo": False,
                    "scrollZoom": True,
                    "responsive": True,
                },
                key=chart_key,
                on_select=_on_map_selection,
                selection_mode="points",
            )
        if map_source.message:
            st.warning(map_source.message)
    else:
        with st.container(border=True, key="map_placeholder"):
            st.subheader("Cartografía temporalmente no disponible")
            st.warning(
                map_source.message
                if map_source is not None and map_source.message
                else "No hay una fuente geográfica municipal disponible."
            )
            st.markdown(
                "La visión regional y la ficha territorial permanecen disponibles. "
                "La aplicación volverá a intentar cargar la caché local o la fuente oficial."
            )

with detail_column:
    render_map_context_panel(regional_snapshot, selected_municipality)

render_footer(selected_year)
