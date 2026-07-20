"""Escenarios exploratorios de redistribución de Policía Local."""

from __future__ import annotations

from html import escape
from pathlib import Path

import pandas as pd
import streamlit as st

from utils.navigation import render_top_navigation
from utils.police_optimization_charts import (
    build_alignment_scatter,
    build_gap_ranking_chart,
    build_transfer_map,
    build_transfer_ranking_chart,
)
from utils.police_optimization_data import (
    GAP_TOLERANCE,
    POLICE_WORKBOOK_PATH,
    VOLUME_SCENARIO,
    WEIGHTED_SCENARIO,
    load_optimization_model,
    prepare_optimization_map,
    scenario_frame,
)
from utils.ui import inject_global_styles


PAGE_ROOT = Path(__file__).resolve().parents[1]
STYLES_PATH = PAGE_ROOT / "assets" / "police_optimization_styles.css"
ALL_MUNICIPALITIES = "TODOS LOS MUNICIPIOS"


def _inject_styles() -> None:
    st.markdown(
        f"<style>{STYLES_PATH.read_text(encoding='utf-8')}</style>",
        unsafe_allow_html=True,
    )


def _integer(value: object) -> str:
    return f"{int(round(float(value))):,}".replace(",", ".")


def _decimal(value: object, digits: int = 2) -> str:
    output = f"{float(value):,.{digits}f}"
    return output.replace(",", "X").replace(".", ",").replace("X", ".")


def _signed_integer(value: object) -> str:
    return f"{int(round(float(value))):+,}".replace(",", ".")


def _signed_pp(value: object) -> str:
    number = float(value) * 100
    prefix = "+" if number >= 0 else ""
    return f"{prefix}{_decimal(number, 2)} pp"


def _section(eyebrow: str, title: str, copy: str) -> None:
    st.markdown(
        f'<header class="optimization-section"><div><span>{escape(eyebrow)}</span>'
        f'<h2>{escape(title)}</h2></div><p>{escape(copy)}</p></header>',
        unsafe_allow_html=True,
    )


def _render_kpi(column, key: str, label: str, value: str, detail: str) -> None:
    with column:
        with st.container(border=True, key=f"optimization_kpi_{key}"):
            st.metric(label, value)
            st.caption(detail)


def _scenario_table(frame: pd.DataFrame, scenario: str) -> pd.DataFrame:
    pressure_label = (
        "Media anual de delitos"
        if scenario == VOLUME_SCENARIO
        else "Media anual del índice ponderado"
    )
    table = frame.copy()
    table["pressure_share_pct"] = table["pressure_share"] * 100
    table["police_share_pct"] = table["police_share"] * 100
    table["gap_pp"] = table["gap"] * 100
    return table[
        [
            "municipality",
            "pressure",
            "eligible_years",
            "pressure_share_pct",
            "police_share_pct",
            "gap_pp",
            "current_police",
            "proposed_police",
            "transfer",
        ]
    ].rename(
        columns={
            "municipality": "Municipio",
            "pressure": pressure_label,
            "eligible_years": "Años elegibles",
            "pressure_share_pct": "% presión",
            "police_share_pct": "% policías",
            "gap_pp": "Brecha (pp)",
            "current_police": "Policías actuales",
            "proposed_police": "Policías propuestos",
            "transfer": "Transferencia",
        }
    )


def _show_scenario_table(frame: pd.DataFrame, scenario: str, key: str) -> None:
    pressure_label = (
        "Media anual de delitos"
        if scenario == VOLUME_SCENARIO
        else "Media anual del índice ponderado"
    )
    st.dataframe(
        _scenario_table(frame, scenario),
        hide_index=True,
        width="stretch",
        key=key,
        column_config={
            pressure_label: st.column_config.NumberColumn(format="%.1f"),
            "Años elegibles": st.column_config.NumberColumn(format="%d"),
            "% presión": st.column_config.NumberColumn(format="%.2f%%"),
            "% policías": st.column_config.NumberColumn(format="%.2f%%"),
            "Brecha (pp)": st.column_config.NumberColumn(format="%+.2f"),
            "Policías actuales": st.column_config.NumberColumn(format="%d"),
            "Policías propuestos": st.column_config.NumberColumn(format="%d"),
            "Transferencia": st.column_config.NumberColumn(format="%+d"),
        },
    )


def _reset_filters() -> None:
    st.session_state["optimization_scenario"] = VOLUME_SCENARIO
    st.session_state["optimization_municipality"] = ALL_MUNICIPALITIES
    st.session_state["optimization_include_madrid"] = False


st.set_page_config(
    page_title="Optimización Policial | Crime Pulse Madrid",
    page_icon=str(PAGE_ROOT / "assets" / "icono.png"),
    layout="wide",
    initial_sidebar_state="collapsed",
)
inject_global_styles()
_inject_styles()
render_top_navigation("optimization")

try:
    model = load_optimization_model()
except (FileNotFoundError, ValueError, OSError) as exc:
    st.error(f"No se ha podido construir el modelo exploratorio: {exc}")
    st.stop()

municipalities = sorted(model.volume["municipality"].astype(str).unique())
municipality_options = [ALL_MUNICIPALITIES, *municipalities]
if st.session_state.get("optimization_scenario") not in (
    VOLUME_SCENARIO,
    WEIGHTED_SCENARIO,
):
    st.session_state["optimization_scenario"] = VOLUME_SCENARIO
if st.session_state.get("optimization_municipality") not in municipality_options:
    st.session_state["optimization_municipality"] = ALL_MUNICIPALITIES
if "optimization_include_madrid" not in st.session_state:
    st.session_state["optimization_include_madrid"] = False

st.markdown(
    '<section class="optimization-hero"><div class="optimization-eyebrow">INTELIGENCIA OPERATIVA · ESCENARIOS TERRITORIALES</div>'
    '<h1>OPTIMIZACIÓN <span>POLICIAL</span></h1>'
    '<h2>¿Está la distribución de Policía Local alineada con la presión criminal observada?</h2>'
    '<p>Dos escenarios teóricos redistribuyen una plantilla constante según volumen o gravedad. '
    'Son modelos exploratorios de alineación proporcional, no una dotación óptima definitiva.</p>'
    '<div class="optimization-flow"><span>MAPA</span><i></i><strong>DÓNDE</strong><b>→</b>'
    '<span>PERFIL</span><i></i><strong>QUÉ</strong><b>→</b>'
    '<span>OPTIMIZACIÓN</span><i></i><strong>ALINEACIÓN TEÓRICA</strong></div></section>',
    unsafe_allow_html=True,
)

with st.container(key="home_filter_bar"):
    filter_columns = st.columns([1.25, 1.55, 1.0, .82], gap="small")
    with filter_columns[0]:
        scenario = st.selectbox(
            "ESCENARIO",
            options=(VOLUME_SCENARIO, WEIGHTED_SCENARIO),
            key="optimization_scenario",
            label_visibility="collapsed",
        )
    with filter_columns[1]:
        municipality_filter = st.selectbox(
            "MUNICIPIO",
            options=municipality_options,
            key="optimization_municipality",
            label_visibility="collapsed",
        )
    with filter_columns[2]:
        include_madrid = st.toggle(
            "INCLUIR MADRID",
            key="optimization_include_madrid",
            help="Solo modifica visualizaciones comparativas; Madrid permanece en todos los cálculos.",
        )
    with filter_columns[3]:
        st.button(
            "RESTABLECER FILTROS",
            width="stretch",
            key="optimization_reset_filters",
            on_click=_reset_filters,
        )

selected_municipality = (
    None if municipality_filter == ALL_MUNICIPALITIES else municipality_filter
)
frame = scenario_frame(model, scenario)
scenario_copy = (
    "Cuota basada en la media anual de delitos observados durante los años elegibles de cada municipio."
    if scenario == VOLUME_SCENARIO
    else "Cuota basada en la media anual del índice criminal ponderado durante los años elegibles."
)

st.markdown(
    f'<div class="optimization-source"><i></i>37 MUNICIPIOS · 2023–2025 · '
    f'{_integer(model.audit.total_police)} POLICÍAS LOCALES · PLANTILLA CONSTANTE · MAYORES RESTOS</div>',
    unsafe_allow_html=True,
)
st.markdown(
    f'<aside class="scenario-definition"><span>{escape(scenario)}</span>'
    f'<p>{escape(scenario_copy)} Las ausencias no elegibles no se imputan como cero.</p></aside>',
    unsafe_allow_html=True,
)

moved_agents = int(frame["transfer"].clip(lower=0).sum())
receivers = int(frame["transfer"].gt(0).sum())
ceders = int(frame["transfer"].lt(0).sum())
pressure_peak = frame.loc[frame["gap"].idxmin()]
police_peak = frame.loc[frame["gap"].idxmax()]

primary_columns = st.columns(4, gap="small")
primary_kpis = (
    ("total", "Plantilla total", _integer(frame["current_police"].sum()), "Constante en todos los escenarios"),
    ("moved", "Agentes reasignados", _integer(moved_agents), "Suma de transferencias positivas"),
    ("receive", "Municipios receptores", _integer(receivers), "Transferencia teórica positiva"),
    ("cede", "Municipios cedentes", _integer(ceders), "Transferencia teórica negativa"),
)
for column, content in zip(primary_columns, primary_kpis):
    _render_kpi(column, *content)

relative_columns = st.columns(2, gap="small")
_render_kpi(
    relative_columns[0],
    "pressure_peak",
    "Mayor presión relativa frente a dotación proporcional",
    str(pressure_peak["municipality"]),
    f'Brecha de {_signed_pp(pressure_peak["gap"])} · lectura proporcional, no operativa',
)
_render_kpi(
    relative_columns[1],
    "police_peak",
    "Mayor dotación proporcional frente a presión",
    str(police_peak["municipality"]),
    f'Brecha de {_signed_pp(police_peak["gap"])} · lectura proporcional, no operativa',
)

if selected_municipality is not None:
    focus = frame.loc[frame["municipality"].eq(selected_municipality)].iloc[0]
    st.markdown(
        '<section class="optimization-focus"><div><span>FOCO MUNICIPAL</span>'
        f'<h3>{escape(selected_municipality)}</h3><p>{int(focus["eligible_years"])} año'
        f'{"s" if int(focus["eligible_years"]) != 1 else ""} elegible'
        f'{"s" if int(focus["eligible_years"]) != 1 else ""} en el periodo.</p></div>'
        f'<article><b>ACTUALES</b><strong>{_integer(focus["current_police"])}</strong></article>'
        f'<article><b>PROPUESTOS</b><strong>{_integer(focus["proposed_police"])}</strong></article>'
        f'<article><b>TRANSFERENCIA</b><strong>{_signed_integer(focus["transfer"])}</strong></article>'
        f'<article><b>BRECHA</b><strong>{_signed_pp(focus["gap"])}</strong></article></section>',
        unsafe_allow_html=True,
    )

_section(
    "ALINEACIÓN ENTRE PRESIÓN Y RECURSOS",
    "Dos cuotas, una referencia común",
    "La diagonal representa alineación proporcional y el equilibrio conserva una tolerancia de ±0,10 puntos porcentuales.",
)
alignment_column, gap_column = st.columns([1.05, 1], gap="large")
with alignment_column:
    st.plotly_chart(
        build_alignment_scatter(frame, include_madrid, selected_municipality),
        width="stretch",
        config={"displayModeBar": False, "responsive": True},
        key=f"optimization-alignment-{scenario}-{include_madrid}-{selected_municipality}",
    )
with gap_column:
    st.plotly_chart(
        build_gap_ranking_chart(
            frame,
            include_madrid=include_madrid,
            selected_municipality=selected_municipality,
        ),
        width="stretch",
        config={"displayModeBar": False, "responsive": True},
        key=f"optimization-gap-{scenario}-{include_madrid}-{selected_municipality}",
    )
st.caption(
    "Brecha = cuota policial − cuota de presión. Una brecha negativa indica mayor presión proporcional; una positiva, mayor dotación proporcional. No demuestra insuficiencia ni exceso operativo."
)

with st.expander("Consultar los 37 municipios"):
    _show_scenario_table(
        frame.sort_values("gap"),
        scenario,
        f"optimization-full-ranking-{scenario}",
    )

_section(
    "REDISTRIBUCIÓN TERRITORIAL",
    "Dónde se producirían los cambios",
    "El mapa mantiene un encuadre fijo; el ranking cuantifica los principales movimientos teóricos.",
)
try:
    map_source = prepare_optimization_map(frame)
except ValueError as exc:
    map_source = None
    st.warning(f"No se ha podido preparar la cartografía: {exc}")

map_column, transfer_column = st.columns([1.55, .8], gap="large")
with map_column:
    with st.container(key="optimization_map_shell"):
        if map_source is not None and map_source.available:
            st.plotly_chart(
                build_transfer_map(map_source, selected_municipality),
                width="stretch",
                config={
                    "displayModeBar": False,
                    "scrollZoom": False,
                    "doubleClick": False,
                    "staticPlot": True,
                    "responsive": True,
                },
                key=f"optimization-map-{scenario}-{selected_municipality}",
            )
        else:
            st.warning(
                map_source.message
                if map_source is not None and map_source.message
                else "La cartografía no está disponible; el ranking permanece operativo."
            )
with transfer_column:
    st.plotly_chart(
        build_transfer_ranking_chart(
            frame,
            include_madrid=include_madrid,
            selected_municipality=selected_municipality,
        ),
        width="stretch",
        config={"displayModeBar": False, "responsive": True},
        key=f"optimization-transfer-ranking-{scenario}-{include_madrid}-{selected_municipality}",
    )

transfer_peak = frame.loc[frame["transfer"].abs().idxmax()]
insights = (
    f'{pressure_peak["municipality"]} presenta la principal desalineación relativa del escenario ({_signed_pp(pressure_peak["gap"])}).',
    f'{transfer_peak["municipality"]} registra el mayor movimiento teórico en valor absoluto ({_signed_integer(transfer_peak["transfer"])} agentes).',
)
insight_html = "".join(
    f'<article><b>{index:02d}</b><p>{escape(text)}</p></article>'
    for index, text in enumerate(insights, start=1)
)
st.markdown(
    '<section class="optimization-insights"><span>LECTURAS DINÁMICAS · SIN INFERENCIA CAUSAL</span>'
    f'<h2>Lecturas del escenario</h2><div>{insight_html}</div></section>',
    unsafe_allow_html=True,
)

with st.expander("Metodología y limitaciones"):
    st.markdown(
        f"""
- **Escenario volumen:** redistribución proporcional basada en la media anual de delitos observados durante los años elegibles de cada municipio.
- **Escenario ponderado:** redistribución proporcional basada en la media anual del índice criminal ponderado durante los años elegibles.
- **Años elegibles:** las combinaciones que no cumplían el umbral poblacional conservan valores criminales nulos; no se interpretan como cero ni se imputan.
- **Redondeo:** método Hamilton o mayores restos, con desempate municipal determinista.
- **Plantilla:** ambos escenarios conservan exactamente {_integer(model.audit.total_police)} agentes y una transferencia neta de cero.
- **Brecha:** cuota policial menos cuota de presión. El equilibrio relativo mantiene una tolerancia de ±{_decimal(GAP_TOLERANCE * 100, 2)} puntos porcentuales.
- **Limitaciones:** el modelo no determina una dotación óptima. No incorpora turnos, extensión territorial, tiempos de respuesta, distribución intramunicipal, eventos extraordinarios, competencias de Policía Nacional o Guardia Civil, especialización, prevención ni necesidades operativas específicas.
- **Fuente:** `{POLICE_WORKBOOK_PATH.name}` y `tabla_maestra.csv`, utilizadas en solo lectura.
        """
    )
