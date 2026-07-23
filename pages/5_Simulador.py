"""Simulador operativo de cobertura policial y presión criminal prevista."""

from __future__ import annotations

from html import escape
from pathlib import Path
from typing import Mapping, Sequence

import pandas as pd
import streamlit as st

from utils.navigation import render_top_navigation
from utils.police_optimization_data import (
    VOLUME_SCENARIO,
    prepare_optimization_map,
)
from utils.predictions_data import FORECAST_QUARTERS
from utils.simulator_charts import (
    build_coverage_change_chart,
    build_scenario_comparison_chart,
    build_simulator_map,
    build_simulation_timeseries_chart,
    build_pressure_gap_timeseries_chart,
)
from utils.simulator_data import (
    AVAILABLE_PROPOSALS,
    CURRENT_DISTRIBUTION,
    DEFAULT_FLEXIBILITY,
    FLEXIBILITY_RATIOS,
    load_simulator_model,
    maximum_transferable,
    proposal_movements,
    simulate_distribution,
    simulation_stats,
    validate_movements,
)
from utils.ui import inject_global_styles


PAGE_ROOT = Path(__file__).resolve().parents[1]
STYLES_PATH = PAGE_ROOT / "assets" / "simulator_styles.css"
DEFAULT_QUARTER = "Q3"


def _inject_styles() -> None:
    st.markdown(f"<style>{STYLES_PATH.read_text(encoding='utf-8')}</style>", unsafe_allow_html=True)


def _integer(value: object) -> str:
    return f"{int(round(float(value))):,}".replace(",", ".")


def _decimal(value: object, digits: int = 2) -> str:
    output = f"{float(value):,.{digits}f}"
    return output.replace(",", "X").replace(".", ",").replace("X", ".")


def _signed_percent(value: object) -> str:
    return f"{float(value):+.1f}%".replace(".", ",")


def _section(title: str) -> None:
    st.markdown(
        f'<header class="simulator-section"><h2>{escape(title)}</h2></header>',
        unsafe_allow_html=True,
    )


def _movement_signature(
    quarter: str,
    criterion: str,
    flexibility: str,
    movements: Sequence[Mapping[str, object]],
) -> tuple[object, ...]:
    return (
        quarter,
        criterion,
        flexibility,
        tuple(
            (
                int(item.get("id", index)),
                str(item["origin"]),
                str(item["destination"]),
                int(item["agents"]),
            )
            for index, item in enumerate(movements)
        ),
    )


def _invalidate_execution() -> None:
    st.session_state["simulator_executed_signature"] = None


def _on_criterion_change() -> None:
    if str(st.session_state.get("simulator_queue_source", "")).startswith("PROPUESTA"):
        st.session_state["simulator_queue_source"] = "PROPUESTA CARGADA · PENDIENTE DE REVISIÓN"
    _invalidate_execution()


def _reset_simulator() -> None:
    for key in list(st.session_state):
        if key.startswith("simulator_queue_amount_"):
            del st.session_state[key]
    st.session_state["simulator_quarter"] = DEFAULT_QUARTER
    st.session_state["simulator_criterion"] = VOLUME_SCENARIO
    st.session_state["simulator_flexibility"] = DEFAULT_FLEXIBILITY
    st.session_state["simulator_movements"] = []
    st.session_state["simulator_next_movement_id"] = 1
    st.session_state["simulator_queue_source"] = "MANUAL"
    st.session_state["simulator_map_view"] = "DESPUÉS"
    _invalidate_execution()


def _new_movement(origin: str, destination: str, agents: int) -> dict[str, object]:
    movement_id = int(st.session_state.get("simulator_next_movement_id", 1))
    st.session_state["simulator_next_movement_id"] = movement_id + 1
    return {"id": movement_id, "origin": origin, "destination": destination, "agents": int(agents)}


def _add_movement() -> None:
    origin = str(st.session_state["simulator_origin"])
    destination = str(st.session_state["simulator_destination"])
    agents = int(st.session_state["simulator_new_agents"])
    movements = [dict(item) for item in st.session_state.get("simulator_movements", [])]
    for movement in movements:
        if movement["origin"] == origin and movement["destination"] == destination:
            movement["agents"] = int(movement["agents"]) + agents
            break
    else:
        movements.append(_new_movement(origin, destination, agents))
    st.session_state["simulator_movements"] = movements
    st.session_state["simulator_queue_source"] = "PERSONALIZADO"
    _invalidate_execution()


def _delete_movement(movement_id: int) -> None:
    st.session_state["simulator_movements"] = [
        dict(item)
        for item in st.session_state.get("simulator_movements", [])
        if int(item.get("id", -1)) != movement_id
    ]
    st.session_state.pop(f"simulator_queue_amount_{movement_id}", None)
    st.session_state["simulator_queue_source"] = "PERSONALIZADO"
    _invalidate_execution()


def _clear_queue() -> None:
    for key in list(st.session_state):
        if key.startswith("simulator_queue_amount_"):
            del st.session_state[key]
    st.session_state["simulator_movements"] = []
    st.session_state["simulator_queue_source"] = "MANUAL"
    _invalidate_execution()


def _adjust_movement(movement_id: int, delta: int) -> None:
    movements = [dict(item) for item in st.session_state.get("simulator_movements", [])]
    for movement in movements:
        if int(movement.get("id", -1)) == movement_id:
            movement["agents"] = max(1, int(movement["agents"]) + int(delta))
            st.session_state[f"simulator_queue_amount_{movement_id}"] = movement["agents"]
            break
    st.session_state["simulator_movements"] = movements
    st.session_state["simulator_queue_source"] = "PERSONALIZADO"
    _invalidate_execution()


def _update_movement_amount(movement_id: int) -> None:
    key = f"simulator_queue_amount_{movement_id}"
    value = max(1, int(st.session_state[key]))
    movements = [dict(item) for item in st.session_state.get("simulator_movements", [])]
    for movement in movements:
        if int(movement.get("id", -1)) == movement_id:
            movement["agents"] = value
            break
    st.session_state["simulator_movements"] = movements
    st.session_state["simulator_queue_source"] = "PERSONALIZADO"
    _invalidate_execution()


def _set_new_agents(value: int) -> None:
    st.session_state["simulator_new_agents"] = max(1, int(value))


def _load_optimized_proposal() -> None:
    criterion = str(st.session_state["simulator_criterion"])
    flexibility = str(st.session_state["simulator_flexibility"])
    movements = proposal_movements(simulator_model, criterion, flexibility)
    st.session_state["simulator_movements"] = [
        _new_movement(str(item["origin"]), str(item["destination"]), int(item["agents"]))
        for item in movements
    ]
    st.session_state["simulator_queue_source"] = f"PROPUESTA ADAPTADA · {criterion}"
    _invalidate_execution()


def _execute_simulation() -> None:
    movements = list(st.session_state.get("simulator_movements", []))
    signature = _movement_signature(
        str(st.session_state["simulator_quarter"]),
        str(st.session_state["simulator_criterion"]),
        str(st.session_state["simulator_flexibility"]),
        movements,
    )
    st.session_state["simulator_executed_signature"] = signature
    st.session_state["simulator_map_view"] = "DESPUÉS"


def _impact_table(frame: pd.DataFrame, affected_only: bool) -> pd.DataFrame:
    output = frame.loc[frame["agent_change"].ne(0)].copy() if affected_only else frame.copy()
    output = output.sort_values("coverage_improvement_pct", ascending=False)
    return output[
        [
            "municipality",
            "predicted_crime",
            "agents_before",
            "agents_after",
            "agent_change",
            "pressure_before",
            "pressure_after",
            "coverage_improvement_pct",
        ]
    ].rename(
        columns={
            "municipality": "Municipio",
            "predicted_crime": "Forecast",
            "agents_before": "Agentes antes",
            "agents_after": "Agentes después",
            "agent_change": "Variación agentes",
            "pressure_before": "Presión antes",
            "pressure_after": "Presión después",
            "coverage_improvement_pct": "Mejora cobertura %",
        }
    )


def _comparison_frame(current: pd.DataFrame, scenario: pd.DataFrame, optimized: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for label, frame in (
        ("DISTRIBUCIÓN ACTUAL", current),
        ("TU ESCENARIO", scenario),
        ("REFERENCIA OPTIMIZADA", optimized),
    ):
        stats = simulation_stats(frame)
        rows.append(
            {
                "scenario": label,
                "gap": stats["gap_after"],
                "redistributed": stats["redistributed_agents"],
                "affected": stats["affected_municipalities"],
                "above_mean": stats["above_mean_after"],
                "max_pressure": stats["max_after_value"],
            }
        )
    return pd.DataFrame(rows)


st.set_page_config(
    page_title="Simulador | Crime Pulse Madrid",
    page_icon=str(PAGE_ROOT / "assets" / "icono.png"),
    layout="wide",
    initial_sidebar_state="collapsed",
)
inject_global_styles()
_inject_styles()
render_top_navigation("simulator")

try:
    simulator_model = load_simulator_model()
except (FileNotFoundError, ValueError, OSError) as exc:
    st.error(f"No se ha podido iniciar el simulador: {exc}")
    st.stop()

st.session_state.setdefault("simulator_quarter", DEFAULT_QUARTER)
st.session_state.setdefault("simulator_criterion", VOLUME_SCENARIO)
st.session_state.setdefault("simulator_flexibility", DEFAULT_FLEXIBILITY)
st.session_state.setdefault("simulator_movements", [])
st.session_state.setdefault("simulator_next_movement_id", 1)
st.session_state.setdefault("simulator_queue_source", "MANUAL")
st.session_state.setdefault("simulator_executed_signature", None)
st.session_state.setdefault("simulator_map_view", "DESPUÉS")

if st.session_state["simulator_quarter"] not in FORECAST_QUARTERS:
    st.session_state["simulator_quarter"] = DEFAULT_QUARTER
if st.session_state["simulator_criterion"] not in AVAILABLE_PROPOSALS:
    st.session_state["simulator_criterion"] = VOLUME_SCENARIO
if st.session_state["simulator_flexibility"] not in FLEXIBILITY_RATIOS:
    st.session_state["simulator_flexibility"] = DEFAULT_FLEXIBILITY

with st.container(key="home_filter_bar"):
    filter_columns = st.columns([0.9, 1.35, 1.15, 0.72], gap="small", vertical_alignment="bottom")
    with filter_columns[0]:
        st.selectbox(
            "Horizonte",
            options=FORECAST_QUARTERS,
            label_visibility="collapsed",
            format_func=lambda value: f"{value} 2026",
            key="simulator_quarter",
            on_change=_invalidate_execution,
        )
    with filter_columns[1]:
        st.selectbox(
            "Criterio de referencia",
            options=AVAILABLE_PROPOSALS,
            label_visibility="collapsed",
            key="simulator_criterion",
            on_change=_on_criterion_change,
        )
    with filter_columns[2]:
        st.selectbox(
            "Flexibilidad",
            options=tuple(FLEXIBILITY_RATIOS),
            label_visibility="collapsed",
            key="simulator_flexibility",
            on_change=_invalidate_execution,
        )
    with filter_columns[3]:
        st.button("RESTABLECER", width="stretch", key="simulator_reset_all", on_click=_reset_simulator)

selected_quarter = str(st.session_state["simulator_quarter"])
selected_criterion = str(st.session_state["simulator_criterion"])
selected_flexibility = str(st.session_state["simulator_flexibility"])

st.markdown(
    '<section class="simulator-hero"><div class="simulator-eyebrow">INTELIGENCIA OPERATIVA · ESCENARIOS INTERACTIVOS</div>'
    '<h1>SIMULADOR DE <span>REDISTRIBUCIÓN</span></h1>'
    '<h2>Cobertura policial frente a presión criminal prevista</h2>'
    '<p>Construye, valida y compara movimientos de agentes entre los municipios analizados.</p>'
    '<div class="simulator-caveat"><b>ESCENARIO NO CAUSAL</b> · Redistribuir agentes modifica la cobertura relativa; '
    'no demuestra ni cuantifica una reducción de la criminalidad.</div></section>',
    unsafe_allow_html=True,
)
st.markdown(
    f'<div class="simulator-source"><i></i>{simulator_model.audit.municipality_count} MUNICIPIOS · '
    f'{_integer(simulator_model.audit.total_agents)} AGENTES · FORECAST Q1–Q3 2026 · RECURSOS CONSTANTES</div>',
    unsafe_allow_html=True,
)

st.markdown(
    f'<aside class="simulator-scope"><b>BASE ÚNICA · DISTRIBUCIÓN POLICIAL ACTUAL</b>'
    f'<p>{escape(selected_quarter)} 2026 · {escape(selected_criterion)} · {escape(selected_flexibility)}. '
    'La presión utiliza el conteo total previsto por agente. El criterio ponderado solo selecciona la referencia existente de Optimización; '
    'no fabrica un forecast ponderado.</p></aside>',
    unsafe_allow_html=True,
)

_section("CENTRO DE MANDO")
st.markdown('<div class="simulator-command-label">CONSTRUIR MANUALMENTE</div>', unsafe_allow_html=True)

pending_movements = [dict(item) for item in st.session_state["simulator_movements"]]
municipalities = sorted(simulator_model.municipal_quarters["municipality"].unique())
st.session_state.setdefault("simulator_origin", municipalities[0])
if st.session_state["simulator_origin"] not in municipalities:
    st.session_state["simulator_origin"] = municipalities[0]
destinations = [name for name in municipalities if name != st.session_state["simulator_origin"]]
if st.session_state.get("simulator_destination") not in destinations:
    st.session_state["simulator_destination"] = destinations[0]

try:
    preview_frame = simulate_distribution(
        simulator_model,
        selected_quarter,
        CURRENT_DISTRIBUTION,
        selected_criterion,
        pending_movements,
    )
except ValueError:
    # Una cola inválida debe poder corregirse en pantalla, no bloquear la página.
    preview_frame = simulate_distribution(
        simulator_model,
        selected_quarter,
        CURRENT_DISTRIBUTION,
        selected_criterion,
        (),
    )
preview_by_name = preview_frame.set_index("municipality")

command_columns = st.columns([1.2, 1.25, 1.2, 0.75], gap="medium", vertical_alignment="top")
with command_columns[0]:
    selected_origin = st.selectbox("Municipio origen", municipalities, key="simulator_origin")

destinations = [name for name in municipalities if name != selected_origin]
if st.session_state.get("simulator_destination") not in destinations:
    st.session_state["simulator_destination"] = destinations[0]

max_transfer = maximum_transferable(
    simulator_model,
    selected_quarter,
    CURRENT_DISTRIBUTION,
    selected_criterion,
    pending_movements,
    selected_origin,
    selected_flexibility,
)
if "simulator_new_agents" not in st.session_state:
    st.session_state["simulator_new_agents"] = min(10, max(1, max_transfer))
if int(st.session_state["simulator_new_agents"]) > max(1, max_transfer):
    st.session_state["simulator_new_agents"] = max(1, max_transfer)

with command_columns[1]:
    movement_agents = st.number_input(
        "Agentes a trasladar",
        min_value=1,
        max_value=max(1, max_transfer),
        step=1,
        key="simulator_new_agents",
        disabled=max_transfer < 1,
        help=f"Capacidad restante del origen con este nivel de flexibilidad: {max_transfer}.",
    )
with command_columns[2]:
    selected_destination = st.selectbox("Municipio destino", destinations, key="simulator_destination")
with command_columns[3]:
    st.button(
        "AÑADIR",
        type="primary",
        width="stretch",
        key="simulator_add_movement",
        disabled=max_transfer < 1 or int(movement_agents) > max_transfer,
        on_click=_add_movement,
    )

origin_row = preview_by_name.loc[selected_origin]
destination_row = preview_by_name.loc[selected_destination]
with command_columns[0]:
    st.markdown(
        f'<div class="simulator-context"><b>{_integer(origin_row["agents_after"])} agentes disponibles</b>'
        f'<span>Presión { _decimal(origin_row["pressure_after"], 2) } · límite restante {max_transfer}</span></div>',
        unsafe_allow_html=True,
    )
with command_columns[1]:
    quick_columns = st.columns(4, gap="small")
    for quick_index, (column, quantity) in enumerate(
        zip(quick_columns, (5, 10, 25, max_transfer))
    ):
        label = "MAX" if quick_index == 3 else str(quantity)
        column.button(
            label,
            key=f"simulator_quick_{quick_index}_{selected_origin}_{max_transfer}",
            disabled=max_transfer < 1,
            on_click=_set_new_agents,
            args=(min(max(1, quantity), max(1, max_transfer)),),
        )
with command_columns[2]:
    st.markdown(
        f'<div class="simulator-context"><b>{_integer(destination_row["agents_after"])} agentes tras la cola</b>'
        f'<span>Presión { _decimal(destination_row["pressure_after"], 2) } · forecast {_decimal(destination_row["predicted_crime"], 1)}</span></div>',
        unsafe_allow_html=True,
    )

action_columns = st.columns([1.35, 0.8, 2.35], gap="medium")
action_columns[0].button(
    "CARGAR PROPUESTA OPTIMIZADA",
    width="stretch",
    key="simulator_load_proposal",
    on_click=_load_optimized_proposal,
)
action_columns[1].button(
    "VACIAR COLA",
    width="stretch",
    key="simulator_clear_queue",
    on_click=_clear_queue,
)
action_columns[2].caption(
    "La propuesta cargada se adapta al límite de flexibilidad para poder editarse. La referencia optimizada original se conserva íntegra en la comparación final."
)

pending_movements = [dict(item) for item in st.session_state["simulator_movements"]]
validation = validate_movements(simulator_model, selected_quarter, pending_movements, selected_flexibility)
transferred_agents = sum(int(item["agents"]) for item in pending_movements)
affected_municipalities = len(
    {str(item["origin"]) for item in pending_movements} | {str(item["destination"]) for item in pending_movements}
)
st.markdown(
    '<div class="simulator-preview">'
    f'<span><b>{len(pending_movements)}</b> movimientos</span>'
    f'<span><b>{_integer(transferred_agents)}</b> agentes en cola</span>'
    f'<span><b>{affected_municipalities}</b> municipios afectados</span>'
    f'<span><b>{_integer(simulator_model.audit.total_agents)}</b> total conservado</span>'
    f'<span class="{"valid" if validation.valid else "invalid"}">'
    f'{"ESCENARIO VÁLIDO" if validation.valid else "REVISIÓN NECESARIA"}</span></div>',
    unsafe_allow_html=True,
)

if pending_movements:
    st.markdown(
        f'<div class="simulator-queue-title"><span>COLA EDITABLE · {escape(str(st.session_state["simulator_queue_source"]))}</span>'
        '<small>Cualquier edición invalida el resultado ejecutado hasta volver a simular.</small></div>',
        unsafe_allow_html=True,
    )
    for movement in pending_movements:
        movement_id = int(movement["id"])
        amount_key = f"simulator_queue_amount_{movement_id}"
        st.session_state.setdefault(amount_key, int(movement["agents"]))
        row = st.columns([2.0, 0.35, 0.72, 0.35, 2.0, 0.55], gap="small", vertical_alignment="center")
        row[0].markdown(f'<div class="movement-place"><small>ORIGEN</small>{escape(str(movement["origin"]))}</div>', unsafe_allow_html=True)
        row[1].button("−", key=f"simulator_minus_{movement_id}", on_click=_adjust_movement, args=(movement_id, -1), width="stretch")
        row[2].number_input(
            "Agentes",
            min_value=1,
            max_value=simulator_model.audit.total_agents - 1,
            step=1,
            key=amount_key,
            label_visibility="collapsed",
            on_change=_update_movement_amount,
            args=(movement_id,),
        )
        row[3].button("+", key=f"simulator_plus_{movement_id}", on_click=_adjust_movement, args=(movement_id, 1), width="stretch")
        row[4].markdown(f'<div class="movement-place"><small>DESTINO</small>{escape(str(movement["destination"]))}</div>', unsafe_allow_html=True)
        row[5].button("×", key=f"simulator_delete_{movement_id}", on_click=_delete_movement, args=(movement_id,), width="stretch")
else:
    st.info("La cola está vacía. Añade un movimiento o carga la propuesta optimizada para comenzar.")

if validation.errors:
    st.error("\n\n".join(validation.errors))

current_signature = _movement_signature(selected_quarter, selected_criterion, selected_flexibility, pending_movements)
executed = bool(pending_movements) and st.session_state.get("simulator_executed_signature") == current_signature
st.button(
    "EJECUTAR ESCENARIO",
    type="primary",
    width="stretch",
    key="simulator_execute",
    disabled=not pending_movements or not validation.valid,
    on_click=_execute_simulation,
)

if pending_movements and not executed:
    st.markdown(
        '<div class="simulator-pending"><b>VISTA PREVIA LISTA</b><span>La cola todavía no afecta a mapas, KPIs ni comparación. '
        'Ejecuta el escenario cuando la validación sea correcta.</span></div>',
        unsafe_allow_html=True,
    )

if executed:
    result_frame = simulate_distribution(
        simulator_model,
        selected_quarter,
        CURRENT_DISTRIBUTION,
        selected_criterion,
        pending_movements,
        selected_flexibility,
    )
    current_frame = simulate_distribution(
        simulator_model,
        selected_quarter,
        CURRENT_DISTRIBUTION,
        selected_criterion,
        (),
    )
    exact_optimized_movements = proposal_movements(simulator_model, selected_criterion)
    optimized_frame = simulate_distribution(
        simulator_model,
        selected_quarter,
        CURRENT_DISTRIBUTION,
        selected_criterion,
        exact_optimized_movements,
    )
    stats = simulation_stats(result_frame)

    _section("RESULTADO EJECUTADO")
    kpis = st.columns(5, gap="small")
    kpi_data = (
        ("redistributed", "Agentes redistribuidos", _integer(stats["redistributed_agents"]), "Suma de transferencias positivas"),
        ("affected", "Municipios afectados", str(stats["affected_municipalities"]), f'de {simulator_model.audit.municipality_count}'),
        ("gap", "Brecha de presión", f'{_decimal(stats["gap_before"], 2)} → {_decimal(stats["gap_after"], 2)}', "Máximo menos mínimo"),
        ("above", "Sobre referencia", f'{stats["above_mean_before"]} → {stats["above_mean_after"]}', "Municipios"),
        ("maximum", "Mayor presión", str(stats["max_after_municipality"]), f'{_decimal(stats["max_before_value"], 2)} → {_decimal(stats["max_after_value"], 2)}'),
    )
    for column, (key, label, value, detail) in zip(kpis, kpi_data):
        with column:
            with st.container(border=True, key=f"simulator_kpi_{key}"):
                st.metric(label, value)
                st.caption(detail)

    _section("MAPA DE PRESIÓN")

    right_map_view = st.radio(
        "Vista del mapa derecho",
        options=("DESPUÉS", "VARIACIÓN"),
        horizontal=True,
        key="simulator_map_view",
    )

    try:
        map_source = prepare_optimization_map(result_frame)
    except ValueError as exc:
        map_source = None
        st.warning(f"No se ha podido preparar la cartografía: {exc}")

    if map_source is not None and map_source.available:
        map_left, map_right = st.columns(2, gap="medium")

        with map_left:
            st.markdown(
                '<div class="simulator-map-title">ANTES</div>',
                unsafe_allow_html=True,
            )
            with st.container(key="simulator_map_before_shell"):
                st.plotly_chart(
                    build_simulator_map(map_source, "ANTES"),
                    width="stretch",
                    config={"displayModeBar": False, "scrollZoom": True, "responsive": True},
                    key=f"simulator-map-before-{selected_quarter}-{hash(current_signature)}",
                )

        with map_right:
            st.markdown(
                f'<div class="simulator-map-title">{escape(right_map_view)}</div>',
                unsafe_allow_html=True,
            )
            with st.container(key="simulator_map_after_shell"):
                st.plotly_chart(
                    build_simulator_map(map_source, right_map_view),
                    width="stretch",
                    config={"displayModeBar": False, "scrollZoom": True, "responsive": True},
                    key=f"simulator-map-right-{selected_quarter}-{right_map_view}-{hash(current_signature)}",
                )
    else:
        st.warning(
            map_source.message
            if map_source is not None and map_source.message
            else "La cartografía municipal no está disponible; los resultados tabulares siguen operativos."
        )

    _section("COMPARACIÓN")
    comparison = _comparison_frame(current_frame, result_frame, optimized_frame)
    comparison_left, comparison_right = st.columns([1.12, 0.88], gap="large")
    with comparison_left:
        st.plotly_chart(
            build_scenario_comparison_chart(comparison),
            width="stretch",
            config={"displayModeBar": False, "responsive": True},
            key=f"simulator-comparison-{hash(current_signature)}",
        )
    with comparison_right:
        st.plotly_chart(
            build_coverage_change_chart(result_frame, count_each=5),
            width="stretch",
            config={"displayModeBar": False, "responsive": True},
            key=f"simulator-coverage-{hash(current_signature)}",
        )

    st.plotly_chart(
        build_simulation_timeseries_chart(
            simulator_model,
            result_frame,
            selected_quarter,
        ),
        width="stretch",
        config={"displayModeBar": False, "responsive": True},
        key=f"simulator-timeseries-{selected_quarter}-{hash(current_signature)}",
    )

    st.markdown(
        '<div class="simulator-subchart-title">IMPACTO DEL ESCENARIO SOBRE LA PRESIÓN TERRITORIAL</div>',
        unsafe_allow_html=True,
    )
    st.plotly_chart(
        build_pressure_gap_timeseries_chart(
            simulator_model,
            result_frame,
        ),
        width="stretch",
        config={"displayModeBar": False, "responsive": True},
        key=f"simulator-pressure-gap-timeseries-{hash(current_signature)}",
    )

    with st.expander("DETALLE MUNICIPAL DEL ESCENARIO"):
        affected_only = st.radio(
            "Ámbito de tabla",
            options=("SOLO MUNICIPIOS AFECTADOS", "TODOS LOS MUNICIPIOS"),
            horizontal=True,
            key="simulator_table_scope",
        ) == "SOLO MUNICIPIOS AFECTADOS"
        st.dataframe(
            _impact_table(result_frame, affected_only),
            hide_index=True,
            width="stretch",
            key=f"simulator-impact-table-{hash(current_signature)}-{affected_only}",
            column_config={
                "Forecast": st.column_config.NumberColumn(format="%.1f"),
                "Agentes antes": st.column_config.NumberColumn(format="%d"),
                "Agentes después": st.column_config.NumberColumn(format="%d"),
                "Variación agentes": st.column_config.NumberColumn(format="%+d"),
                "Presión antes": st.column_config.NumberColumn(format="%.2f"),
                "Presión después": st.column_config.NumberColumn(format="%.2f"),
                "Mejora cobertura %": st.column_config.NumberColumn(format="%+.1f%%"),
            },
        )

    gap_before = float(stats["gap_before"])
    gap_after = float(stats["gap_after"])
    gap_verb = "reduce" if gap_after < gap_before else "aumenta" if gap_after > gap_before else "mantiene"
    best_improvement = (
        f' La mayor mejora relativa de cobertura corresponde a <b>{escape(str(stats["best_improvement_municipality"]))}</b> '
        f'({_signed_percent(stats["best_improvement_pct"])}).'
        if float(stats["best_improvement_pct"]) > 0
        else ""
    )
    st.markdown(
        '<section class="simulator-insight"><span>LECTURA AUTOMÁTICA · SIN INFERENCIA CAUSAL</span>'
        f'<p>El escenario {gap_verb} la brecha máxima de presión de <b>{_decimal(gap_before, 2)}</b> a '
        f'<b>{_decimal(gap_after, 2)}</b>, redistribuye <b>{_integer(stats["redistributed_agents"])}</b> agentes y mantiene '
        f'constante la plantilla regional de <b>{_integer(stats["total_agents_after"])}</b>.{best_improvement} '
        'Estos resultados expresan capacidad relativa de cobertura y no una reducción causal de la criminalidad.</p></section>',
        unsafe_allow_html=True,
    )
else:
    st.markdown(
        '<section class="simulator-empty-result"><b>RESULTADOS PENDIENTES</b>'
        '<p>Los KPIs, mapas y comparaciones se generarán al ejecutar una cola válida. La edición previa permanece ligera.</p></section>',
        unsafe_allow_html=True,
    )

with st.expander("METODOLOGÍA Y LÍMITES"):
    st.markdown(
        '<div class="simulator-methodology"><article><b>PREDICCIÓN CRIMINAL</b><p>Forecast estadístico Q1–Q3 2026, '
        'agregado por municipio mediante suma. No se recalcula el modelo.</p></article>'
        '<article><b>COBERTURA</b><p>Presión = conteo previsto total / agentes. Una presión menor expresa mayor capacidad relativa, '
        'no menos delitos causados por la redistribución.</p></article>'
        '<article><b>REDISTRIBUCIÓN</b><p>Cada traslado resta agentes enteros al origen y los suma al destino. La plantilla regional '
        'permanece exactamente en 9.824 efectivos.</p></article>'
        '<article><b>FLEXIBILIDAD</b><p>Limita el total cedido por cada origen al 10 %, 25 % o 50 % de su plantilla actual, '
        'manteniendo al menos un agente.</p></article>'
        '<article><b>REFERENCIA OPTIMIZADA</b><p>El benchmark conserva la propuesta original de Optimización. Al cargarla en la cola, '
        'se adapta al límite elegido y por ello puede no reproducir exactamente la asignación original.</p></article>'
        '<article class="wide"><b>LIMITACIONES</b><p>No incorpora turnos, extensión territorial, tiempos de respuesta, prevención, '
        'eventos, especialización ni competencias de otros cuerpos. Es un modelo exploratorio de cobertura, no una dotación óptima definitiva.</p></article></div>',
        unsafe_allow_html=True,
    )

st.markdown(
    '<footer class="app-footer"><span>CRIME PULSE MADRID · SIMULADOR</span>'
    '<p>Predicción criminal, cobertura y escenario simulado permanecen conceptualmente separados.</p>'
    '<span class="footer-status"><i></i>DATOS EN SOLO LECTURA</span></footer>',
    unsafe_allow_html=True,
)
