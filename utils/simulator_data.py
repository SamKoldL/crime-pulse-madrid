"""Datos y reglas de conservación del simulador de cobertura policial.

La distribución policial actual es siempre el punto de partida. El forecast se
mantiene fijo y los movimientos solo modifican la presión prevista por agente.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import floor
from typing import Mapping, Sequence

import numpy as np
import pandas as pd
import streamlit as st

from utils.police_optimization_data import (
    EXPECTED_MUNICIPALITIES,
    EXPECTED_TOTAL_POLICE,
    VOLUME_SCENARIO,
    WEIGHTED_SCENARIO,
    load_optimization_model,
)
from utils.predictions_data import FORECAST_QUARTERS, load_prediction_model


CURRENT_DISTRIBUTION = "DISTRIBUCIÓN ACTUAL"
OPTIMIZED_DISTRIBUTION = "PROPUESTA OPTIMIZADA"
AVAILABLE_BASES = (CURRENT_DISTRIBUTION, OPTIMIZED_DISTRIBUTION)
AVAILABLE_PROPOSALS = (VOLUME_SCENARIO, WEIGHTED_SCENARIO)

CONSERVATIVE_FLEXIBILITY = "CONSERVADORA · 10 %"
MODERATE_FLEXIBILITY = "MODERADA · 25 %"
EXPERIMENTAL_FLEXIBILITY = "EXPERIMENTAL · 50 %"
FLEXIBILITY_RATIOS = {
    CONSERVATIVE_FLEXIBILITY: 0.10,
    MODERATE_FLEXIBILITY: 0.25,
    EXPERIMENTAL_FLEXIBILITY: 0.50,
}
DEFAULT_FLEXIBILITY = MODERATE_FLEXIBILITY


@dataclass(frozen=True)
class SimulatorAudit:
    municipality_count: int
    total_agents: int
    forecast_rows: int
    forecast_by_quarter: dict[str, float]
    volume_proposed_total: int
    weighted_proposed_total: int


@dataclass(frozen=True)
class SimulatorModel:
    municipal_quarters: pd.DataFrame
    audit: SimulatorAudit


@dataclass(frozen=True)
class MovementValidation:
    """Resultado ligero de validar la cola, sin preparar mapas ni figuras."""

    valid: bool
    errors: tuple[str, ...]
    outgoing_by_origin: dict[str, int]
    transfer_caps: dict[str, int]


def _proposal_column(proposal_scenario: str) -> str:
    if proposal_scenario == VOLUME_SCENARIO:
        return "volume_proposed_police"
    if proposal_scenario == WEIGHTED_SCENARIO:
        return "weighted_proposed_police"
    raise ValueError(f"Propuesta policial no reconocida: {proposal_scenario}.")


@st.cache_data(show_spinner=False)
def load_simulator_model() -> SimulatorModel:
    """Reconcilia forecast y distribución policial, ambos en solo lectura."""
    prediction_model = load_prediction_model()
    optimization_model = load_optimization_model()

    forecast = (
        prediction_model.predictions.groupby(
            ["municipio", "trimestre"], as_index=False, observed=True
        )["conteo_predicho"]
        .sum()
        .rename(
            columns={
                "municipio": "municipality",
                "trimestre": "quarter",
                "conteo_predicho": "predicted_crime",
            }
        )
    )
    police = optimization_model.volume[
        ["municipality", "current_police", "proposed_police"]
    ].rename(columns={"proposed_police": "volume_proposed_police"})
    police = police.merge(
        optimization_model.weighted[["municipality", "proposed_police"]].rename(
            columns={"proposed_police": "weighted_proposed_police"}
        ),
        on="municipality",
        validate="one_to_one",
    )
    municipal_quarters = forecast.merge(
        police,
        on="municipality",
        how="inner",
        validate="many_to_one",
    )

    expected_rows = EXPECTED_MUNICIPALITIES * len(FORECAST_QUARTERS)
    if len(municipal_quarters) != expected_rows:
        raise ValueError("El simulador no ha enlazado los 37 municipios en los tres trimestres.")
    if municipal_quarters.duplicated(["municipality", "quarter"]).any():
        raise ValueError("El forecast agregado contiene municipios/trimestres duplicados.")
    if set(municipal_quarters["quarter"]) != set(FORECAST_QUARTERS):
        raise ValueError("El simulador solo admite Q1, Q2 y Q3 de 2026.")
    if municipal_quarters["municipality"].nunique() != EXPECTED_MUNICIPALITIES:
        raise ValueError("La cobertura territorial del simulador debe ser de 37 municipios.")
    if (municipal_quarters["predicted_crime"] < 0).any():
        raise ValueError("La criminalidad prevista contiene valores negativos.")

    for column in (
        "current_police",
        "volume_proposed_police",
        "weighted_proposed_police",
    ):
        if not np.allclose(municipal_quarters[column], np.round(municipal_quarters[column])):
            raise ValueError("La distribución policial contiene agentes no enteros.")
        municipal_quarters[column] = municipal_quarters[column].astype(int)
        quarter_totals = municipal_quarters.groupby("quarter")[column].sum()
        if not quarter_totals.eq(EXPECTED_TOTAL_POLICE).all():
            raise ValueError(f"{column} no conserva los 9.824 agentes.")
        if municipal_quarters[column].le(0).any():
            raise ValueError(f"{column} contiene municipios sin agentes.")

    audit = SimulatorAudit(
        municipality_count=EXPECTED_MUNICIPALITIES,
        total_agents=EXPECTED_TOTAL_POLICE,
        forecast_rows=len(forecast),
        forecast_by_quarter={
            str(quarter): float(value)
            for quarter, value in forecast.groupby("quarter")["predicted_crime"].sum().items()
        },
        volume_proposed_total=int(optimization_model.volume["proposed_police"].sum()),
        weighted_proposed_total=int(optimization_model.weighted["proposed_police"].sum()),
    )
    return SimulatorModel(
        municipal_quarters.sort_values(["quarter", "municipality"]).reset_index(drop=True),
        audit,
    )


def _normalise_movement(movement: Mapping[str, object]) -> tuple[str, str, int]:
    origin = str(movement.get("origin", "")).strip()
    destination = str(movement.get("destination", "")).strip()
    raw_agents = movement.get("agents", 0)
    try:
        numeric_agents = float(raw_agents)
    except (TypeError, ValueError) as exc:
        raise ValueError("Cada movimiento debe utilizar un número entero de agentes.") from exc
    if isinstance(raw_agents, bool) or not numeric_agents.is_integer():
        raise ValueError("Cada movimiento debe utilizar un número entero de agentes.")
    agents = int(numeric_agents)
    if not origin or not destination:
        raise ValueError("Cada movimiento debe indicar origen y destino.")
    if origin == destination:
        raise ValueError("El municipio de origen y destino deben ser diferentes.")
    if agents <= 0:
        raise ValueError("El número de agentes trasladados debe ser positivo.")
    return origin, destination, agents


def _quarter_scope(model: SimulatorModel, quarter: str) -> pd.DataFrame:
    if quarter not in FORECAST_QUARTERS:
        raise ValueError(f"Horizonte no reconocido: {quarter}.")
    return model.municipal_quarters.loc[
        model.municipal_quarters["quarter"].eq(quarter)
    ].copy()


def transfer_cap(current_agents: int, flexibility: str) -> int:
    """Máximo agregado que puede ceder un municipio en el escenario."""
    if flexibility not in FLEXIBILITY_RATIOS:
        raise ValueError(f"Nivel de flexibilidad no reconocido: {flexibility}.")
    return max(0, min(int(current_agents) - 1, floor(int(current_agents) * FLEXIBILITY_RATIOS[flexibility])))


def validate_movements(
    model: SimulatorModel,
    quarter: str,
    movements: Sequence[Mapping[str, object]],
    flexibility: str,
) -> MovementValidation:
    """Valida municipios, enteros, conservación y límite agregado por origen."""
    scope = _quarter_scope(model, quarter)
    current = scope.set_index("municipality")["current_police"].astype(int).to_dict()
    available = dict(current)
    caps = {name: transfer_cap(agents, flexibility) for name, agents in current.items()}
    outgoing = {name: 0 for name in current}
    errors: list[str] = []

    for index, movement in enumerate(movements, start=1):
        try:
            origin, destination, agents = _normalise_movement(movement)
        except ValueError as exc:
            errors.append(f"Movimiento {index}: {exc}")
            continue
        if origin not in current or destination not in current:
            errors.append(f"Movimiento {index}: contiene un municipio fuera del modelo.")
            continue
        outgoing[origin] += agents
        if outgoing[origin] > caps[origin]:
            errors.append(
                f"{origin} cede {outgoing[origin]} agentes y supera su límite de {caps[origin]} "
                f"con flexibilidad {flexibility.lower()}."
            )
        if agents >= available[origin]:
            errors.append(
                f"Movimiento {index}: {origin} dispone de {available[origin]} agentes en ese punto "
                "y debe conservar al menos uno."
            )
            continue
        available[origin] -= agents
        available[destination] += agents

    return MovementValidation(
        valid=not errors,
        errors=tuple(dict.fromkeys(errors)),
        outgoing_by_origin={name: value for name, value in outgoing.items() if value},
        transfer_caps=caps,
    )


def simulate_distribution(
    model: SimulatorModel,
    quarter: str,
    base_mode: str = CURRENT_DISTRIBUTION,
    proposal_scenario: str = VOLUME_SCENARIO,
    movements: Sequence[Mapping[str, object]] = (),
    flexibility: str | None = None,
) -> pd.DataFrame:
    """Aplica traslados enteros sin crear agentes ni modificar el forecast."""
    if base_mode not in AVAILABLE_BASES:
        raise ValueError(f"Distribución base no reconocida: {base_mode}.")
    frame = _quarter_scope(model, quarter)
    start_column = "current_police" if base_mode == CURRENT_DISTRIBUTION else _proposal_column(proposal_scenario)
    frame["agents_before"] = frame[start_column].astype(int)
    frame["agents_after"] = frame["agents_before"].copy()

    if flexibility is not None:
        if base_mode != CURRENT_DISTRIBUTION:
            raise ValueError("Los límites de flexibilidad solo se aplican sobre la distribución actual.")
        validation = validate_movements(model, quarter, movements, flexibility)
        if not validation.valid:
            raise ValueError(" ".join(validation.errors))

    municipality_index = set(frame["municipality"])
    for movement in movements:
        origin, destination, agents = _normalise_movement(movement)
        if origin not in municipality_index or destination not in municipality_index:
            raise ValueError("Un movimiento contiene un municipio fuera del modelo.")
        origin_index = frame.index[frame["municipality"].eq(origin)][0]
        destination_index = frame.index[frame["municipality"].eq(destination)][0]
        available = int(frame.at[origin_index, "agents_after"])
        if agents >= available:
            raise ValueError(
                f"{origin} dispone de {available} agentes en ese punto del escenario; debe conservar al menos uno."
            )
        frame.at[origin_index, "agents_after"] = available - agents
        frame.at[destination_index, "agents_after"] = int(frame.at[destination_index, "agents_after"]) + agents

    total_before = int(frame["agents_before"].sum())
    total_after = int(frame["agents_after"].sum())
    if total_before != model.audit.total_agents or total_after != total_before:
        raise ValueError("La simulación no conserva exactamente la plantilla regional.")
    if frame["agents_after"].le(0).any():
        raise ValueError("La simulación dejaría un municipio sin agentes.")

    frame["agent_change"] = frame["agents_after"] - frame["agents_before"]
    frame["pressure_before"] = frame["predicted_crime"] / frame["agents_before"]
    frame["pressure_after"] = frame["predicted_crime"] / frame["agents_after"]
    regional_pressure = float(frame["predicted_crime"].sum() / total_before)
    frame["regional_pressure"] = regional_pressure
    frame["deviation_before"] = frame["pressure_before"] / regional_pressure - 1
    frame["deviation_after"] = frame["pressure_after"] / regional_pressure - 1
    frame["pressure_change_pct"] = (frame["pressure_after"] / frame["pressure_before"] - 1) * 100
    frame["coverage_improvement_pct"] = -frame["pressure_change_pct"]
    pressure_ceiling = float(max(frame["pressure_before"].max(), frame["pressure_after"].max()))
    frame["pressure_index_before"] = frame["pressure_before"] / pressure_ceiling * 100
    frame["pressure_index_after"] = frame["pressure_after"] / pressure_ceiling * 100
    return frame.sort_values("pressure_after", ascending=False).reset_index(drop=True)


def maximum_transferable(
    model: SimulatorModel,
    quarter: str,
    base_mode: str,
    proposal_scenario: str,
    movements: Sequence[Mapping[str, object]],
    origin: str,
    flexibility: str = DEFAULT_FLEXIBILITY,
) -> int:
    """Capacidad restante del origen según su plantilla actual y flexibilidad."""
    if base_mode != CURRENT_DISTRIBUTION:
        raise ValueError("El simulador operativo comienza siempre desde la distribución actual.")
    scope = _quarter_scope(model, quarter)
    row = scope.loc[scope["municipality"].eq(origin)]
    if len(row) != 1:
        raise ValueError(f"Municipio de origen no reconocido: {origin}.")
    cap = transfer_cap(int(row.iloc[0]["current_police"]), flexibility)
    outgoing = 0
    for movement in movements:
        try:
            movement_origin, _, agents = _normalise_movement(movement)
        except ValueError:
            continue
        if movement_origin == origin:
            outgoing += agents
    return max(0, cap - outgoing)


def proposal_movements(
    model: SimulatorModel,
    proposal_scenario: str,
    flexibility: str | None = None,
) -> list[dict[str, object]]:
    """Convierte la propuesta de Optimización en una cola de traslados.

    Sin flexibilidad reproduce exactamente el benchmark original. Con un nivel
    de flexibilidad crea una adaptación editable y recorta únicamente el flujo
    saliente de cada donante al límite permitido.
    """
    scope = _quarter_scope(model, FORECAST_QUARTERS[0])
    target_column = _proposal_column(proposal_scenario)
    scope["delta"] = scope[target_column] - scope["current_police"]
    donors = [
        [str(row["municipality"]), int(-row["delta"])]
        for _, row in scope.loc[scope["delta"].lt(0)].sort_values(["delta", "municipality"]).iterrows()
    ]
    receivers = [
        [str(row["municipality"]), int(row["delta"])]
        for _, row in scope.loc[scope["delta"].gt(0)]
        .sort_values(["delta", "municipality"], ascending=[False, True])
        .iterrows()
    ]
    exact: list[dict[str, object]] = []
    donor_index = receiver_index = 0
    while donor_index < len(donors) and receiver_index < len(receivers):
        amount = min(donors[donor_index][1], receivers[receiver_index][1])
        exact.append({"origin": donors[donor_index][0], "destination": receivers[receiver_index][0], "agents": int(amount)})
        donors[donor_index][1] -= amount
        receivers[receiver_index][1] -= amount
        if donors[donor_index][1] == 0:
            donor_index += 1
        if receivers[receiver_index][1] == 0:
            receiver_index += 1
    if donor_index != len(donors) or receiver_index != len(receivers):
        raise ValueError("La propuesta optimizada no puede expresarse como traslados equilibrados.")

    if flexibility is None:
        result = simulate_distribution(model, FORECAST_QUARTERS[0], CURRENT_DISTRIBUTION, proposal_scenario, exact)
        target = scope.set_index("municipality")[target_column].astype(int)
        actual = result.set_index("municipality")["agents_after"].astype(int).reindex(target.index)
        if not actual.equals(target):
            raise ValueError("Los movimientos no reproducen exactamente la propuesta optimizada.")
        return exact

    current = scope.set_index("municipality")["current_police"].astype(int).to_dict()
    remaining = {name: transfer_cap(agents, flexibility) for name, agents in current.items()}
    adapted: list[dict[str, object]] = []
    for movement in exact:
        origin, destination, agents = _normalise_movement(movement)
        amount = min(agents, remaining[origin])
        if amount > 0:
            adapted.append({"origin": origin, "destination": destination, "agents": amount})
            remaining[origin] -= amount
    validation = validate_movements(model, FORECAST_QUARTERS[0], adapted, flexibility)
    if not validation.valid:
        raise ValueError("La adaptación de la propuesta no cumple el nivel de flexibilidad.")
    return adapted


def simulation_stats(frame: pd.DataFrame) -> dict[str, object]:
    """Resume antes/después sin atribuir cambios criminales a la dotación."""
    before_max = frame.loc[frame["pressure_before"].idxmax()]
    after_max = frame.loc[frame["pressure_after"].idxmax()]
    before_min = frame.loc[frame["pressure_before"].idxmin()]
    after_min = frame.loc[frame["pressure_after"].idxmin()]
    best = frame.loc[frame["coverage_improvement_pct"].idxmax()]
    mean_pressure = float(frame["regional_pressure"].iloc[0])
    return {
        "total_agents_before": int(frame["agents_before"].sum()),
        "total_agents_after": int(frame["agents_after"].sum()),
        "redistributed_agents": int(frame["agent_change"].clip(lower=0).sum()),
        "affected_municipalities": int(frame["agent_change"].ne(0).sum()),
        "regional_pressure_before": mean_pressure,
        "regional_pressure_after": float(frame["predicted_crime"].sum() / frame["agents_after"].sum()),
        "max_before_municipality": str(before_max["municipality"]),
        "max_before_value": float(before_max["pressure_before"]),
        "max_after_municipality": str(after_max["municipality"]),
        "max_after_value": float(after_max["pressure_after"]),
        "min_before_municipality": str(before_min["municipality"]),
        "min_before_value": float(before_min["pressure_before"]),
        "min_after_municipality": str(after_min["municipality"]),
        "min_after_value": float(after_min["pressure_after"]),
        "gap_before": float(frame["pressure_before"].max() - frame["pressure_before"].min()),
        "gap_after": float(frame["pressure_after"].max() - frame["pressure_after"].min()),
        "above_mean_before": int(frame["pressure_before"].gt(mean_pressure).sum()),
        "above_mean_after": int(frame["pressure_after"].gt(mean_pressure).sum()),
        "best_improvement_municipality": str(best["municipality"]),
        "best_improvement_pct": float(best["coverage_improvement_pct"]),
    }
