"""Fuentes y cálculos del modelo exploratorio de redistribución policial.

La plantilla y el volumen proceden del Excel policial. La presión ponderada
procede de ``tabla_maestra.csv`` para conservar exactamente la metodología
criminal vigente en el proyecto. Ninguna fuente se modifica o recalcula.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

from utils.map_data import (
    GEOJSON_PATH,
    MapSource,
    prepare_project_map_source,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
POLICE_WORKBOOK_PATH = PROJECT_ROOT / "data" / "Distirbucion Policial (Chao).xlsx"
MASTER_DATA_PATH = PROJECT_ROOT / "data" / "tabla_maestra.csv"
POLICE_SHEET = "Distribucion policial-crimen"
SOURCE_REDISTRIBUTION_SHEET = "Redistribucion"

VOLUME_SCENARIO = "VOLUMEN DELICTIVO"
WEIGHTED_SCENARIO = "GRAVEDAD PONDERADA"
EXPECTED_MUNICIPALITIES = 37
EXPECTED_TOTAL_POLICE = 9_824
GAP_TOLERANCE = 0.001  # ±0,10 puntos porcentuales de cuota.

POLICE_COLUMNS = {
    "Municipio": "municipality",
    "Conteo de Crimen": "crime_count",
    "Conteo Policias locales (2024)": "current_police",
    "Crime Share": "source_crime_share",
    "Police Share": "source_police_share",
    "Difference": "source_gap",
    "Crime/police officer": "source_crime_per_officer",
}


@dataclass(frozen=True)
class OptimizationAudit:
    """Reconciliaciones críticas realizadas durante la carga."""

    municipality_count: int
    total_police: int
    total_volume: int
    total_weighted_pressure: float
    total_volume_annual_mean_pressure: float
    total_weighted_annual_mean_pressure: float
    volume_proposed_total: int
    weighted_proposed_total: int
    volume_transfer_total: int
    weighted_transfer_total: int
    source_redistribution_total: int
    source_redistribution_transfer_total: int
    max_source_share_difference: float
    ineligible_placeholder_rows: int
    eligible_years_by_municipality: dict[str, int]
    observed_years_by_municipality: dict[str, tuple[int, ...]]


@dataclass(frozen=True)
class OptimizationModel:
    """Los dos escenarios enteros y su comparación municipal."""

    volume: pd.DataFrame
    weighted: pd.DataFrame
    comparison: pd.DataFrame
    audit: OptimizationAudit


def _require_columns(frame: pd.DataFrame, required: set[str], source: str) -> None:
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(
            f"{source} no contiene las columnas requeridas: {', '.join(sorted(missing))}."
        )


def _classify_gap(gap: pd.Series) -> pd.Series:
    return pd.Series(
        np.select(
            [gap.lt(-GAP_TOLERANCE), gap.gt(GAP_TOLERANCE)],
            [
                "Presión proporcionalmente mayor",
                "Dotación proporcionalmente mayor",
            ],
            default="Equilibrio relativo",
        ),
        index=gap.index,
        dtype="string",
    )


def _hamilton_allocation(
    base: pd.DataFrame,
    pressure_column: str,
    scenario: str,
) -> pd.DataFrame:
    """Distribuye una plantilla fija mediante el método de mayores restos."""
    frame = base.copy()
    pressure_total = float(frame[pressure_column].sum())
    if pressure_total <= 0:
        raise ValueError(f"La presión total de {scenario} debe ser positiva.")

    total_police = int(frame["current_police"].sum())
    frame["pressure"] = pd.to_numeric(frame[pressure_column], errors="raise")
    frame["pressure_share"] = frame["pressure"] / pressure_total
    frame["police_share"] = frame["current_police"] / total_police
    frame["gap"] = frame["police_share"] - frame["pressure_share"]

    # La cuota decimal queda interna; solo proposed_police y transfer llegan a UI.
    frame["_quota"] = frame["pressure_share"] * total_police
    frame["proposed_police"] = np.floor(frame["_quota"]).astype(int)
    frame["_remainder"] = frame["_quota"] - frame["proposed_police"]
    remaining = total_police - int(frame["proposed_police"].sum())
    remainder_order = frame.sort_values(
        ["_remainder", "municipality"],
        ascending=[False, True],
        kind="mergesort",
    ).index
    if remaining:
        frame.loc[remainder_order[:remaining], "proposed_police"] += 1

    frame["current_police"] = frame["current_police"].astype(int)
    frame["proposed_police"] = frame["proposed_police"].astype(int)
    frame["transfer"] = frame["proposed_police"] - frame["current_police"]
    frame["alignment"] = _classify_gap(frame["gap"])
    frame["scenario"] = scenario

    if int(frame["proposed_police"].sum()) != total_police:
        raise ValueError(f"La asignación de {scenario} no conserva la plantilla total.")
    if int(frame["transfer"].sum()) != 0:
        raise ValueError(f"Las transferencias de {scenario} no están equilibradas.")
    return frame.sort_values("gap").reset_index(drop=True)


def _build_comparison(volume: pd.DataFrame, weighted: pd.DataFrame) -> pd.DataFrame:
    comparison = volume[
        ["municipality", "current_police", "proposed_police", "transfer"]
    ].rename(
        columns={
            "proposed_police": "volume_proposed",
            "transfer": "volume_transfer",
        }
    )
    comparison = comparison.merge(
        weighted[["municipality", "proposed_police", "transfer"]].rename(
            columns={
                "proposed_police": "weighted_proposed",
                "transfer": "weighted_transfer",
            }
        ),
        on="municipality",
        how="inner",
        validate="one_to_one",
    )
    comparison["transfer_difference"] = (
        comparison["weighted_transfer"] - comparison["volume_transfer"]
    )
    comparison["same_transfer"] = comparison["transfer_difference"].eq(0)
    comparison["sign_switch"] = (
        comparison["volume_transfer"].gt(0)
        & comparison["weighted_transfer"].lt(0)
    ) | (
        comparison["volume_transfer"].lt(0)
        & comparison["weighted_transfer"].gt(0)
    )
    significant_threshold = max(5, int(round(EXPECTED_TOTAL_POLICE * 0.0005)))
    comparison["significant_change"] = comparison["transfer_difference"].abs().ge(
        significant_threshold
    )
    return comparison.sort_values(
        "transfer_difference", key=lambda values: values.abs(), ascending=False
    ).reset_index(drop=True)


@st.cache_data(show_spinner=False)
def load_optimization_model(
    police_path: Path = POLICE_WORKBOOK_PATH,
    master_path: Path = MASTER_DATA_PATH,
) -> OptimizationModel:
    """Carga las fuentes en solo lectura, valida totales y crea ambos escenarios."""
    if not police_path.exists():
        raise FileNotFoundError(f"No existe el archivo {police_path.name}.")
    if not master_path.exists():
        raise FileNotFoundError(f"No existe el archivo {master_path.name}.")

    raw_police = pd.read_excel(police_path, sheet_name=POLICE_SHEET, engine="openpyxl")
    _require_columns(raw_police, set(POLICE_COLUMNS), POLICE_SHEET)
    police = raw_police[list(POLICE_COLUMNS)].rename(columns=POLICE_COLUMNS).copy()
    police["municipality"] = police["municipality"].astype(str).str.strip()
    for column in ("crime_count", "current_police"):
        police[column] = pd.to_numeric(police[column], errors="raise")
    for column in ("source_crime_share", "source_police_share", "source_gap"):
        police[column] = pd.to_numeric(police[column], errors="coerce")

    if len(police) != EXPECTED_MUNICIPALITIES or police["municipality"].nunique() != EXPECTED_MUNICIPALITIES:
        raise ValueError(f"{POLICE_SHEET} debe contener 37 municipios únicos.")
    if police[["municipality", "crime_count", "current_police"]].isna().any().any():
        raise ValueError(f"{POLICE_SHEET} contiene valores base vacíos.")
    if (police[["crime_count", "current_police"]].lt(0)).any().any():
        raise ValueError(f"{POLICE_SHEET} contiene conteos negativos.")
    if not np.allclose(police["current_police"], np.round(police["current_police"])):
        raise ValueError("La plantilla policial contiene valores no enteros.")
    police["current_police"] = police["current_police"].astype(int)
    if int(police["current_police"].sum()) != EXPECTED_TOTAL_POLICE:
        raise ValueError("El total de policías locales no coincide con 9.824.")

    master = pd.read_csv(master_path)
    required_master = {
        "Municipio",
        "Año",
        "Población",
        "Delitos_totales",
        "Indice_criminal_ponderado",
    }
    _require_columns(master, required_master, master_path.name)
    master = master[list(required_master)].copy()
    master["Municipio"] = master["Municipio"].astype(str).str.strip()
    for column in ("Año", "Población", "Delitos_totales", "Indice_criminal_ponderado"):
        master[column] = pd.to_numeric(master[column], errors="coerce")
    if set(master["Año"].dropna().astype(int)) != {2023, 2024, 2025}:
        raise ValueError("La tabla maestra no contiene exactamente 2023, 2024 y 2025.")
    master = master.sort_values(["Municipio", "Año"])

    if master.duplicated(["Municipio", "Año"]).any():
        raise ValueError("La tabla maestra duplica combinaciones municipio-año.")

    volume_available = master["Delitos_totales"].notna()
    weighted_available = master["Indice_criminal_ponderado"].notna()
    if not volume_available.equals(weighted_available):
        raise ValueError(
            "La elegibilidad anual no coincide entre volumen e índice ponderado."
        )
    eligible_master = master.loc[volume_available & weighted_available].copy()
    if eligible_master.empty:
        raise ValueError("La tabla maestra no contiene años municipales elegibles.")

    annual_pressure = (
        eligible_master.groupby("Municipio", as_index=False)
        .agg(
            volume_pressure=("Delitos_totales", "mean"),
            weighted_pressure=(
                "Indice_criminal_ponderado",
                "mean",
            ),
            master_crime_count=("Delitos_totales", lambda values: values.sum(min_count=1)),
            population_latest=("Población", "last"),
            eligible_years=("Año", "nunique"),
            observed_years=(
                "Año",
                lambda values: tuple(
                    int(value) for value in sorted(values.astype(int).unique())
                ),
            ),
        )
        .rename(columns={"Municipio": "municipality"})
    )
    if annual_pressure["municipality"].nunique() != EXPECTED_MUNICIPALITIES:
        raise ValueError("La tabla maestra no contiene los 37 municipios del modelo.")
    pressure_columns = ["volume_pressure", "weighted_pressure", "master_crime_count"]
    if annual_pressure[pressure_columns].isna().any().any():
        raise ValueError("La presión anual media contiene valores nulos.")
    if annual_pressure["eligible_years"].lt(1).any():
        raise ValueError("Todos los municipios deben aportar al menos un año elegible.")
    if annual_pressure["eligible_years"].gt(len({2023, 2024, 2025})).any():
        raise ValueError("Un municipio aporta más años que el periodo analizado.")

    base = police.merge(
        annual_pressure,
        on="municipality",
        how="inner",
        validate="one_to_one",
    )
    if len(base) != EXPECTED_MUNICIPALITIES:
        missing_police = sorted(
            set(police["municipality"]) - set(annual_pressure["municipality"])
        )
        missing_master = sorted(
            set(annual_pressure["municipality"]) - set(police["municipality"])
        )
        raise ValueError(
            "Los nombres municipales no coinciden exactamente entre fuentes. "
            f"Solo Excel: {missing_police}; solo tabla maestra: {missing_master}."
        )
    if not np.allclose(base["crime_count"], base["master_crime_count"]):
        raise ValueError("El volumen municipal no coincide entre el Excel y la tabla maestra.")

    recomputed_crime_share = base["crime_count"] / base["crime_count"].sum()
    recomputed_police_share = base["current_police"] / base["current_police"].sum()
    max_source_difference = float(
        pd.concat(
            [
                (recomputed_crime_share - base["source_crime_share"]).abs(),
                (recomputed_police_share - base["source_police_share"]).abs(),
                (
                    recomputed_police_share
                    - recomputed_crime_share
                    - base["source_gap"]
                ).abs(),
            ]
        ).max()
    )

    volume_scenario = _hamilton_allocation(
        base,
        "volume_pressure",
        VOLUME_SCENARIO,
    )
    weighted_scenario = _hamilton_allocation(
        base, "weighted_pressure", WEIGHTED_SCENARIO
    )
    comparison = _build_comparison(volume_scenario, weighted_scenario)

    source_redistribution = pd.read_excel(
        police_path, sheet_name=SOURCE_REDISTRIBUTION_SHEET, engine="openpyxl"
    )
    source_proposed = pd.to_numeric(source_redistribution.iloc[:, 4], errors="coerce")
    source_transfer = pd.to_numeric(source_redistribution.iloc[:, 5], errors="coerce")

    audit = OptimizationAudit(
        municipality_count=len(base),
        total_police=int(base["current_police"].sum()),
        total_volume=int(eligible_master["Delitos_totales"].sum()),
        total_weighted_pressure=float(
            eligible_master["Indice_criminal_ponderado"].sum()
        ),
        total_volume_annual_mean_pressure=float(base["volume_pressure"].sum()),
        total_weighted_annual_mean_pressure=float(base["weighted_pressure"].sum()),
        volume_proposed_total=int(volume_scenario["proposed_police"].sum()),
        weighted_proposed_total=int(weighted_scenario["proposed_police"].sum()),
        volume_transfer_total=int(volume_scenario["transfer"].sum()),
        weighted_transfer_total=int(weighted_scenario["transfer"].sum()),
        source_redistribution_total=int(source_proposed.sum()),
        source_redistribution_transfer_total=int(source_transfer.sum()),
        max_source_share_difference=max_source_difference,
        ineligible_placeholder_rows=int((~volume_available).sum()),
        eligible_years_by_municipality={
            str(row["municipality"]): int(row["eligible_years"])
            for _, row in annual_pressure.iterrows()
        },
        observed_years_by_municipality={
            str(row["municipality"]): tuple(row["observed_years"])
            for _, row in annual_pressure.iterrows()
        },
    )
    return OptimizationModel(volume_scenario, weighted_scenario, comparison, audit)


def scenario_frame(model: OptimizationModel, scenario: str) -> pd.DataFrame:
    if scenario == VOLUME_SCENARIO:
        return model.volume.copy()
    if scenario == WEIGHTED_SCENARIO:
        return model.weighted.copy()
    raise ValueError(f"Escenario no reconocido: {scenario}.")


def prepare_optimization_map(
    frame: pd.DataFrame,
    geojson_path: Path = GEOJSON_PATH,
) -> MapSource:
    """Enlaza datos dinámicos con la geometría municipal compartida."""
    return prepare_project_map_source(frame, "municipality", geojson_path)
