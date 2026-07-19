"""Métricas regionales y municipales para Mapa Criminal."""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from utils.map_charts import RISK_ORDER, classify_relative_risk
from utils.map_data import CRIME_INDEX


WEIGHTED_COUNT = "Indice_criminal_ponderado"
POPULATION = "Población"


@dataclass(frozen=True)
class MapRegionalSnapshot:
    """Un único cálculo anual compartido por mapa y panel contextual."""

    classified: pd.DataFrame
    eligible: pd.DataFrame
    eligible_count: int
    regional_index: float | None
    municipal_median: float | None
    top_municipality: str | None
    top_index: float | None
    risk_counts: tuple[tuple[str, int], ...]


def build_map_regional_snapshot(frame: pd.DataFrame) -> MapRegionalSnapshot:
    """Clasifica una vez y resume exclusivamente municipios elegibles."""
    classified = classify_relative_risk(frame)
    eligible = classified.loc[classified["_eligible"]].copy()
    if eligible.empty:
        return MapRegionalSnapshot(
            classified=classified,
            eligible=eligible,
            eligible_count=0,
            regional_index=None,
            municipal_median=None,
            top_municipality=None,
            top_index=None,
            risk_counts=tuple((risk, 0) for risk in RISK_ORDER),
        )

    population = eligible[POPULATION].sum(min_count=1)
    weighted_count = eligible[WEIGHTED_COUNT].sum(min_count=1)
    regional_index = (
        float(weighted_count) / float(population) * 10_000
        if pd.notna(population)
        and pd.notna(weighted_count)
        and float(population) > 0
        else None
    )
    top_row = eligible.loc[eligible[CRIME_INDEX].idxmax()]
    counts = eligible["_risk_band"].astype("string").value_counts()
    return MapRegionalSnapshot(
        classified=classified,
        eligible=eligible,
        eligible_count=len(eligible),
        regional_index=regional_index,
        municipal_median=float(eligible[CRIME_INDEX].median()),
        top_municipality=str(top_row["Municipio"]),
        top_index=float(top_row[CRIME_INDEX]),
        risk_counts=tuple((risk, int(counts.get(risk, 0))) for risk in RISK_ORDER),
    )
