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

# Asociaciones descriptivas calculadas sobre TABLA_MAESTRA_INDICES del workbook
# BASE_DATOS_CRIMINALIDAD_CON_INDICES_2023_2025.
# Pearson entre Indice_Criminal y los índices territoriales agregados.
# No representan causalidad ni efectos estimados.
TERRITORIAL_ASSOCIATIONS: dict[int, tuple[tuple[str, float], ...]] = {
    2023: (
        ("Ocio", 0.4474),
        ("Socioeconómico", 0.1395),
        ("Urbano", 0.0635),
        ("Movilidad", -0.2587),
        ("Servicios", -0.2619),
    ),
    2024: (
        ("Socioeconómico", 0.2495),
        ("Ocio", 0.2317),
        ("Urbano", 0.0944),
        ("Servicios", -0.2031),
        ("Movilidad", -0.2557),
    ),
    2025: (
        ("Ocio", 0.4120),
        ("Socioeconómico", 0.2100),
        ("Urbano", 0.1155),
        ("Movilidad", -0.1543),
        ("Servicios", -0.2305),
    ),
}


def _format_decimal_es(value: float, digits: int = 1) -> str:
    formatted = f"{float(value):,.{digits}f}"
    return formatted.replace(",", "X").replace(".", ",").replace("X", ".")


def build_map_insights(
    snapshot: MapRegionalSnapshot,
    selected_municipality: str,
    year: int,
    all_municipalities_label: str,
) -> tuple[str, str, str]:
    """Construye tres lecturas territoriales descriptivas para el contexto activo."""
    if snapshot.eligible.empty:
        return (
            "No existen municipios elegibles suficientes para identificar un foco territorial.",
            "No existe una distribución municipal suficiente para cuantificar la desigualdad territorial.",
            "No hay base territorial suficiente para describir asociaciones entre índices.",
        )

    ranked = snapshot.eligible.sort_values(CRIME_INDEX, ascending=False).reset_index(drop=True)

    if (
        selected_municipality != all_municipalities_label
        and selected_municipality in set(ranked["Municipio"].astype(str))
    ):
        selected_row = ranked.loc[
            ranked["Municipio"].astype(str).eq(str(selected_municipality))
        ].iloc[0]
        selected_rank = int(
            ranked.index[
                ranked["Municipio"].astype(str).eq(str(selected_municipality))
            ][0]
        ) + 1
        focus_insight = (
            f"{selected_municipality} ocupa la posición #{selected_rank} de "
            f"{len(ranked)} municipios elegibles, con un índice criminal de "
            f"{_format_decimal_es(float(selected_row[CRIME_INDEX]))} por 10.000 hab."
        )
    elif snapshot.top_municipality is not None and snapshot.top_index is not None:
        focus_insight = (
            f"{snapshot.top_municipality} concentra la mayor presión relativa del mapa, "
            f"con un índice criminal de {_format_decimal_es(snapshot.top_index)} por 10.000 hab."
        )
    else:
        focus_insight = "No se ha podido identificar el municipio con mayor presión relativa."

    if (
        snapshot.top_index is not None
        and snapshot.municipal_median is not None
        and snapshot.municipal_median > 0
    ):
        gap_pct = (snapshot.top_index / snapshot.municipal_median - 1) * 100
        inequality_insight = (
            f"El máximo regional se sitúa un {_format_decimal_es(gap_pct)}% por encima "
            f"de la mediana municipal ({_format_decimal_es(snapshot.municipal_median)}), "
            "evidenciando un contraste territorial relevante."
        )
    else:
        inequality_insight = (
            "La distribución disponible no permite cuantificar de forma estable "
            "la distancia entre el máximo y la mediana municipal."
        )

    associations = TERRITORIAL_ASSOCIATIONS.get(int(year), ())
    positive = [item for item in associations if item[1] > 0]
    if positive:
        strongest_name, strongest_r = max(positive, key=lambda item: item[1])
        if int(year) == 2024:
            second = sorted(positive, key=lambda item: item[1], reverse=True)[1]
            profile_insight = (
                f"En {year}, el índice {strongest_name.lower()} muestra la asociación positiva "
                f"más alta con el Índice Criminal (r≈{_format_decimal_es(strongest_r, 2)}), "
                f"seguido muy de cerca por ocio (r≈{_format_decimal_es(second[1], 2)})."
            )
        else:
            profile_insight = (
                f"En {year}, el índice de {strongest_name.lower()} presenta la asociación positiva "
                f"más alta con el Índice Criminal (r≈{_format_decimal_es(strongest_r, 2)})."
            )
    else:
        profile_insight = (
            "Los índices territoriales disponibles no muestran una asociación positiva "
            "destacada con el Índice Criminal en el año seleccionado."
        )

    return focus_insight, inequality_insight, profile_insight

