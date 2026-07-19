"""Dashboard aggregations based exclusively on existing source fields."""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd


CRIME_INDEX = "Indice_criminal_ponderado_por_10000_hab"


@dataclass(frozen=True)
class DashboardSnapshot:
    """All display values required to render one annual dashboard state."""

    year: int
    year_df: pd.DataFrame
    municipality_count: int
    population: int
    total_crimes: int
    top_municipality: str
    top_index: float
    top_rank: int
    top_crime_share: float
    median_index: float
    difference_from_median: float
    previous_year: int | None
    previous_index: float | None
    annual_change: float | None


def build_dashboard_snapshot(frame: pd.DataFrame, year: int) -> DashboardSnapshot:
    """Build annual display metrics without changing the underlying methodology."""
    year = int(year)
    year_df_all = frame.loc[frame["Año"].eq(year)].copy()
    year_df = year_df_all.dropna(subset=[CRIME_INDEX]).copy()
    valid_index = year_df

    if year_df_all.empty:
        raise ValueError(f"No existen datos para {year}.")
    if valid_index.empty:
        raise ValueError(f"No existen valores del índice criminal para {year}.")

    top_row = valid_index.loc[valid_index[CRIME_INDEX].idxmax()]
    total_crimes_value = year_df["Delitos_totales"].sum(min_count=1)
    total_crimes = int(total_crimes_value) if pd.notna(total_crimes_value) else 0
    top_crimes = top_row["Delitos_totales"]
    top_crime_share = (
        float(top_crimes) / total_crimes * 100
        if total_crimes > 0 and pd.notna(top_crimes)
        else 0.0
    )

    median_index = float(valid_index[CRIME_INDEX].median())
    top_index = float(top_row[CRIME_INDEX])
    difference_from_median = (
        (top_index - median_index) / median_index * 100 if median_index else 0.0
    )

    previous_year = year - 1 if (year - 1) in frame["Año"].values else None
    previous_index = None
    annual_change = None
    if previous_year is not None:
        previous_match = frame.loc[
            frame["Año"].eq(previous_year)
            & frame["Municipio"].eq(top_row["Municipio"]),
            CRIME_INDEX,
        ].dropna()
        if not previous_match.empty:
            previous_index = float(previous_match.iloc[0])
            if previous_index:
                annual_change = (top_index - previous_index) / previous_index * 100

    return DashboardSnapshot(
        year=year,
        year_df=year_df,
        municipality_count=int(year_df["Municipio"].nunique()),
        population=int(year_df["Población"].sum()),
        total_crimes=total_crimes,
        top_municipality=str(top_row["Municipio"]),
        top_index=top_index,
        top_rank=int(top_row["Ranking_criminal_anual"]),
        top_crime_share=top_crime_share,
        median_index=median_index,
        difference_from_median=difference_from_median,
        previous_year=previous_year,
        previous_index=previous_index,
        annual_change=annual_change,
    )
