"""Agregaciones ejecutivas y comparables para la Home de Crime Pulse Madrid."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
import streamlit as st

from utils.crime_profile_data import QUARTERS, load_profile_workbook
from utils.data import load_master_data
from utils.metrics import CRIME_INDEX
from utils.police_optimization_data import prepare_optimization_map
from utils.predictions_data import (
    ALL_CRIME_TYPES,
    ALL_MUNICIPALITIES,
    FORECAST_QUARTERS,
    PREDICTION_KIND,
    REAL_KIND,
    load_prediction_model,
    scope_series,
)


ALL_HOME_MUNICIPALITIES = "TODOS LOS MUNICIPIOS"
ALL_HOME_CRIME_TYPES = "TODOS LOS TIPOS DE DELITO"


@dataclass(frozen=True)
class HomeModel:
    master: pd.DataFrame
    detail: pd.DataFrame
    prediction_model: object
    years: tuple[int, ...]
    municipalities: tuple[str, ...]
    crime_types: tuple[str, ...]


@dataclass(frozen=True)
class HomeSnapshot:
    year: int
    municipality: str
    crime_type: str
    territorial: pd.DataFrame
    quarterly: pd.DataFrame
    drivers: pd.DataFrame
    total_crimes: int
    municipality_count: int
    population_covered: int
    relative_metric_value: float | None
    relative_metric_label: str
    relative_metric_detail: str
    selected_rank: int | None
    selected_rank_total: int
    previous_year: int | None
    yoy_current: float | None
    yoy_previous: float | None
    yoy_change: float | None
    comparable_municipalities: int
    focus_municipality: str | None
    focus_metric: float | None
    forecast_real_2025: float | None
    forecast_predicted_2026: float | None
    forecast_change: float | None


@dataclass(frozen=True)
class HomeSignal:
    signal_type: str
    title: str
    text: str
    module: str


@st.cache_data(show_spinner=False)
def load_home_model() -> HomeModel:
    """Carga una vez las fuentes ya validadas por sus módulos propietarios."""
    master = load_master_data().copy()
    detail, _weights, audit = load_profile_workbook()
    prediction_model = load_prediction_model()

    if audit.duplicate_count:
        raise ValueError("El detalle delictivo contiene duplicados no compatibles con Home.")
    if audit.crime_type_count != 16 or audit.municipality_count != 37:
        raise ValueError("Home requiere 37 municipios y 16 tipologías delictivas.")

    annual_detail = (
        detail.groupby(["year", "municipality"], as_index=False, observed=True)
        .agg(detail_count=("count", "sum"))
    )
    reconciliation = master.merge(
        annual_detail,
        left_on=["Año", "Municipio"],
        right_on=["year", "municipality"],
        how="inner",
        validate="one_to_one",
    )
    eligible = reconciliation[CRIME_INDEX].notna()
    max_count_difference = float(
        (
            reconciliation.loc[eligible, "Delitos_totales"]
            - reconciliation.loc[eligible, "detail_count"]
        )
        .abs()
        .max()
    )
    if max_count_difference > 1e-9:
        raise ValueError("Los conteos del detalle no reconcilian con tabla_maestra.csv.")

    historic_types = set(detail["crime_type"].astype(str))
    predicted_types = set(prediction_model.predictions["tipo de crimen"].astype(str))
    if historic_types != predicted_types:
        raise ValueError("Las tipologías históricas y predictivas no coinciden exactamente.")

    return HomeModel(
        master=master,
        detail=detail,
        prediction_model=prediction_model,
        years=tuple(sorted(master["Año"].dropna().astype(int).unique(), reverse=True)),
        municipalities=tuple(sorted(master["Municipio"].dropna().astype(str).unique())),
        crime_types=tuple(sorted(historic_types)),
    )


def _territorial_frame(
    model: HomeModel,
    year: int,
    crime_type: str,
) -> tuple[pd.DataFrame, str, str]:
    base = model.master.loc[
        model.master["Año"].eq(int(year)),
        [
            "Municipio",
            "Población",
            "Delitos_totales",
            "Indice_criminal_ponderado",
            CRIME_INDEX,
            "Ranking_criminal_anual",
        ],
    ].rename(
        columns={
            "Municipio": "municipality",
            "Población": "population",
            "Delitos_totales": "crime_count",
            "Indice_criminal_ponderado": "weighted_count",
            CRIME_INDEX: "relative_metric",
            "Ranking_criminal_anual": "source_rank",
        }
    )
    if len(base) != 37 or base["municipality"].nunique() != 37:
        raise ValueError(f"La vista territorial de {year} no contiene 37 municipios.")

    if crime_type == ALL_HOME_CRIME_TYPES:
        label = "Índice criminal ponderado regional / 10.000 hab."
        detail = "Métrica oficial ponderada por gravedad y normalizada por población"
    else:
        type_scope = model.detail.loc[
            model.detail["year"].eq(int(year))
            & model.detail["crime_type"].eq(crime_type)
        ]
        type_summary = (
            type_scope.groupby("municipality", as_index=False, observed=True)
            .agg(crime_count_type=("count", "sum"), weighted_count_type=("weighted_count", "sum"))
        )
        base = base.drop(columns=["crime_count", "weighted_count", "relative_metric", "source_rank"])
        base = base.merge(type_summary, on="municipality", how="left", validate="one_to_one")
        base = base.rename(
            columns={
                "crime_count_type": "crime_count",
                "weighted_count_type": "weighted_count",
            }
        )
        base["relative_metric"] = (
            base["weighted_count"] / base["population"] * 10_000
        )
        base["source_rank"] = pd.NA
        label = "Tasa ponderada de la tipología / 10.000 hab."
        detail = "Conteo de la tipología × peso de gravedad, normalizado por población"

    base["eligible"] = base["relative_metric"].notna() & base["crime_count"].notna()
    eligible = base.loc[base["eligible"]].copy()
    eligible["computed_rank"] = (
        eligible["relative_metric"].rank(method="min", ascending=False).astype(int)
    )
    base = base.merge(
        eligible[["municipality", "computed_rank"]],
        on="municipality",
        how="left",
        validate="one_to_one",
    )
    base["computed_rank"] = base["computed_rank"].astype("Int64")
    return base.sort_values("municipality").reset_index(drop=True), label, detail


def _scope_rows(territorial: pd.DataFrame, municipality: str) -> pd.DataFrame:
    eligible = territorial.loc[territorial["eligible"]]
    if municipality == ALL_HOME_MUNICIPALITIES:
        return eligible.copy()
    return eligible.loc[eligible["municipality"].eq(municipality)].copy()


def _interannual_comparison(
    model: HomeModel,
    year: int,
    municipality: str,
    current: pd.DataFrame,
    previous: pd.DataFrame | None,
) -> tuple[int | None, float | None, float | None, float | None, int]:
    previous_year = year - 1 if year - 1 in model.years else None
    if previous_year is None or previous is None:
        return None, None, None, None, 0
    current_eligible = current.loc[current["eligible"], ["municipality", "crime_count"]]
    previous_eligible = previous.loc[previous["eligible"], ["municipality", "crime_count"]]

    if municipality != ALL_HOME_MUNICIPALITIES:
        current_eligible = current_eligible.loc[
            current_eligible["municipality"].eq(municipality)
        ]
        previous_eligible = previous_eligible.loc[
            previous_eligible["municipality"].eq(municipality)
        ]

    comparable = current_eligible.merge(
        previous_eligible,
        on="municipality",
        suffixes=("_current", "_previous"),
        how="inner",
        validate="one_to_one",
    )
    if comparable.empty:
        return previous_year, None, None, None, 0
    current_value = float(comparable["crime_count_current"].sum())
    previous_value = float(comparable["crime_count_previous"].sum())
    change = (
        (current_value - previous_value) / previous_value * 100
        if previous_value > 0
        else None
    )
    return previous_year, current_value, previous_value, change, len(comparable)


def _quarterly_series(
    model: HomeModel,
    municipality: str,
    crime_type: str,
) -> pd.DataFrame:
    scope = model.detail
    if municipality != ALL_HOME_MUNICIPALITIES:
        scope = scope.loc[scope["municipality"].eq(municipality)]
    if crime_type != ALL_HOME_CRIME_TYPES:
        scope = scope.loc[scope["crime_type"].eq(crime_type)]
    grouped = (
        scope.groupby(["year", "quarter", "period_index", "period_label"], as_index=False, observed=True)["count"]
        .sum()
        .rename(columns={"count": "value"})
    )
    skeleton = pd.DataFrame(
        [
            {
                "year": year,
                "quarter": quarter,
                "period_index": year * 10 + index,
                "period_label": f"{year} {quarter}",
            }
            for year in sorted(model.years)
            for index, quarter in enumerate(QUARTERS, start=1)
        ]
    )
    return skeleton.merge(
        grouped,
        on=["year", "quarter", "period_index", "period_label"],
        how="left",
        validate="one_to_one",
    ).sort_values("period_index").reset_index(drop=True)


def _change_drivers(
    model: HomeModel,
    year: int,
    municipality: str,
    crime_type: str,
    current_territorial: pd.DataFrame,
    previous_territorial: pd.DataFrame | None,
) -> pd.DataFrame:
    if (
        crime_type != ALL_HOME_CRIME_TYPES
        or year - 1 not in model.years
        or previous_territorial is None
    ):
        return pd.DataFrame()
    if municipality == ALL_HOME_MUNICIPALITIES:
        comparable_names = set(
            current_territorial.loc[current_territorial["eligible"], "municipality"]
        ) & set(
            previous_territorial.loc[previous_territorial["eligible"], "municipality"]
        )
    else:
        current_valid = current_territorial.loc[
            current_territorial["eligible"]
            & current_territorial["municipality"].eq(municipality)
        ]
        previous_valid = previous_territorial.loc[
            previous_territorial["eligible"]
            & previous_territorial["municipality"].eq(municipality)
        ]
        comparable_names = {municipality} if not current_valid.empty and not previous_valid.empty else set()
    if not comparable_names:
        return pd.DataFrame()

    scope = model.detail.loc[
        model.detail["municipality"].isin(comparable_names)
        & model.detail["year"].isin([year - 1, year])
    ]
    annual = (
        scope.groupby(["year", "crime_id", "crime_type"], as_index=False, observed=True)["count"]
        .sum()
        .pivot(index=["crime_id", "crime_type"], columns="year", values="count")
        .reset_index()
    )
    for required_year in (year - 1, year):
        if required_year not in annual:
            annual[required_year] = np.nan
    annual = annual.dropna(subset=[year - 1, year]).copy()
    annual = annual.rename(columns={year - 1: "previous_count", year: "current_count"})
    annual["change_absolute"] = annual["current_count"] - annual["previous_count"]
    annual["combined_volume"] = annual["current_count"] + annual["previous_count"]
    minimum_volume = max(5.0, float(annual["current_count"].sum()) * .0001)
    annual = annual.loc[annual["combined_volume"].ge(minimum_volume)]
    positive = annual.loc[annual["change_absolute"].gt(0)].nlargest(3, "change_absolute").copy()
    negative = annual.loc[annual["change_absolute"].lt(0)].nsmallest(3, "change_absolute").copy()
    positive["direction"] = "Contribución positiva"
    negative["direction"] = "Contribución negativa"
    return pd.concat([negative, positive], ignore_index=True).sort_values(
        "change_absolute"
    ).reset_index(drop=True)


def _forecast_signal(
    model: HomeModel,
    municipality: str,
    crime_type: str,
) -> tuple[float | None, float | None, float | None]:
    prediction_municipality = (
        ALL_MUNICIPALITIES
        if municipality == ALL_HOME_MUNICIPALITIES
        else municipality
    )
    prediction_type = ALL_CRIME_TYPES if crime_type == ALL_HOME_CRIME_TYPES else crime_type
    series = scope_series(
        model.prediction_model,
        prediction_municipality,
        prediction_type,
    )
    real = float(
        series.loc[
            series["tipo_dato"].eq(REAL_KIND)
            & series["año"].eq(2025)
            & series["trimestre"].isin(FORECAST_QUARTERS),
            "valor",
        ].sum()
    )
    predicted = float(
        series.loc[
            series["tipo_dato"].eq(PREDICTION_KIND)
            & series["año"].eq(2026)
            & series["trimestre"].isin(FORECAST_QUARTERS),
            "valor",
        ].sum()
    )
    change = (predicted - real) / real * 100 if real > 0 else None
    return real, predicted, change


def build_home_snapshot(
    model: HomeModel,
    year: int,
    municipality: str,
    crime_type: str,
) -> HomeSnapshot:
    year = int(year)
    territorial, metric_label, metric_detail = _territorial_frame(
        model, year, crime_type
    )
    previous_territorial = None
    if year - 1 in model.years:
        previous_territorial, _previous_label, _previous_detail = _territorial_frame(
            model, year - 1, crime_type
        )
    if (
        crime_type == ALL_HOME_CRIME_TYPES
        and municipality != ALL_HOME_MUNICIPALITIES
    ):
        metric_label = "Índice criminal ponderado municipal / 10.000 hab."
    scope = _scope_rows(territorial, municipality)
    total_crimes = int(scope["crime_count"].sum()) if not scope.empty else 0
    population = int(scope["population"].sum()) if not scope.empty else 0
    weighted_total = float(scope["weighted_count"].sum()) if not scope.empty else 0.0
    relative_value = (
        weighted_total / population * 10_000 if population > 0 else None
    )

    eligible = territorial.loc[territorial["eligible"]].copy()
    selected_rank = None
    if municipality != ALL_HOME_MUNICIPALITIES:
        selected = eligible.loc[eligible["municipality"].eq(municipality)]
        if not selected.empty:
            selected_rank = int(selected.iloc[0]["computed_rank"])

    previous_year, yoy_current, yoy_previous, yoy_change, comparable_count = (
        _interannual_comparison(
            model,
            year,
            municipality,
            territorial,
            previous_territorial,
        )
    )
    focus_municipality = None
    focus_metric = None
    if not eligible.empty:
        focus = eligible.loc[eligible["relative_metric"].idxmax()]
        focus_municipality = str(focus["municipality"])
        focus_metric = float(focus["relative_metric"])

    forecast_real, forecast_predicted, forecast_change = _forecast_signal(
        model, municipality, crime_type
    )
    return HomeSnapshot(
        year=year,
        municipality=municipality,
        crime_type=crime_type,
        territorial=territorial,
        quarterly=_quarterly_series(model, municipality, crime_type),
        drivers=_change_drivers(
            model,
            year,
            municipality,
            crime_type,
            territorial,
            previous_territorial,
        ),
        total_crimes=total_crimes,
        municipality_count=len(scope),
        population_covered=population,
        relative_metric_value=relative_value,
        relative_metric_label=metric_label,
        relative_metric_detail=metric_detail,
        selected_rank=selected_rank,
        selected_rank_total=len(eligible),
        previous_year=previous_year,
        yoy_current=yoy_current,
        yoy_previous=yoy_previous,
        yoy_change=yoy_change,
        comparable_municipalities=comparable_count,
        focus_municipality=focus_municipality,
        focus_metric=focus_metric,
        forecast_real_2025=forecast_real,
        forecast_predicted_2026=forecast_predicted,
        forecast_change=forecast_change,
    )


def build_home_signals(snapshot: HomeSnapshot) -> tuple[HomeSignal, ...]:
    signals: list[HomeSignal] = []
    if snapshot.focus_municipality and snapshot.focus_metric is not None:
        if snapshot.municipality == ALL_HOME_MUNICIPALITIES:
            focus_text = (
                f"{snapshot.focus_municipality} presenta el mayor valor territorial "
                f"del filtro activo ({snapshot.focus_metric:,.1f} por 10.000 hab.)."
            )
        elif snapshot.selected_rank is not None:
            focus_text = (
                f"{snapshot.municipality} ocupa la posición #{snapshot.selected_rank} de "
                f"{snapshot.selected_rank_total}; el máximo regional corresponde a "
                f"{snapshot.focus_municipality}."
            )
        else:
            focus_text = f"{snapshot.municipality} no pertenece al universo criminal válido de {snapshot.year}."
    else:
        focus_text = "No existe un foco territorial comparable para el filtro activo."
    signals.append(HomeSignal("FOCO TERRITORIAL", "Dónde se concentra", focus_text, "MAPA CRIMINAL"))

    if snapshot.crime_type == ALL_HOME_CRIME_TYPES and not snapshot.drivers.empty:
        driver = snapshot.drivers.loc[snapshot.drivers["change_absolute"].abs().idxmax()]
        direction = "aumenta" if driver["change_absolute"] > 0 else "disminuye"
        trend_text = (
            f"{driver['crime_type']} {direction} en {abs(int(driver['change_absolute'])):,} casos "
            "dentro de la cohorte comparable."
        )
    elif snapshot.crime_type != ALL_HOME_CRIME_TYPES and snapshot.yoy_change is not None:
        direction = "aumenta" if snapshot.yoy_change >= 0 else "disminuye"
        trend_text = (
            f"{snapshot.crime_type} {direction} un {abs(snapshot.yoy_change):.1f}% frente a "
            f"{snapshot.previous_year} en el mismo ámbito comparable."
        )
    else:
        trend_text = "No existe histórico comparable suficiente para identificar una tendencia delictiva."
    signals.append(HomeSignal("TENDENCIA DELICTIVA", "Qué explica el cambio", trend_text, "PERFIL DELICTIVO"))

    if snapshot.yoy_change is None:
        yoy_text = "Sin histórico comparable para el ámbito y periodo seleccionados."
    else:
        direction = "sube" if snapshot.yoy_change >= 0 else "baja"
        yoy_text = (
            f"El volumen agregado {direction} un {abs(snapshot.yoy_change):.1f}% frente a "
            f"{snapshot.previous_year}, usando {snapshot.comparable_municipalities} municipios comunes."
        )
    signals.append(HomeSignal("CAMBIO INTERANUAL", "Cómo evoluciona", yoy_text, "PERFIL DELICTIVO"))

    if snapshot.forecast_change is None:
        forecast_text = "No existe una base Q1–Q3 2025 compatible con el filtro activo."
    else:
        direction = "por encima" if snapshot.forecast_change >= 0 else "por debajo"
        forecast_text = (
            f"El forecast Q1–Q3 2026 se sitúa un {abs(snapshot.forecast_change):.1f}% "
            f"{direction} de Q1–Q3 2025 para este ámbito."
        )
    signals.append(HomeSignal("PERSPECTIVA 2026", "Qué anticipa el forecast", forecast_text, "PREDICCIONES"))
    return tuple(signals[:4])


def prepare_home_map(snapshot: HomeSnapshot):
    """Reutiliza el enlace cartográfico exacto empleado por Optimización."""
    return prepare_optimization_map(snapshot.territorial)
