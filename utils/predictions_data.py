"""Carga y agregaciones verificadas para la página Predicciones 2026."""

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
PREDICTIONS_PATH = PROJECT_ROOT / "data" / "predicciones_criminalidad_2026.csv"
HISTORY_FORECAST_PATH = PROJECT_ROOT / "data" / "crime_lab_historico_predicciones.csv"

PREDICTION_KIND = "Predicción"
REAL_KIND = "Real"
ALL_MUNICIPALITIES = "TODOS LOS MUNICIPIOS"
ALL_CRIME_TYPES = "TODOS LOS TIPOS DE CRIMEN"
ALL_CRIME_GROUPS = "TODAS LAS AGRUPACIONES"
TYPE_LEVEL = "TIPOS DE DELITO"
GROUP_LEVEL = "COMPOSICIÓN POR GRUPOS"

# Correspondencia exacta validada contra DIM_Peso_Crimen del libro maestro.
CRIME_TYPE_TO_GROUP = {
    "Homicidios dolosos y asesinatos consumados": "Violencia extrema",
    "Homicidios dolosos y asesinatos en grado tentativa": "Violencia extrema",
    "Secuestro": "Violencia extrema",
    "Delitos graves y menos graves de lesiones y riña tumultuaria": "Violencia física",
    "Delitos contra la libertad sexual": "Delitos sexuales",
    "Agresión sexual con penetración": "Delitos sexuales",
    "Resto de delitos contra la libertad sexual": "Delitos sexuales",
    "Robos con violencia e intimidación": "Delitos patrimoniales violentos",
    "Robos con fuerza en domicilios, establecimientos y otras instalaciones": "Delitos contra el patrimonio",
    "Robos con fuerza en domicilios": "Delitos contra el patrimonio",
    "Hurtos": "Delitos contra el patrimonio",
    "Sustracciones de vehículos": "Delitos contra el patrimonio",
    "Tráfico de drogas": "Drogas",
    "Resto de criminalidad convencional": "Otros delitos convencionales",
    "Estafas informáticas": "Ciberdelincuencia",
    "Otros ciberdelitos": "Ciberdelincuencia",
}
CRIME_GROUPS = (
    "Violencia extrema",
    "Violencia física",
    "Delitos sexuales",
    "Delitos patrimoniales violentos",
    "Delitos contra el patrimonio",
    "Drogas",
    "Otros delitos convencionales",
    "Ciberdelincuencia",
)
HYBRID_MODEL = "MODELO HÍBRIDO"
MIXED_CONFIDENCE = "Mixta"
STANDARD_CONFIDENCE = "Estándar"
HIGH_UNCERTAINTY = "Alta incertidumbre"
GRADIENT_MODEL = "Gradient Boosting"
MOVING_MEDIAN_MODEL = "Mediana móvil 4"
FORECAST_QUARTERS = ("Q1", "Q2", "Q3")
QUARTER_ORDER = {"Q1": 1, "Q2": 2, "Q3": 3, "Q4": 4}

RARE_CRIME_TYPES = {
    "Secuestro",
    "Homicidios dolosos y asesinatos consumados",
    "Homicidios dolosos y asesinatos en grado tentativa",
}

# Resultados de validación facilitados junto con el modelo predictivo.
WAPE_BY_HORIZON = {"Q1": 8.51, "Q2": 8.09, "Q3": 11.30, "Q4": 11.55}

PREDICTION_COLUMNS = {
    "municipio",
    "tipo de crimen",
    "año",
    "trimestre",
    "num_trimestre",
    "conteo_predicho",
    "modelo",
    "nivel_confianza",
}
LAB_COLUMNS = {
    "municipio",
    "tipo de crimen",
    "año",
    "trimestre",
    "num_trimestre",
    "valor",
    "tipo_dato",
    "modelo",
    "nivel_confianza",
}


@dataclass(frozen=True)
class PredictionAudit:
    prediction_rows: int
    prediction_rows_by_quarter: dict[str, int]
    historical_rows: int
    municipality_count: int
    crime_type_count: int
    duplicate_count: int
    negative_prediction_count: int
    rare_prediction_rows: int
    cross_source_max_difference: float
    historical_municipalities_by_year: dict[int, int]


@dataclass(frozen=True)
class PredictionModel:
    predictions: pd.DataFrame
    lab: pd.DataFrame
    type_trends: pd.DataFrame
    emerging_threshold: float
    emerging_trend: pd.Series
    audit: PredictionAudit


@dataclass(frozen=True)
class PredictionSnapshot:
    municipality: str
    crime_type: str
    quarter: str
    predicted_count: float
    actual_2025: float
    change_absolute: float
    change_percent: float | None
    model: str
    confidence: str
    municipality_count: int
    crime_type_count: int
    includes_high_uncertainty: bool


def _require_columns(frame: pd.DataFrame, required: set[str], source: str) -> None:
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(
            f"{source} no contiene las columnas requeridas: {', '.join(sorted(missing))}."
        )


def _prepare_period_columns(frame: pd.DataFrame) -> pd.DataFrame:
    output = frame.copy()
    output["quarter_order"] = output["trimestre"].map(QUARTER_ORDER)
    if output["quarter_order"].isna().any():
        invalid = sorted(output.loc[output["quarter_order"].isna(), "trimestre"].unique())
        raise ValueError(f"Existen trimestres no reconocidos: {invalid}.")
    output["period_index"] = output["año"] * 10 + output["quarter_order"]
    output["period_label"] = output["año"].astype(str) + " " + output["trimestre"]
    return output


def _build_type_trends(lab: pd.DataFrame) -> tuple[pd.DataFrame, float]:
    real_scope = lab.loc[
        lab["tipo_dato"].eq(REAL_KIND)
        & lab["año"].eq(2025)
        & lab["trimestre"].isin(FORECAST_QUARTERS)
    ]
    prediction_scope = lab.loc[
        lab["tipo_dato"].eq(PREDICTION_KIND)
        & lab["año"].eq(2026)
        & lab["trimestre"].isin(FORECAST_QUARTERS)
    ]
    real = (
        real_scope.groupby("tipo de crimen", as_index=False)["valor"]
        .sum()
        .rename(columns={"valor": "real_2025"})
    )
    predicted = (
        prediction_scope.groupby(
            ["tipo de crimen", "modelo", "nivel_confianza"], as_index=False
        )["valor"]
        .sum()
        .rename(columns={"valor": "predicted_2026"})
    )
    trends = real.merge(predicted, on="tipo de crimen", how="inner", validate="one_to_one")
    trends["change_absolute"] = trends["predicted_2026"] - trends["real_2025"]
    trends["change_percent"] = np.where(
        trends["real_2025"].gt(0),
        trends["change_absolute"] / trends["real_2025"] * 100,
        np.nan,
    )
    # El percentil 25 elimina la cola de eventos raros; el suelo evita que un
    # conjunto futuro muy pequeño rebaje el umbral hasta valores inestables.
    threshold = max(100.0, float(trends["real_2025"].quantile(.25)))
    trends["emerging_eligible"] = trends["real_2025"].ge(threshold)
    return trends.sort_values("change_percent", ascending=False).reset_index(drop=True), threshold


@st.cache_data(show_spinner=False)
def load_prediction_model(
    predictions_path: Path = PREDICTIONS_PATH,
    lab_path: Path = HISTORY_FORECAST_PATH,
) -> PredictionModel:
    """Carga ambos CSV, los reconcilia y deriva las comparaciones observadas."""
    if not predictions_path.exists():
        raise FileNotFoundError(f"No existe el archivo {predictions_path.name}.")
    if not lab_path.exists():
        raise FileNotFoundError(f"No existe el archivo {lab_path.name}.")

    predictions = pd.read_csv(predictions_path)
    lab = pd.read_csv(lab_path)
    _require_columns(predictions, PREDICTION_COLUMNS, predictions_path.name)
    _require_columns(lab, LAB_COLUMNS, lab_path.name)
    predictions = predictions[list(PREDICTION_COLUMNS)].copy()
    lab = lab[list(LAB_COLUMNS)].copy()

    text_columns = ("municipio", "tipo de crimen", "trimestre", "modelo", "nivel_confianza")
    for column in text_columns:
        predictions[column] = predictions[column].astype(str).str.strip()
        lab[column] = lab[column].astype(str).str.strip()
    lab["tipo_dato"] = lab["tipo_dato"].astype(str).str.strip()

    known_types = set(CRIME_TYPE_TO_GROUP)
    prediction_types = set(predictions["tipo de crimen"].unique())
    lab_types = set(lab["tipo de crimen"].unique())
    unmapped = sorted((prediction_types | lab_types).difference(known_types))
    if unmapped:
        raise ValueError(
            "Existen tipologías predictivas sin agrupación en DIM_Peso_Crimen: "
            + ", ".join(unmapped)
        )
    predictions["Grupo"] = predictions["tipo de crimen"].map(CRIME_TYPE_TO_GROUP)
    lab["Grupo"] = lab["tipo de crimen"].map(CRIME_TYPE_TO_GROUP)

    for column in ("año", "num_trimestre"):
        predictions[column] = pd.to_numeric(predictions[column], errors="raise").astype(int)
        lab[column] = pd.to_numeric(lab[column], errors="raise").astype(int)
    predictions["conteo_predicho"] = pd.to_numeric(
        predictions["conteo_predicho"], errors="raise"
    )
    lab["valor"] = pd.to_numeric(lab["valor"], errors="raise")

    if predictions.isna().any().any() or lab.isna().any().any():
        raise ValueError("Las fuentes predictivas contienen valores vacíos.")
    prediction_grain = ["municipio", "tipo de crimen", "año", "trimestre"]
    prediction_duplicates = int(predictions.duplicated(prediction_grain).sum())
    lab_duplicates = int(
        lab.duplicated([*prediction_grain, "tipo_dato"]).sum()
    )
    if prediction_duplicates or lab_duplicates:
        raise ValueError("Las fuentes predictivas contienen duplicados en su granularidad.")
    if len(predictions) != 1_776:
        raise ValueError("El forecast debe contener exactamente 1.776 predicciones.")
    if predictions["municipio"].nunique() != 37 or predictions["tipo de crimen"].nunique() != 16:
        raise ValueError("El forecast debe cubrir 37 municipios y 16 tipologías.")
    if set(predictions["año"]) != {2026} or set(predictions["trimestre"]) != set(FORECAST_QUARTERS):
        raise ValueError("El forecast debe limitarse a Q1, Q2 y Q3 de 2026.")
    rows_by_quarter = {
        quarter: int(count)
        for quarter, count in predictions.groupby("trimestre").size().items()
    }
    if rows_by_quarter != {"Q1": 592, "Q2": 592, "Q3": 592}:
        raise ValueError("Cada trimestre predictivo debe contener 592 observaciones.")
    if (predictions["conteo_predicho"] < 0).any():
        raise ValueError("El forecast contiene predicciones negativas.")

    rare_rows = predictions.loc[predictions["nivel_confianza"].eq(HIGH_UNCERTAINTY)]
    if set(rare_rows["tipo de crimen"]) != RARE_CRIME_TYPES:
        raise ValueError("Las categorías de alta incertidumbre no coinciden con la metodología.")
    if not rare_rows["modelo"].eq(MOVING_MEDIAN_MODEL).all():
        raise ValueError("Las categorías raras no utilizan Mediana móvil 4.")
    standard_rows = predictions.loc[~predictions["tipo de crimen"].isin(RARE_CRIME_TYPES)]
    if not (
        standard_rows["modelo"].eq(GRADIENT_MODEL).all()
        and standard_rows["nivel_confianza"].eq(STANDARD_CONFIDENCE).all()
    ):
        raise ValueError("Las categorías estándar no utilizan Gradient Boosting.")

    lab_predictions = lab.loc[lab["tipo_dato"].eq(PREDICTION_KIND)].copy()
    reconciliation = predictions.merge(
        lab_predictions,
        on=["municipio", "tipo de crimen", "año", "trimestre", "num_trimestre"],
        how="outer",
        suffixes=("_forecast", "_lab"),
        indicator=True,
        validate="one_to_one",
    )
    if not reconciliation["_merge"].eq("both").all():
        raise ValueError("Los dos CSV no contienen las mismas claves predictivas.")
    max_difference = float(
        (reconciliation["conteo_predicho"] - reconciliation["valor"]).abs().max()
    )
    if max_difference > 1e-9:
        raise ValueError("Los valores predictivos difieren entre los dos CSV.")
    if not (
        reconciliation["modelo_forecast"].eq(reconciliation["modelo_lab"]).all()
        and reconciliation["nivel_confianza_forecast"].eq(
            reconciliation["nivel_confianza_lab"]
        ).all()
    ):
        raise ValueError("Modelo o confianza difieren entre los dos CSV.")

    predictions = _prepare_period_columns(predictions)
    lab = _prepare_period_columns(lab)
    type_trends, threshold = _build_type_trends(lab)
    eligible = type_trends.loc[type_trends["emerging_eligible"]]
    if eligible.empty:
        raise ValueError("Ninguna tipología supera el umbral de tendencia emergente.")
    emerging = eligible.sort_values("change_percent", ascending=False).iloc[0].copy()

    historical = lab.loc[lab["tipo_dato"].eq(REAL_KIND)]
    audit = PredictionAudit(
        prediction_rows=len(predictions),
        prediction_rows_by_quarter=rows_by_quarter,
        historical_rows=len(historical),
        municipality_count=int(predictions["municipio"].nunique()),
        crime_type_count=int(predictions["tipo de crimen"].nunique()),
        duplicate_count=prediction_duplicates + lab_duplicates,
        negative_prediction_count=int(predictions["conteo_predicho"].lt(0).sum()),
        rare_prediction_rows=len(rare_rows),
        cross_source_max_difference=max_difference,
        historical_municipalities_by_year={
            int(year): int(count)
            for year, count in historical.groupby("año")["municipio"].nunique().items()
        },
    )
    return PredictionModel(predictions, lab, type_trends, threshold, emerging, audit)



def _selected_crime_types(selection_level: str, selection: str) -> set[str] | None:
    """Devuelve las tipologías incluidas por una selección de tipo o agrupación."""
    if selection_level == TYPE_LEVEL:
        return None if selection == ALL_CRIME_TYPES else {selection}
    if selection_level == GROUP_LEVEL:
        if selection == ALL_CRIME_GROUPS:
            return None
        return {
            crime_type
            for crime_type, group in CRIME_TYPE_TO_GROUP.items()
            if group == selection
        }
    raise ValueError(f"Nivel de análisis no reconocido: {selection_level}.")


def _selection_metadata(
    model: PredictionModel,
    selection_level: str,
    selection: str,
) -> tuple[str, str, int, bool]:
    """Resume modelo, confianza y cobertura de tipologías de la selección."""
    selected_types = _selected_crime_types(selection_level, selection)
    metadata = model.predictions
    if selected_types is not None:
        metadata = metadata.loc[metadata["tipo de crimen"].isin(selected_types)]
    unique_types = sorted(metadata["tipo de crimen"].unique())
    unique_models = sorted(metadata["modelo"].unique())
    unique_confidence = sorted(metadata["nivel_confianza"].unique())

    prediction_model = (
        unique_models[0] if len(unique_models) == 1 else HYBRID_MODEL
    )
    prediction_confidence = (
        unique_confidence[0] if len(unique_confidence) == 1 else MIXED_CONFIDENCE
    )
    includes_high_uncertainty = bool(
        set(unique_types).intersection(RARE_CRIME_TYPES)
    )
    return (
        prediction_model,
        prediction_confidence,
        len(unique_types),
        includes_high_uncertainty,
    )


def _filter_selection(
    frame: pd.DataFrame,
    selection_level: str,
    selection: str,
) -> pd.DataFrame:
    selected_types = _selected_crime_types(selection_level, selection)
    if selected_types is None:
        return frame
    return frame.loc[frame["tipo de crimen"].isin(selected_types)]


def prediction_snapshot(
    model: PredictionModel,
    municipality: str,
    selection: str,
    quarter: str,
    selection_level: str = TYPE_LEVEL,
) -> PredictionSnapshot:
    series = scope_series(model, municipality, selection, selection_level)
    prediction = series.loc[
        series["tipo_dato"].eq(PREDICTION_KIND)
        & series["año"].eq(2026)
        & series["trimestre"].eq(quarter)
    ]
    actual = series.loc[
        series["tipo_dato"].eq(REAL_KIND)
        & series["año"].eq(2025)
        & series["trimestre"].eq(quarter)
    ]
    if len(prediction) != 1 or len(actual) != 1:
        raise ValueError("El ámbito seleccionado no tiene una comparación 2025–2026 única.")
    prediction_row = prediction.iloc[0]
    actual_value = float(actual.iloc[0]["valor"])
    predicted_value = float(prediction_row["valor"])
    change_absolute = predicted_value - actual_value
    change_percent = (
        change_absolute / actual_value * 100 if actual_value > 0 else None
    )
    model_name, confidence, type_count, includes_high_uncertainty = _selection_metadata(
        model,
        selection_level,
        selection,
    )
    return PredictionSnapshot(
        municipality=municipality,
        crime_type=selection,
        quarter=quarter,
        predicted_count=predicted_value,
        actual_2025=actual_value,
        change_absolute=change_absolute,
        change_percent=change_percent,
        model=model_name,
        confidence=confidence,
        municipality_count=(
            model.audit.municipality_count
            if municipality == ALL_MUNICIPALITIES
            else 1
        ),
        crime_type_count=type_count,
        includes_high_uncertainty=includes_high_uncertainty,
    )


def scope_series(
    model: PredictionModel,
    municipality: str,
    selection: str,
    selection_level: str = TYPE_LEVEL,
) -> pd.DataFrame:
    """Agrega histórico y forecast por municipio y por tipo o grupo delictivo."""
    scope = model.lab
    if municipality != ALL_MUNICIPALITIES:
        scope = scope.loc[scope["municipio"].eq(municipality)]
    scope = _filter_selection(scope, selection_level, selection)
    if scope.empty:
        raise ValueError("El ámbito predictivo seleccionado no contiene observaciones.")

    period_columns = [
        "año",
        "trimestre",
        "num_trimestre",
        "quarter_order",
        "period_index",
        "period_label",
        "tipo_dato",
    ]
    output = scope.groupby(period_columns, as_index=False, observed=True)["valor"].sum()
    if output.duplicated(["año", "trimestre", "tipo_dato"]).any():
        raise ValueError("La agregación predictiva ha generado periodos duplicados.")

    prediction_model, prediction_confidence, _, _ = _selection_metadata(
        model,
        selection_level,
        selection,
    )
    predicted_mask = output["tipo_dato"].eq(PREDICTION_KIND)
    output["modelo"] = "Dato observado"
    output["nivel_confianza"] = REAL_KIND
    output.loc[predicted_mask, "modelo"] = prediction_model
    output.loc[predicted_mask, "nivel_confianza"] = prediction_confidence
    output["municipio"] = municipality
    output["tipo de crimen"] = selection
    output["nivel_analisis"] = selection_level
    return output.sort_values(["period_index", "tipo_dato"]).reset_index(drop=True)


def individual_series(
    model: PredictionModel,
    municipality: str,
    selection: str,
    selection_level: str = TYPE_LEVEL,
) -> pd.DataFrame:
    """Serie individual o agregada según tipo de delito o agrupación."""
    return scope_series(model, municipality, selection, selection_level)


def entity_trends_for_scope(
    model: PredictionModel,
    municipality: str,
    selection_level: str = TYPE_LEVEL,
) -> pd.DataFrame:
    """Compara Q1–Q3 por tipología o agrupación en el ámbito municipal activo."""
    scope = model.lab
    if municipality != ALL_MUNICIPALITIES:
        scope = scope.loc[scope["municipio"].eq(municipality)]

    entity_column = "tipo de crimen" if selection_level == TYPE_LEVEL else "Grupo"
    actual = (
        scope.loc[
            scope["tipo_dato"].eq(REAL_KIND)
            & scope["año"].eq(2025)
            & scope["trimestre"].isin(FORECAST_QUARTERS)
        ]
        .groupby(entity_column, as_index=False)["valor"]
        .sum()
        .rename(columns={"valor": "real_2025", entity_column: "tipo de crimen"})
    )
    predicted = (
        scope.loc[
            scope["tipo_dato"].eq(PREDICTION_KIND)
            & scope["año"].eq(2026)
            & scope["trimestre"].isin(FORECAST_QUARTERS)
        ]
        .groupby(entity_column, as_index=False)["valor"]
        .sum()
        .rename(columns={"valor": "predicted_2026", entity_column: "tipo de crimen"})
    )
    output = actual.merge(predicted, on="tipo de crimen", validate="one_to_one")
    output["change_absolute"] = output["predicted_2026"] - output["real_2025"]
    output["change_percent"] = np.where(
        output["real_2025"].gt(0),
        output["change_absolute"] / output["real_2025"] * 100,
        np.nan,
    )

    if selection_level == TYPE_LEVEL:
        metadata = model.predictions[
            ["tipo de crimen", "modelo", "nivel_confianza"]
        ].drop_duplicates()
        output = output.merge(metadata, on="tipo de crimen", how="left", validate="one_to_one")
    else:
        group_metadata = model.predictions.copy()
        group_metadata["Grupo"] = group_metadata["tipo de crimen"].map(CRIME_TYPE_TO_GROUP)
        meta_rows = []
        for group, group_scope in group_metadata.groupby("Grupo"):
            models = sorted(group_scope["modelo"].unique())
            confidences = sorted(group_scope["nivel_confianza"].unique())
            meta_rows.append(
                {
                    "tipo de crimen": group,
                    "modelo": models[0] if len(models) == 1 else HYBRID_MODEL,
                    "nivel_confianza": (
                        confidences[0] if len(confidences) == 1 else MIXED_CONFIDENCE
                    ),
                }
            )
        output = output.merge(pd.DataFrame(meta_rows), on="tipo de crimen", how="left", validate="one_to_one")

    threshold = max(
        10.0 if municipality != ALL_MUNICIPALITIES else 100.0,
        float(output["real_2025"].quantile(.25)),
    )
    output["emerging_eligible"] = output["real_2025"].ge(threshold)
    return output.sort_values("change_percent", ascending=False).reset_index(drop=True)


def type_trends_for_scope(
    model: PredictionModel,
    municipality: str,
) -> pd.DataFrame:
    """Compatibilidad con la API anterior: tendencias por tipología."""
    return entity_trends_for_scope(model, municipality, TYPE_LEVEL)



def annual_comparison(series: pd.DataFrame) -> pd.DataFrame:
    actual = series.loc[
        series["tipo_dato"].eq(REAL_KIND)
        & series["año"].eq(2025)
        & series["trimestre"].isin(FORECAST_QUARTERS),
        ["trimestre", "valor"],
    ].rename(columns={"valor": "actual_2025"})
    predicted = series.loc[
        series["tipo_dato"].eq(PREDICTION_KIND)
        & series["año"].eq(2026)
        & series["trimestre"].isin(FORECAST_QUARTERS),
        ["trimestre", "valor"],
    ].rename(columns={"valor": "predicted_2026"})
    output = actual.merge(predicted, on="trimestre", validate="one_to_one")
    output["quarter_order"] = output["trimestre"].map(QUARTER_ORDER)
    output["change_absolute"] = output["predicted_2026"] - output["actual_2025"]
    output["change_percent"] = np.where(
        output["actual_2025"].gt(0),
        output["change_absolute"] / output["actual_2025"] * 100,
        np.nan,
    )
    return output.sort_values("quarter_order").reset_index(drop=True)


def territorial_summary(
    model: PredictionModel,
    selection: str,
    quarter: str,
    selection_level: str = TYPE_LEVEL,
) -> pd.DataFrame:
    predicted_scope = model.predictions.loc[
        model.predictions["trimestre"].eq(quarter)
    ]
    actual_scope = model.lab.loc[
        model.lab["tipo_dato"].eq(REAL_KIND)
        & model.lab["año"].eq(2025)
        & model.lab["trimestre"].eq(quarter)
    ]
    predicted_scope = _filter_selection(predicted_scope, selection_level, selection)
    actual_scope = _filter_selection(actual_scope, selection_level, selection)

    predicted = (
        predicted_scope.groupby("municipio", as_index=False)["conteo_predicho"]
        .sum()
        .rename(columns={"conteo_predicho": "predicted_count"})
    )
    actual = (
        actual_scope.groupby("municipio", as_index=False)["valor"]
        .sum()
        .rename(columns={"valor": "actual_2025"})
    )
    output = predicted.merge(actual, on="municipio", validate="one_to_one")

    model_name, confidence, _, _ = _selection_metadata(
        model,
        selection_level,
        selection,
    )
    output["modelo"] = model_name
    output["nivel_confianza"] = confidence
    output["change_absolute"] = output["predicted_count"] - output["actual_2025"]
    output["change_percent"] = np.where(
        output["actual_2025"].gt(0),
        output["change_absolute"] / output["actual_2025"] * 100,
        np.nan,
    )
    return output.sort_values("predicted_count", ascending=False).reset_index(drop=True)



def emerging_quarterly_comparison(model: PredictionModel) -> pd.DataFrame:
    crime_type = str(model.emerging_trend["tipo de crimen"])
    actual = model.lab.loc[
        model.lab["tipo_dato"].eq(REAL_KIND)
        & model.lab["tipo de crimen"].eq(crime_type)
        & model.lab["año"].eq(2025)
        & model.lab["trimestre"].isin(FORECAST_QUARTERS)
    ].groupby("trimestre", as_index=False)["valor"].sum().rename(
        columns={"valor": "actual_2025"}
    )
    predicted = model.lab.loc[
        model.lab["tipo_dato"].eq(PREDICTION_KIND)
        & model.lab["tipo de crimen"].eq(crime_type)
        & model.lab["trimestre"].isin(FORECAST_QUARTERS)
    ].groupby("trimestre", as_index=False)["valor"].sum().rename(
        columns={"valor": "predicted_2026"}
    )
    output = actual.merge(predicted, on="trimestre", validate="one_to_one")
    output["quarter_order"] = output["trimestre"].map(QUARTER_ORDER)
    return output.sort_values("quarter_order").reset_index(drop=True)


def prepare_prediction_map(
    territorial: pd.DataFrame,
    geojson_path: Path = GEOJSON_PATH,
) -> MapSource:
    """Enlaza los 37 resultados predictivos con el GeoJSON local por nombre exacto."""
    return prepare_project_map_source(territorial, "municipio", geojson_path)
