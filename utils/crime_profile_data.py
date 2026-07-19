"""Carga, validación y agregaciones del Perfil Delictivo.

La fuente Excel se trata como solo lectura. Los IDs se conservan como cadenas
canónicas para que 5, 5.1, 5.2, 7 y 7.1 permanezcan siempre independientes.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd
import streamlit as st


PROJECT_ROOT = Path(__file__).resolve().parents[1]
PROFILE_DATA_PATH = PROJECT_ROOT / "data" / "Criminalidad madrid 2023-2025.xlsb.xlsx"
MAIN_SHEET = "Tabla Maestra"
WEIGHT_SHEET = "Peso Crimen"

ALL_YEARS_LABEL = "Todos / 2023-2025"
ALL_MUNICIPALITIES_LABEL = "Todos los municipios"
ALL_CRIME_TYPES_LABEL = "Todos los tipos de delito"
TYPE_LEVEL = "Tipo de crimen"
GROUP_LEVEL = "Grupo"
QUARTERLY_VIEW = "TRIMESTRAL"
ANNUAL_VIEW = "ANUAL"
VOLUME_VIEW = "VOLUMEN"
IMPACT_VIEW = "IMPACTO"

MAIN_COLUMNS = {
    "año": "year",
    "trimestre": "quarter",
    "ID crimen": "crime_id",
    "tipo de crimen": "crime_type",
    "Ccodigo postal": "postal_code",
    "municipio": "municipality",
    "conteo": "count",
    "Conteo Día": "day_count",
    "Conteo Noche": "night_count",
}
WEIGHT_COLUMNS = {
    "ID crimen": "crime_id",
    "crimen": "crime_type",
    "Puntuación crimen": "weight",
    "Grupo": "group",
}
YEARS = (2023, 2024, 2025)
QUARTERS = ("Q1", "Q2", "Q3", "Q4")


@dataclass(frozen=True)
class ProfileDataAudit:
    """Controles de calidad realizados al cargar el libro."""

    row_count: int
    municipality_count: int
    crime_type_count: int
    group_count: int
    municipalities_by_year: dict[int, int]
    duplicate_count: int
    weight_match_count: int
    postal_code_missing_rows: int
    day_night_difference_rows: int
    day_night_max_abs_difference: int
    day_night_total_difference: int


@dataclass(frozen=True)
class ProfileKPIs:
    """KPIs derivados del filtro activo."""

    total_crimes: int
    total_night: int
    night_share: float
    top_type: str
    top_type_id: str
    top_type_count: int
    top_type_share: float
    top_impact_type: str
    top_impact_group: str
    top_weighted_count: float
    weighted_load: float
    growth_type: str | None
    growth_id: str | None
    growth_value: float | None
    growth_period: str


@dataclass(frozen=True)
class DrugCase:
    """Serie observada y lectura territorial del grupo Drogas."""

    label: str
    annual: pd.DataFrame
    comparable_annual: pd.DataFrame
    quarterly: pd.DataFrame
    top_volume: pd.DataFrame
    top_growth: pd.DataFrame
    change_23_24: float | None
    change_24_25: float | None
    cumulative_change: float | None
    comparable_cumulative_change: float | None
    comparable_municipality_count: int
    accelerating: bool
    strongest_sustained_growth: bool


def _canonical_crime_id(value: object) -> str:
    """Conserva IDs decimales sin convertir 5.1 o 7.1 en categorías padre."""
    if pd.isna(value):
        raise ValueError("La fuente contiene un ID de crimen vacío.")
    number = float(value)
    return str(int(number)) if number.is_integer() else format(number, "g")


def _safe_change(previous: float, current: float) -> float | None:
    if pd.isna(previous) or pd.isna(current) or previous == 0:
        return None
    return (float(current) - float(previous)) / float(previous) * 100


def _require_columns(frame: pd.DataFrame, required: set[str], sheet: str) -> None:
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(
            f"La hoja {sheet} no contiene las columnas requeridas: "
            + ", ".join(sorted(missing))
        )


@st.cache_data(show_spinner=False)
def load_profile_workbook(
    path: Path = PROFILE_DATA_PATH,
) -> tuple[pd.DataFrame, pd.DataFrame, ProfileDataAudit]:
    """Carga las dos hojas autorizadas y valida el enlace exacto por ID."""
    if not path.exists():
        raise FileNotFoundError(f"No existe el archivo {path.name}.")

    raw_main = pd.read_excel(path, sheet_name=MAIN_SHEET, engine="openpyxl")
    raw_weights = pd.read_excel(path, sheet_name=WEIGHT_SHEET, engine="openpyxl")
    _require_columns(raw_main, set(MAIN_COLUMNS), MAIN_SHEET)
    _require_columns(raw_weights, set(WEIGHT_COLUMNS), WEIGHT_SHEET)

    main = raw_main[list(MAIN_COLUMNS)].rename(columns=MAIN_COLUMNS).copy()
    weights = raw_weights[list(WEIGHT_COLUMNS)].rename(columns=WEIGHT_COLUMNS).copy()

    main["crime_id"] = main["crime_id"].map(_canonical_crime_id)
    weights["crime_id"] = weights["crime_id"].map(_canonical_crime_id)
    main["crime_type"] = main["crime_type"].astype(str).str.strip()
    weights["crime_type"] = weights["crime_type"].astype(str).str.strip()
    main["municipality"] = main["municipality"].astype(str).str.strip()
    weights["group"] = weights["group"].astype(str).str.strip()
    main["quarter"] = main["quarter"].astype(str).str.strip().str.upper()
    main["postal_code"] = main["postal_code"].astype("string")

    for column in ("year", "count", "day_count", "night_count"):
        main[column] = pd.to_numeric(main[column], errors="raise")
    weights["weight"] = pd.to_numeric(weights["weight"], errors="raise")
    main["year"] = main["year"].astype(int)

    required_main_values = [
        "year",
        "quarter",
        "crime_id",
        "crime_type",
        "municipality",
        "count",
        "day_count",
        "night_count",
    ]
    if main[required_main_values].isna().any().any():
        raise ValueError(f"La hoja {MAIN_SHEET} contiene valores vacíos requeridos.")
    if weights[list(WEIGHT_COLUMNS.values())].isna().any().any():
        raise ValueError(f"La hoja {WEIGHT_SHEET} contiene valores vacíos requeridos.")
    if weights["crime_id"].duplicated().any():
        duplicated = sorted(weights.loc[weights["crime_id"].duplicated(), "crime_id"].unique())
        raise ValueError(f"La hoja {WEIGHT_SHEET} duplica IDs: {', '.join(duplicated)}.")

    main_catalog = main[["crime_id", "crime_type"]].drop_duplicates()
    weight_catalog = weights[["crime_id", "crime_type"]].drop_duplicates()
    if main_catalog["crime_id"].duplicated().any():
        raise ValueError("Un mismo ID está asociado a varios tipos en Tabla Maestra.")

    missing_weights = sorted(set(main_catalog["crime_id"]) - set(weight_catalog["crime_id"]))
    extra_weights = sorted(set(weight_catalog["crime_id"]) - set(main_catalog["crime_id"]))
    if missing_weights or extra_weights:
        raise ValueError(
            "La relación de pesos no cubre exactamente los IDs del detalle. "
            f"Sin peso: {missing_weights or 'ninguno'}; sin detalle: {extra_weights or 'ninguno'}."
        )

    name_check = main_catalog.merge(
        weight_catalog,
        on="crime_id",
        how="outer",
        suffixes=("_main", "_weight"),
        validate="one_to_one",
    )
    mismatches = name_check.loc[
        name_check["crime_type_main"].ne(name_check["crime_type_weight"])
    ]
    if not mismatches.empty:
        ids = ", ".join(mismatches["crime_id"].astype(str))
        raise ValueError(f"Los nombres no coinciden entre hojas para los IDs: {ids}.")

    duplicate_columns = ["year", "quarter", "crime_id", "municipality"]
    duplicate_count = int(main.duplicated(duplicate_columns).sum())
    if duplicate_count:
        raise ValueError(
            f"Existen {duplicate_count} duplicados en año-trimestre-ID-municipio."
        )

    frame = main.merge(
        weights[["crime_id", "weight", "group"]],
        on="crime_id",
        how="left",
        validate="many_to_one",
    )
    frame["weighted_count"] = frame["count"] * frame["weight"]
    quarter_number = frame["quarter"].str.extract(r"(\d+)", expand=False).astype(int)
    frame["period_index"] = frame["year"] * 10 + quarter_number
    frame["period_label"] = frame["year"].astype(str) + " " + frame["quarter"]
    frame["type_label"] = frame["crime_id"] + " · " + frame["crime_type"]

    frame["day_night_difference"] = (
        frame["count"] - frame["day_count"] - frame["night_count"]
    )
    difference = frame["day_night_difference"]
    audit = ProfileDataAudit(
        row_count=len(frame),
        municipality_count=int(frame["municipality"].nunique()),
        crime_type_count=int(frame["crime_id"].nunique()),
        group_count=int(frame["group"].nunique()),
        municipalities_by_year={
            int(year): int(count)
            for year, count in frame.groupby("year")["municipality"].nunique().items()
        },
        duplicate_count=duplicate_count,
        weight_match_count=int(weights["crime_id"].nunique()),
        postal_code_missing_rows=int(frame["postal_code"].isna().sum()),
        day_night_difference_rows=int(difference.ne(0).sum()),
        day_night_max_abs_difference=int(difference.abs().max()),
        day_night_total_difference=int(difference.sum()),
    )
    return frame, weights.sort_values("crime_id").reset_index(drop=True), audit


def filter_profile_data(
    frame: pd.DataFrame,
    selected_year: int | None,
    selected_municipality: str,
    selected_crime_id: str | None = None,
) -> pd.DataFrame:
    """Aplica filtros sin agregar o reetiquetar categorías."""
    filtered = frame
    if selected_year is not None:
        filtered = filtered.loc[filtered["year"].eq(int(selected_year))]
    if selected_municipality != ALL_MUNICIPALITIES_LABEL:
        filtered = filtered.loc[filtered["municipality"].eq(selected_municipality)]
    if selected_crime_id is not None:
        filtered = filtered.loc[filtered["crime_id"].eq(str(selected_crime_id))]
    return filtered.copy()


def common_municipalities(
    frame: pd.DataFrame,
    years: tuple[int, ...] = YEARS,
) -> tuple[str, ...]:
    """Return the observed municipalities shared by every requested year."""
    requested = tuple(int(year) for year in years)
    if not requested:
        return ()
    observed = [
        set(frame.loc[frame["year"].eq(year), "municipality"].astype(str))
        for year in requested
    ]
    return tuple(sorted(set.intersection(*observed))) if observed else ()


def build_comparable_scope(
    frame: pd.DataFrame,
    selected_municipality: str,
    years: tuple[int, ...],
) -> tuple[pd.DataFrame, int]:
    """Create a temporal cohort without imputing absent municipalities."""
    requested = tuple(int(year) for year in years)
    scope = frame.loc[frame["year"].isin(requested)]
    if selected_municipality == ALL_MUNICIPALITIES_LABEL:
        municipalities = common_municipalities(frame, requested)
        scope = scope.loc[scope["municipality"].isin(municipalities)]
        cohort_size = len(municipalities)
    else:
        scope = scope.loc[scope["municipality"].eq(selected_municipality)]
        observed_years = set(scope["year"].astype(int).unique())
        cohort_size = int(set(requested).issubset(observed_years))
    return scope.copy(), cohort_size


def summarize_crime_types(frame: pd.DataFrame) -> pd.DataFrame:
    """Agrega por ID independiente; nunca colapsa categorías relacionadas."""
    summary = (
        frame.groupby(["crime_id", "crime_type", "type_label", "group", "weight"], as_index=False)
        .agg(
            count=("count", "sum"),
            day_count=("day_count", "sum"),
            night_count=("night_count", "sum"),
            weighted_count=("weighted_count", "sum"),
        )
    )
    total = float(summary["count"].sum())
    summary["share_total"] = summary["count"] / total * 100 if total else 0.0
    summary["night_share"] = summary["night_count"].div(summary["count"]).mul(100)
    summary.loc[summary["count"].eq(0), "night_share"] = pd.NA
    summary["id_order"] = summary["crime_id"].astype(float)
    return summary.sort_values("id_order").reset_index(drop=True)


def summarize_groups(frame: pd.DataFrame) -> pd.DataFrame:
    summary = (
        frame.groupby("group", as_index=False)
        .agg(
            count=("count", "sum"),
            night_count=("night_count", "sum"),
            weighted_count=("weighted_count", "sum"),
        )
    )
    total = float(summary["count"].sum())
    summary["share_total"] = summary["count"] / total * 100 if total else 0.0
    summary["night_share"] = summary["night_count"].div(summary["count"]).mul(100)
    return summary.sort_values("count", ascending=False).reset_index(drop=True)


def _entity_keys(level: str) -> tuple[list[str], str]:
    if level == TYPE_LEVEL:
        return ["crime_id", "crime_type", "type_label"], "type_label"
    if level == GROUP_LEVEL:
        return ["group"], "group"
    raise ValueError(f"Nivel no reconocido: {level}.")


def build_time_series(
    frame: pd.DataFrame,
    level: str,
    cadence: str,
    selected_entities: list[str],
) -> pd.DataFrame:
    """Serie trimestral o anual para tipos o grupos seleccionados."""
    entity_keys, label_column = _entity_keys(level)
    scope = frame.loc[frame[label_column].isin(selected_entities)].copy()
    if cadence == QUARTERLY_VIEW:
        keys = ["period_index", "period_label", *entity_keys]
    elif cadence == ANNUAL_VIEW:
        keys = ["year", *entity_keys]
    else:
        raise ValueError(f"Cadencia no reconocida: {cadence}.")
    return (
        scope.groupby(keys, as_index=False)
        .agg(count=("count", "sum"), weighted_count=("weighted_count", "sum"))
        .sort_values(keys[0])
    )


def build_entity_trends(frame: pd.DataFrame, level: str = TYPE_LEVEL) -> pd.DataFrame:
    """Calcula variaciones descriptivas sobre la cohorte territorial recibida."""
    entity_keys, label_column = _entity_keys(level)
    annual = frame.groupby(["year", *entity_keys], as_index=False)["count"].sum()
    pivot = annual.pivot(index=entity_keys, columns="year", values="count").reset_index()
    for year in YEARS:
        if year not in pivot.columns:
            pivot[year] = pd.NA

    rows: list[dict[str, object]] = []
    for _, row in pivot.iterrows():
        values = [row[year] for year in YEARS]
        y23, y24, y25 = values
        complete = all(pd.notna(value) for value in values)
        change_23_24 = _safe_change(y23, y24)
        change_24_25 = _safe_change(y24, y25)
        cumulative_change = _safe_change(y23, y25)

        if not complete:
            classification = "Cobertura parcial"
        elif y23 < y24 < y25:
            classification = "Crecimiento sostenido"
        elif y23 > y24 > y25:
            classification = "Descenso sostenido"
        else:
            mean_value = float(pd.Series(values, dtype=float).mean())
            spread = float(max(values) - min(values))
            classification = (
                "Tendencia estable"
                if spread <= max(3.0, mean_value * 0.03)
                else "Comportamiento irregular"
            )

        record: dict[str, object] = {
            label_column: row[label_column],
            "count_2023": y23,
            "count_2024": y24,
            "count_2025": y25,
            "change_23_24": change_23_24,
            "change_24_25": change_24_25,
            "cumulative_change": cumulative_change,
            "acceleration": (
                change_24_25 - change_23_24
                if change_23_24 is not None and change_24_25 is not None
                else None
            ),
            "classification": classification,
        }
        for key in entity_keys:
            record[key] = row[key]
        rows.append(record)
    return pd.DataFrame(rows)


def build_global_relevance_reference(
    frame: pd.DataFrame,
) -> tuple[frozenset[str], float]:
    """IDs elegibles según la mediana del volumen global acumulado 2023–2025."""
    accumulated = frame.groupby("crime_id", as_index=False)["count"].sum()
    threshold = float(accumulated["count"].median())
    eligible_ids = frozenset(
        accumulated.loc[accumulated["count"].gt(threshold), "crime_id"].astype(str)
    )
    return eligible_ids, threshold


def relevant_trends(
    trends: pd.DataFrame,
    change_column: str,
    eligible_ids: frozenset[str],
) -> pd.DataFrame:
    """Aplica solo a titulares la elegibilidad global estable por volumen."""
    return trends.loc[
        trends[change_column].notna()
        & trends["crime_id"].astype(str).isin(eligible_ids)
    ].copy()


def strongest_comparable_growth(
    trends: pd.DataFrame,
    change_column: str,
    selected_crime_id: str | None = None,
) -> pd.Series | None:
    """Devuelve el crecimiento máximo validado sobre la cohorte recibida.

    En el periodo completo se priorizan las dinámicas de crecimiento sostenido,
    la misma regla utilizada por el hallazgo validado de la página.
    """
    candidates = trends.loc[trends[change_column].notna()].copy()
    if selected_crime_id is not None:
        candidates = candidates.loc[
            candidates["crime_id"].astype(str).eq(str(selected_crime_id))
        ]
    elif change_column == "cumulative_change":
        sustained = candidates.loc[
            candidates["classification"].eq("Crecimiento sostenido")
        ]
        if not sustained.empty:
            candidates = sustained
    if candidates.empty:
        return None
    return candidates.sort_values(change_column, ascending=False).iloc[0]


def _growth_column(selected_year: int | None) -> tuple[str | None, str]:
    if selected_year is None:
        return "cumulative_change", "2023→2025"
    if selected_year == 2025:
        return "change_24_25", "2024→2025"
    if selected_year == 2024:
        return "change_23_24", "2023→2024"
    return None, "Sin año anterior"


def build_profile_kpis(
    filtered: pd.DataFrame,
    comparative_scope: pd.DataFrame,
    trend_scope: pd.DataFrame,
    selected_year: int | None,
    selected_crime_id: str | None = None,
    trends: pd.DataFrame | None = None,
) -> ProfileKPIs:
    type_summary = summarize_crime_types(filtered)
    total_crimes = int(type_summary["count"].sum())
    total_night = int(type_summary["night_count"].sum())
    night_share = total_night / total_crimes * 100 if total_crimes else 0.0
    top_type = type_summary.sort_values("count", ascending=False).iloc[0]
    top_impact = type_summary.sort_values("weighted_count", ascending=False).iloc[0]
    comparative_total = float(comparative_scope["count"].sum())
    top_type_share = (
        float(top_type["count"]) / comparative_total * 100
        if comparative_total
        else 0.0
    )

    trend_table = (
        trends
        if trends is not None
        else build_entity_trends(trend_scope, TYPE_LEVEL)
    )
    growth_column, growth_period = _growth_column(selected_year)
    growth_row = None
    if growth_column is not None:
        growth_row = strongest_comparable_growth(
            trend_table,
            growth_column,
            selected_crime_id,
        )

    return ProfileKPIs(
        total_crimes=total_crimes,
        total_night=total_night,
        night_share=night_share,
        top_type=str(top_type["crime_type"]),
        top_type_id=str(top_type["crime_id"]),
        top_type_count=int(top_type["count"]),
        top_type_share=top_type_share,
        top_impact_type=str(top_impact["crime_type"]),
        top_impact_group=str(top_impact["group"]),
        top_weighted_count=float(top_impact["weighted_count"]),
        weighted_load=float(type_summary["weighted_count"].sum()),
        growth_type=str(growth_row["crime_type"]) if growth_row is not None else None,
        growth_id=str(growth_row["crime_id"]) if growth_row is not None else None,
        growth_value=float(growth_row[growth_column]) if growth_row is not None else None,
        growth_period=growth_period,
    )


def summarize_territory(
    frame: pd.DataFrame,
    crime_id: str,
    selected_year: int | None,
) -> pd.DataFrame:
    scope = frame
    if selected_year is not None:
        scope = scope.loc[scope["year"].eq(int(selected_year))]
    municipality_totals = (
        scope.groupby("municipality", as_index=False)["count"]
        .sum()
        .rename(columns={"count": "municipality_total"})
    )
    selected_type = scope.loc[scope["crime_id"].eq(crime_id)]
    summary = (
        selected_type.groupby("municipality", as_index=False)
        .agg(
            count=("count", "sum"),
            night_count=("night_count", "sum"),
            weighted_count=("weighted_count", "sum"),
        )
        .merge(
            municipality_totals,
            on="municipality",
            how="left",
            validate="one_to_one",
        )
    )
    total = float(summary["count"].sum())
    summary["share_total"] = summary["count"] / total * 100 if total else 0.0
    summary["share_municipality"] = (
        summary["count"].div(summary["municipality_total"]).mul(100)
    )
    return summary.sort_values("count", ascending=False).reset_index(drop=True)


@st.cache_data(show_spinner=False)
def build_drug_case(frame: pd.DataFrame) -> DrugCase:
    """Calcula el caso Drogas sin valores ni conclusiones hardcodeadas."""
    drug_rows = frame.loc[frame["group"].eq("Drogas")].copy()
    if drug_rows.empty:
        raise ValueError("La hoja Peso Crimen no contiene un grupo Drogas.")
    labels = drug_rows["crime_type"].drop_duplicates().tolist()
    label = " / ".join(labels)
    annual = drug_rows.groupby("year", as_index=False)["count"].sum().sort_values("year")
    comparable_scope, comparable_municipality_count = build_comparable_scope(
        frame,
        ALL_MUNICIPALITIES_LABEL,
        YEARS,
    )
    comparable_drugs = comparable_scope.loc[comparable_scope["group"].eq("Drogas")]
    comparable_annual = (
        comparable_drugs.groupby("year", as_index=False)["count"]
        .sum()
        .sort_values("year")
    )
    quarterly = (
        drug_rows.groupby(["period_index", "period_label"], as_index=False)["count"]
        .sum()
        .sort_values("period_index")
    )

    annual_lookup = annual.set_index("year")["count"].to_dict()
    change_23_24 = _safe_change(annual_lookup.get(2023), annual_lookup.get(2024))
    change_24_25 = _safe_change(annual_lookup.get(2024), annual_lookup.get(2025))
    cumulative_change = _safe_change(annual_lookup.get(2023), annual_lookup.get(2025))
    comparable_lookup = comparable_annual.set_index("year")["count"].to_dict()
    comparable_cumulative_change = _safe_change(
        comparable_lookup.get(2023),
        comparable_lookup.get(2025),
    )
    accelerating = (
        change_23_24 is not None
        and change_24_25 is not None
        and change_23_24 > 0
        and change_24_25 > change_23_24
    )

    trends = build_entity_trends(comparable_scope, TYPE_LEVEL)
    drug_ids = set(drug_rows["crime_id"].unique())
    strongest_growth = strongest_comparable_growth(trends, "cumulative_change")
    strongest_sustained_growth = (
        strongest_growth is not None
        and str(strongest_growth["crime_id"]) in drug_ids
    )

    volume_2025 = (
        drug_rows.loc[drug_rows["year"].eq(2025)]
        .groupby("municipality", as_index=False)["count"]
        .sum()
        .sort_values("count", ascending=False)
        .head(10)
    )
    municipal_annual = (
        drug_rows.groupby(["municipality", "year"], as_index=False)["count"]
        .sum()
        .pivot(index="municipality", columns="year", values="count")
        .reset_index()
    )
    comparable = municipal_annual.dropna(subset=[2023, 2025]).copy()
    comparable["change_abs"] = comparable[2025] - comparable[2023]
    comparable["change_pct"] = comparable.apply(
        lambda row: _safe_change(row[2023], row[2025]), axis=1
    )
    top_growth = comparable.sort_values("change_abs", ascending=False).head(10)

    return DrugCase(
        label=label,
        annual=annual,
        comparable_annual=comparable_annual,
        quarterly=quarterly,
        top_volume=volume_2025,
        top_growth=top_growth,
        change_23_24=change_23_24,
        change_24_25=change_24_25,
        cumulative_change=cumulative_change,
        comparable_cumulative_change=comparable_cumulative_change,
        comparable_municipality_count=comparable_municipality_count,
        accelerating=accelerating,
        strongest_sustained_growth=strongest_sustained_growth,
    )
