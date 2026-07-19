"""Municipal eligibility and official GeoJSON source management."""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator

import pandas as pd
import requests
import streamlit as st


PROJECT_ROOT = Path(__file__).resolve().parents[1]
GEOJSON_PATH = PROJECT_ROOT / "data" / "municipios_madrid.geojson"
ARCGIS_LAYER_URL = (
    "https://services1.arcgis.com/nCKYwcSONQTkPA4K/ArcGIS/rest/services/"
    "muni/FeatureServer/0"
)

CRIME_INDEX = "Indice_criminal_ponderado_por_10000_hab"
REQUIRED_PROPERTIES = {"NAMEUNIT", "codine", "NATCODE", "CODNUT3"}

# Explicit-only alias policy. The July 2026 audit found all 37 project names to
# be exact NAMEUNIT matches, so no aliases are currently required. Add any
# future verified discrepancy here; fuzzy matching is deliberately forbidden.
MUNICIPALITY_NAME_ALIASES: dict[str, str] = {}


@dataclass(frozen=True)
class MapSource:
    """A project-scoped polygon source ready for Plotly."""

    mode: str
    frame: pd.DataFrame
    geojson: dict[str, Any] | None = None
    message: str | None = None
    viewport: dict[str, object] | None = None
    center: dict[str, float] | None = None
    territorial_viewport: dict[str, object] | None = None

    @property
    def available(self) -> bool:
        return self.mode == "polygons" and not self.frame.empty and self.geojson is not None


@dataclass(frozen=True)
class ProjectMapGeometry:
    """Geometría inmutable compartida por todas las vistas municipales."""

    geojson: dict[str, Any]
    feature_ids: dict[str, str]
    bounds: tuple[float, float, float, float]
    viewport: dict[str, object]
    center: dict[str, float]
    territorial_viewport: dict[str, object]


def _iter_positions(coordinates: Any) -> Iterator[tuple[float, float]]:
    """Yield polygon positions without requiring a geospatial dependency."""
    if (
        isinstance(coordinates, list)
        and len(coordinates) >= 2
        and isinstance(coordinates[0], (int, float))
        and isinstance(coordinates[1], (int, float))
    ):
        yield float(coordinates[0]), float(coordinates[1])
        return
    if isinstance(coordinates, list):
        for child in coordinates:
            yield from _iter_positions(child)


def geojson_bounds(geojson: dict[str, Any]) -> tuple[float, float, float, float]:
    """Return west, south, east and north for a FeatureCollection."""
    positions = [
        point
        for feature in geojson["features"]
        for point in _iter_positions(feature["geometry"]["coordinates"])
    ]
    if not positions:
        raise ValueError("Las geometrías municipales no contienen coordenadas.")
    longitudes, latitudes = zip(*positions)
    return min(longitudes), min(latitudes), max(longitudes), max(latitudes)


def map_center_from_bounds(
    bounds: tuple[float, float, float, float],
) -> dict[str, float]:
    """Preserve the arithmetic center historically used by the main map."""
    west, south, east, north = bounds
    return {"lon": (west + east) / 2, "lat": (south + north) / 2}


def mapbox_viewport_from_bounds(
    bounds: tuple[float, float, float, float],
    *,
    width: int,
    height: int,
    padding_ratio: float = 0.08,
) -> dict[str, object]:
    """Calculate the existing Mapbox viewport from precomputed bounds."""
    if width <= 0 or height <= 0:
        raise ValueError("El viewport cartográfico requiere dimensiones positivas.")
    if not 0 <= padding_ratio < 0.5:
        raise ValueError("El padding cartográfico debe estar entre 0 y 0,5.")

    west, south, east, north = bounds

    def mercator_y(latitude: float) -> float:
        bounded = min(max(latitude, -85.05112878), 85.05112878)
        radians = math.radians(bounded)
        return math.log(math.tan(math.pi / 4 + radians / 2))

    south_y, north_y = mercator_y(south), mercator_y(north)
    center_y = (south_y + north_y) / 2
    center_latitude = math.degrees(
        2 * math.atan(math.exp(center_y)) - math.pi / 2
    )
    longitude_fraction = max((east - west) / 360, 1e-12)
    latitude_fraction = max(abs(north_y - south_y) / (2 * math.pi), 1e-12)
    usable_width = width * (1 - 2 * padding_ratio)
    usable_height = height * (1 - 2 * padding_ratio)
    zoom_longitude = math.log2(usable_width / 512 / longitude_fraction)
    zoom_latitude = math.log2(usable_height / 512 / latitude_fraction)
    return {
        "center": {"lon": (west + east) / 2, "lat": center_latitude},
        "zoom": max(0.0, min(zoom_longitude, zoom_latitude)),
    }


def mapbox_viewport_from_geojson(
    geojson: dict[str, Any],
    *,
    width: int,
    height: int,
    padding_ratio: float = 0.08,
) -> dict[str, object]:
    """Compatibility wrapper for callers without a prepared geometry."""
    return mapbox_viewport_from_bounds(
        geojson_bounds(geojson),
        width=width,
        height=height,
        padding_ratio=padding_ratio,
    )


def get_eligible_municipalities(frame: pd.DataFrame, year: int) -> pd.DataFrame:
    """Return only annual rows with an existing weighted crime index."""
    return (
        frame.loc[frame["Año"].eq(int(year)) & frame[CRIME_INDEX].notna()]
        .copy()
        .sort_values(["Ranking_criminal_anual", "Municipio"], na_position="last")
        .reset_index(drop=True)
    )


def municipality_from_map_selection(
    event: object,
    eligible_names: tuple[str, ...],
) -> str | None:
    """Extract one eligible municipality from a Streamlit Plotly event."""
    try:
        points = event["selection"]["points"]  # type: ignore[index]
    except (KeyError, TypeError):
        return None
    if not points:
        return None
    clicked = points[0].get("customdata")
    if isinstance(clicked, (list, tuple)):
        clicked = clicked[0] if clicked else None
    return str(clicked) if clicked in set(eligible_names) else None


def _validate_geojson(payload: dict[str, Any]) -> None:
    """Validate the cached/downloaded municipal FeatureCollection."""
    if payload.get("type") != "FeatureCollection":
        raise ValueError("La fuente geográfica no es un FeatureCollection.")

    features = payload.get("features")
    if not isinstance(features, list) or len(features) < 37:
        raise ValueError("La fuente geográfica no contiene suficientes municipios.")

    for feature in features:
        geometry = feature.get("geometry") or {}
        properties = feature.get("properties") or {}
        if geometry.get("type") not in {"Polygon", "MultiPolygon"}:
            raise ValueError("La fuente contiene una geometría no poligonal.")
        if REQUIRED_PROPERTIES.difference(properties):
            raise ValueError("La fuente geográfica no contiene todos los campos requeridos.")
        if properties["CODNUT3"] != "ES300":
            raise ValueError("La caché contiene entidades fuera de CODNUT3 ES300.")
        if not str(properties["codine"]).startswith("28"):
            raise ValueError("La caché contiene una unidad no municipal de Madrid.")


@st.cache_resource(show_spinner=False)
def _read_local_geojson(path_text: str, modified_ns: int) -> dict[str, Any]:
    """Read the immutable local geometry once, keyed by its timestamp."""
    del modified_ns  # Used exclusively as a cache invalidation key.
    with Path(path_text).open(encoding="utf-8") as geojson_file:
        payload = json.load(geojson_file)
    _validate_geojson(payload)
    return payload


def _download_geojson(path: Path) -> dict[str, Any]:
    """Download Madrid municipalities in GeoJSON/WGS84 and cache atomically."""
    response = requests.get(
        f"{ARCGIS_LAYER_URL}/query",
        params={
            # ES300 also contains two non-municipal communal units (codes 53xxx).
            # The explicit 28% condition retains the 179 Madrid municipalities.
            "where": "CODNUT3='ES300' AND codine LIKE '28%'",
            "outFields": "NAMEUNIT,codine,NATCODE,CODNUT3",
            "returnGeometry": "true",
            "outSR": "4326",
            "f": "geojson",
        },
        timeout=45,
    )
    response.raise_for_status()
    payload = response.json()
    _validate_geojson(payload)

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_suffix(".tmp")
    with temporary_path.open("w", encoding="utf-8") as geojson_file:
        json.dump(payload, geojson_file, ensure_ascii=False, separators=(",", ":"))
    temporary_path.replace(path)
    return payload


def load_municipality_geojson(path: Path = GEOJSON_PATH) -> tuple[dict[str, Any] | None, str | None]:
    """Prefer the local cache; download only when it does not yet exist."""
    path = Path(path)
    if path.exists():
        try:
            return _read_local_geojson(str(path), path.stat().st_mtime_ns), None
        except (OSError, ValueError, json.JSONDecodeError) as exc:
            local_error = f"La caché geográfica local no es válida: {exc}"
        else:  # pragma: no cover - kept for clarity around the fallback path.
            local_error = None
    else:
        local_error = None

    try:
        return _download_geojson(path), local_error
    except (OSError, ValueError, requests.RequestException, json.JSONDecodeError) as exc:
        message = f"No se ha podido cargar la cartografía municipal: {exc}"
        if local_error:
            message = f"{local_error} {message}"
        return None, message


@st.cache_resource(show_spinner=False)
def _load_project_map_geometry(
    path_text: str,
    modified_ns: int,
    project_names: tuple[str, ...],
) -> tuple[ProjectMapGeometry | None, str | None]:
    """Resolve and prepare the 37 static polygons once per GeoJSON version."""
    del modified_ns  # Cache invalidation key; the loader reads the same path.
    payload, load_message = load_municipality_geojson(Path(path_text))
    if payload is None:
        return None, load_message

    feature_by_name = {
        str(feature["properties"]["NAMEUNIT"]): feature
        for feature in payload["features"]
    }
    resolved_features: list[dict[str, Any]] = []
    feature_ids: dict[str, str] = {}
    missing_names: list[str] = []
    for project_name in project_names:
        source_name = MUNICIPALITY_NAME_ALIASES.get(project_name, project_name)
        source_feature = feature_by_name.get(source_name)
        if source_feature is None:
            missing_names.append(project_name)
            continue

        feature_id = str(source_feature["properties"]["codine"])
        properties = dict(source_feature["properties"])
        properties["project_name"] = project_name
        properties["feature_id"] = feature_id
        resolved_features.append(
            {
                "type": "Feature",
                "geometry": source_feature["geometry"],
                "properties": properties,
            }
        )
        feature_ids[project_name] = feature_id

    if missing_names:
        missing = ", ".join(missing_names)
        return None, (
            "No existe una correspondencia geográfica explícita para: "
            f"{missing}. No se aplican coincidencias aproximadas."
        )
    if len(resolved_features) != 37 or len(project_names) != 37:
        return None, "La cartografía no contiene exactamente los 37 municipios del proyecto."

    project_geojson = {"type": "FeatureCollection", "features": resolved_features}
    bounds = geojson_bounds(project_geojson)
    return (
        ProjectMapGeometry(
            geojson=project_geojson,
            feature_ids=feature_ids,
            bounds=bounds,
            viewport=mapbox_viewport_from_bounds(
                bounds,
                width=720,
                height=410,
                padding_ratio=0.09,
            ),
            center=map_center_from_bounds(bounds),
            territorial_viewport=mapbox_viewport_from_bounds(
                bounds,
                width=920,
                height=620,
                padding_ratio=0.12,
            ),
        ),
        load_message,
    )


def load_project_map_geometry(
    municipality_names: pd.Series,
    path: Path = GEOJSON_PATH,
) -> tuple[ProjectMapGeometry | None, str | None]:
    """Return the shared project geometry for an exact set of 37 names."""
    path = Path(path)
    project_names = tuple(sorted(municipality_names.dropna().astype(str).unique()))
    modified_ns = path.stat().st_mtime_ns if path.exists() else -1
    return _load_project_map_geometry(str(path), modified_ns, project_names)


def prepare_project_map_source(
    frame: pd.DataFrame,
    municipality_column: str,
    geojson_path: Path = GEOJSON_PATH,
) -> MapSource:
    """Attach dynamic municipal data to the shared static geometry."""
    if municipality_column not in frame.columns:
        raise ValueError(f"No existe la columna municipal {municipality_column}.")
    if len(frame) != 37 or frame[municipality_column].nunique() != 37:
        return MapSource(
            mode="unavailable",
            frame=pd.DataFrame(),
            message="La vista no contiene exactamente los 37 municipios del proyecto.",
        )

    geometry, load_message = load_project_map_geometry(
        frame[municipality_column], geojson_path
    )
    if geometry is None:
        return MapSource(
            mode="unavailable",
            frame=pd.DataFrame(),
            message=load_message,
        )

    map_frame = frame.copy()
    map_frame["_feature_id"] = map_frame[municipality_column].map(
        geometry.feature_ids
    )
    return MapSource(
        mode="polygons",
        frame=map_frame,
        geojson=geometry.geojson,
        message=load_message,
        viewport=geometry.viewport,
        center=geometry.center,
        territorial_viewport=geometry.territorial_viewport,
    )


def prepare_map_source(
    frame: pd.DataFrame,
    year: int,
    geojson_path: Path = GEOJSON_PATH,
) -> MapSource:
    """Select one year and link its 37 project municipalities to polygons."""
    year_frame = (
        frame.loc[frame["Año"].eq(int(year))]
        .copy()
        .sort_values("Municipio")
        .reset_index(drop=True)
    )
    source = prepare_project_map_source(year_frame, "Municipio", geojson_path)
    if not source.available:
        return source
    map_frame = source.frame.copy()
    map_frame["_eligible"] = map_frame[CRIME_INDEX].notna()
    return MapSource(
        mode="polygons",
        frame=map_frame,
        geojson=source.geojson,
        message=source.message,
        viewport=source.viewport,
        center=source.center,
        territorial_viewport=source.territorial_viewport,
    )
