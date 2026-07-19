# Cartografía municipal de Madrid

- Fuente: ArcGIS FeatureServer `muni`, basado en datos del Instituto Geográfico Nacional (IGN/CNIG).
- Capa: `https://services1.arcgis.com/nCKYwcSONQTkPA4K/ArcGIS/rest/services/muni/FeatureServer/0`
- Formato: GeoJSON con geometrías poligonales en WGS84 (EPSG:4326).
- Atribución: `Fuente cartográfica: Obra derivada de BDLJE CC-BY 4.0 ign.es`.

## Consulta y depuración explícita

La descarga utiliza:

```text
CODNUT3 = 'ES300' AND codine LIKE '28%'
```

`CODNUT3 = 'ES300'` devuelve también dos unidades no municipales con códigos
53xxx (`Los Baldios` y `El Redegüelo`). El filtro adicional conserva los 179
municipios oficiales con código provincial 28.

## Correspondencia con Crime Pulse Madrid

Los 37 valores de `Municipio` presentes en `tabla_maestra.csv` coinciden
exactamente con `NAMEUNIT`. No se aplican coincidencias aproximadas ni
normalizaciones silenciosas. El diccionario explícito
`MUNICIPALITY_NAME_ALIASES` de `utils/map_data.py` está actualmente vacío y es
el único lugar autorizado para documentar futuras equivalencias verificadas.
