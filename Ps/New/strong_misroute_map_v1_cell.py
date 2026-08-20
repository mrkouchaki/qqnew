# ============================================================
# STRONG MISROUTE MAP V1
# Run after strong_misroutes_1000m_to_30000m_v2_1_cell.py
#
# Displays:
#   - Top N calls in the selected routed-distance range.
#   - Full routed and expected PSAP service polygons.
#   - Call uncertainty circles.
#   - Lines from each call to the nearest point on each PSAP boundary.
#
# PSAP markers are representative points inside service polygons, not
# physical dispatch-center addresses.
# ============================================================

from pathlib import Path
from html import escape

import numpy as np
import pandas as pd

try:
    import folium
    from folium.plugins import MarkerCluster
except ImportError as exc:
    raise ImportError(
        "This map cell requires folium. Install it once in the active "
        "notebook kernel with: %pip install folium"
    ) from exc

try:
    from shapely.geometry import Point, mapping
    from shapely.ops import nearest_points
except ImportError as exc:
    raise ImportError(
        "This map cell requires shapely. Install it once in the active "
        "notebook kernel with: %pip install shapely"
    ) from exc

from IPython.display import display, Markdown


# ------------------------------------------------------------
# Settings
# ------------------------------------------------------------

MAP_TOP_N_CALLS = 25
MAP_MIN_ROUTED_DISTANCE_M = 1000
MAP_MAX_ROUTED_DISTANCE_M = 30000  # Use None for no upper limit.

# False draws the complete polygons for only the routed/expected PSAPs used
# by the selected calls. True also draws the entire available PSAP catalog and
# can create a much larger/slower HTML file.
SHOW_ALL_PSAP_BOUNDARIES = False

OUTPUT_DIR = Path(globals().get(
    "OUTPUT_DIR", r"C:\temp\gmlc_v2\outputs_psap_rca_v4"
))
MAP_OUTPUT_PATH = OUTPUT_DIR / "strong_misroute_top_calls_map_v1.html"


def map_norm_id(value):
    if value is None or pd.isna(value):
        return ""
    text = str(value).strip().strip('"').upper()
    return text[:-2] if text.endswith(".0") else text


def map_number(value, default=np.nan):
    number = pd.to_numeric(pd.Series([value]), errors="coerce").iloc[0]
    return default if pd.isna(number) else float(number)


def map_text(value, default="<MISSING>"):
    if value is None or pd.isna(value):
        return default
    text = str(value).strip()
    return text if text else default


def representative_point_latlon(geometry):
    if geometry is None or geometry.is_empty:
        return np.nan, np.nan
    point = geometry.representative_point()
    return float(point.y), float(point.x)


# ------------------------------------------------------------
# 1. Load the V2.1 call report or use its in-memory DataFrame
# ------------------------------------------------------------

if (
    "strong_misroute_calls_v2" in globals()
    and isinstance(strong_misroute_calls_v2, pd.DataFrame)
):
    map_calls_source = strong_misroute_calls_v2.copy()
else:
    preferred = OUTPUT_DIR / (
        "strong_misroute_calls_1000m_to_30000m_v2_1.csv"
    )
    candidates = [preferred] + sorted(
        OUTPUT_DIR.glob("strong_misroute_calls_*_v2_1.csv"),
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )
    source_path = next((path for path in candidates if path.exists()), None)
    if source_path is None:
        raise FileNotFoundError(
            "No V2.1 call report was found. Run the V2.1 strong-misroute "
            "cell first."
        )
    map_calls_source = pd.read_csv(
        source_path, dtype="string", low_memory=False,
        encoding_errors="replace",
    )

map_calls_source.columns = [
    str(column).strip().upper().replace(" ", "_")
    for column in map_calls_source.columns
]

required_call_columns = {
    "FCC_PSAP_ID",
    "EXPECTED_FCC_PSAP_ID",
    "LATITUDE",
    "LONGITUDE",
    "DISTANCE_TO_ROUTED_BOUNDARY_M",
    "DISTANCE_TO_EXPECTED_BOUNDARY_M",
}
missing_call_columns = sorted(
    required_call_columns - set(map_calls_source.columns)
)
if missing_call_columns:
    raise ValueError(
        f"The call report is missing required columns: {missing_call_columns}"
    )

for column in [
    "LATITUDE",
    "LONGITUDE",
    "UNCERT_METERS",
    "DISTANCE_TO_ROUTED_BOUNDARY_M",
    "DISTANCE_TO_EXPECTED_BOUNDARY_M",
]:
    if column in map_calls_source.columns:
        map_calls_source[column] = pd.to_numeric(
            map_calls_source[column], errors="coerce"
        )

for column in ["FCC_PSAP_ID", "EXPECTED_FCC_PSAP_ID"]:
    map_calls_source[column] = map_calls_source[column].map(map_norm_id)

valid_location = (
    map_calls_source["LATITUDE"].between(-90, 90)
    & map_calls_source["LONGITUDE"].between(-180, 180)
)
distance_filter = map_calls_source[
    "DISTANCE_TO_ROUTED_BOUNDARY_M"
].ge(MAP_MIN_ROUTED_DISTANCE_M)
if MAP_MAX_ROUTED_DISTANCE_M is not None:
    distance_filter &= map_calls_source[
        "DISTANCE_TO_ROUTED_BOUNDARY_M"
    ].le(MAP_MAX_ROUTED_DISTANCE_M)

map_calls = (
    map_calls_source[valid_location & distance_filter]
    .sort_values(
        "DISTANCE_TO_ROUTED_BOUNDARY_M",
        ascending=False,
        kind="stable",
    )
    .head(MAP_TOP_N_CALLS)
    .reset_index(drop=True)
)
map_calls.insert(0, "MAP_RANK", range(1, len(map_calls) + 1))

if map_calls.empty:
    raise ValueError(
        "No calls with valid coordinates satisfy the selected map distance "
        "range. Change MAP_MIN_ROUTED_DISTANCE_M or "
        "MAP_MAX_ROUTED_DISTANCE_M."
    )


# ------------------------------------------------------------
# 2. Obtain the reconstructed PSAP service polygons
# ------------------------------------------------------------

if (
    "psap_geometry_by_fcc" in globals()
    and isinstance(psap_geometry_by_fcc, dict)
):
    map_geometry_by_fcc = {
        map_norm_id(fcc): geometry
        for fcc, geometry in psap_geometry_by_fcc.items()
        if geometry is not None
    }
elif "boundary_by_fcc" in globals() and isinstance(boundary_by_fcc, dict):
    map_geometry_by_fcc = {
        map_norm_id(fcc): geometry
        for fcc, geometry in boundary_by_fcc.items()
        if geometry is not None
    }
elif "boundaries" in globals() and isinstance(boundaries, pd.DataFrame):
    if not {"FCC_PSAP_ID", "GEOMETRY"}.issubset(boundaries.columns):
        raise ValueError(
            "The in-memory boundaries DataFrame does not contain "
            "FCC_PSAP_ID and GEOMETRY."
        )
    map_geometry_by_fcc = {
        map_norm_id(fcc): geometry
        for fcc, geometry in zip(
            boundaries["FCC_PSAP_ID"], boundaries["GEOMETRY"]
        )
        if geometry is not None
    }
else:
    raise RuntimeError(
        "PSAP polygons are not in memory. Run notebook Section 2 and then "
        "the V2.1 report cell before this map cell."
    )


# ------------------------------------------------------------
# 3. Create map and draw complete relevant PSAP polygons
# ------------------------------------------------------------

center_lat = float(map_calls["LATITUDE"].median())
center_lon = float(map_calls["LONGITUDE"].median())
misroute_map_v1 = folium.Map(
    location=[center_lat, center_lon],
    zoom_start=5,
    tiles="CartoDB positron",
    control_scale=True,
    prefer_canvas=True,
)

all_boundaries_layer = folium.FeatureGroup(
    name="All available PSAP boundaries",
    show=SHOW_ALL_PSAP_BOUNDARIES,
)
routed_boundaries_layer = folium.FeatureGroup(
    name="Routed PSAP full boundaries",
    show=True,
)
expected_boundaries_layer = folium.FeatureGroup(
    name="Expected PSAP full boundaries",
    show=True,
)
routed_lines_layer = folium.FeatureGroup(
    name="Distance to routed boundary",
    show=True,
)
expected_lines_layer = folium.FeatureGroup(
    name="Distance to expected boundary",
    show=True,
)
uncertainty_layer = folium.FeatureGroup(
    name="Call uncertainty circles",
    show=True,
)
call_cluster = MarkerCluster(name="Problematic calls", show=True)

if SHOW_ALL_PSAP_BOUNDARIES:
    for fcc, geometry in map_geometry_by_fcc.items():
        if geometry is None or geometry.is_empty:
            continue
        folium.GeoJson(
            data=mapping(geometry),
            style_function=lambda _feature: {
                "color": "#777777",
                "weight": 0.6,
                "fillOpacity": 0.0,
            },
            tooltip=f"FCC PSAP {escape(fcc)}",
        ).add_to(all_boundaries_layer)
    all_boundaries_layer.add_to(misroute_map_v1)

routed_ids = set(map_calls["FCC_PSAP_ID"])
expected_ids = set(map_calls["EXPECTED_FCC_PSAP_ID"])
map_extent_points = []

for role, psap_ids, layer, color in [
    ("Routed", routed_ids, routed_boundaries_layer, "#c62828"),
    ("Expected", expected_ids, expected_boundaries_layer, "#2e7d32"),
]:
    for fcc in sorted(psap_ids):
        geometry = map_geometry_by_fcc.get(fcc)
        if geometry is None or geometry.is_empty:
            continue

        folium.GeoJson(
            data=mapping(geometry),
            style_function=lambda _feature, line_color=color: {
                "color": line_color,
                "weight": 3,
                "fillColor": line_color,
                "fillOpacity": 0.08,
            },
            tooltip=f"{role} PSAP FCC {escape(fcc)}",
        ).add_to(layer)

        rep_lat, rep_lon = representative_point_latlon(geometry)
        if np.isfinite(rep_lat) and np.isfinite(rep_lon):
            folium.CircleMarker(
                location=[rep_lat, rep_lon],
                radius=6,
                color=color,
                fill=True,
                fill_color=color,
                fill_opacity=0.9,
                tooltip=(
                    f"{role} PSAP FCC {escape(fcc)} representative point "
                    "(not building location)"
                ),
            ).add_to(layer)
            map_extent_points.append([rep_lat, rep_lon])

        min_lon, min_lat, max_lon, max_lat = geometry.bounds
        map_extent_points.extend([
            [float(min_lat), float(min_lon)],
            [float(max_lat), float(max_lon)],
        ])

routed_boundaries_layer.add_to(misroute_map_v1)
expected_boundaries_layer.add_to(misroute_map_v1)


# ------------------------------------------------------------
# 4. Draw calls, uncertainty circles, and both boundary lines
# ------------------------------------------------------------

for _, row in map_calls.iterrows():
    rank = int(row["MAP_RANK"])
    call_lat = float(row["LATITUDE"])
    call_lon = float(row["LONGITUDE"])
    call_point = Point(call_lon, call_lat)
    uncertainty_m = max(map_number(row.get("UNCERT_METERS"), 0.0), 0.0)
    routed_distance_m = map_number(
        row.get("DISTANCE_TO_ROUTED_BOUNDARY_M")
    )
    expected_distance_m = map_number(
        row.get("DISTANCE_TO_EXPECTED_BOUNDARY_M")
    )
    routed_fcc = map_norm_id(row.get("FCC_PSAP_ID"))
    expected_fcc = map_norm_id(row.get("EXPECTED_FCC_PSAP_ID"))
    routed_geometry = map_geometry_by_fcc.get(routed_fcc)
    expected_geometry = map_geometry_by_fcc.get(expected_fcc)

    popup_fields = [
        ("Rank", rank),
        ("Call time UTC", map_text(row.get("CALL_TIME_UTC"))),
        ("Call key", map_text(row.get("CALL_KEY"))),
        ("USID", map_text(row.get("USID"))),
        ("ESRK", map_text(row.get("ESRK"))),
        ("Call latitude", f"{call_lat:.6f}"),
        ("Call longitude", f"{call_lon:.6f}"),
        ("Uncertainty", f"{uncertainty_m:,.1f} m"),
        (
            "Routed PSAP",
            f"{routed_fcc} — {map_text(row.get('ROUTED_PSAP_NAME'))}",
        ),
        (
            "Expected PSAP",
            f"{expected_fcc} — {map_text(row.get('EXPECTED_PSAP_NAME'))}",
        ),
        ("Routed-boundary distance", f"{routed_distance_m:,.1f} m"),
        ("Expected-boundary distance", f"{expected_distance_m:,.1f} m"),
        ("Route status", map_text(row.get("ROUTE_STATUS_GMLC"))),
        ("Fallback/WDLS", map_text(row.get("ROUTE_FALLBACK"))),
        (
            "Additional evidence",
            map_text(row.get("ADDITIONAL_NETWORK_EVIDENCE")),
        ),
    ]
    popup_html = "<br>".join(
        f"<b>{escape(str(label))}:</b> {escape(str(value))}"
        for label, value in popup_fields
    )

    folium.Marker(
        location=[call_lat, call_lon],
        tooltip=(
            f"#{rank}: {routed_fcc} → {expected_fcc}; "
            f"routed distance {routed_distance_m:,.0f} m"
        ),
        popup=folium.Popup(popup_html, max_width=500),
        icon=folium.Icon(color="blue", icon="phone", prefix="fa"),
    ).add_to(call_cluster)

    if uncertainty_m > 0:
        folium.Circle(
            location=[call_lat, call_lon],
            radius=uncertainty_m,
            color="#1565c0",
            weight=2,
            fill=True,
            fill_color="#1565c0",
            fill_opacity=0.10,
            tooltip=f"Call #{rank} uncertainty radius: {uncertainty_m:,.1f} m",
        ).add_to(uncertainty_layer)

    for role, geometry, distance_m, color, layer in [
        (
            "Routed",
            routed_geometry,
            routed_distance_m,
            "#c62828",
            routed_lines_layer,
        ),
        (
            "Expected",
            expected_geometry,
            expected_distance_m,
            "#2e7d32",
            expected_lines_layer,
        ),
    ]:
        if geometry is None or geometry.is_empty:
            continue
        try:
            boundary_point = nearest_points(call_point, geometry.boundary)[1]
            folium.PolyLine(
                locations=[
                    [call_lat, call_lon],
                    [float(boundary_point.y), float(boundary_point.x)],
                ],
                color=color,
                weight=3,
                opacity=0.85,
                dash_array=None if role == "Routed" else "7,6",
                tooltip=(
                    f"Call #{rank} → {role.lower()} PSAP boundary: "
                    f"{distance_m:,.1f} m"
                ),
            ).add_to(layer)
        except Exception:
            pass

    map_extent_points.append([call_lat, call_lon])

uncertainty_layer.add_to(misroute_map_v1)
routed_lines_layer.add_to(misroute_map_v1)
expected_lines_layer.add_to(misroute_map_v1)
call_cluster.add_to(misroute_map_v1)

folium.LayerControl(collapsed=False).add_to(misroute_map_v1)

legend_html = """
<div style="position: fixed; bottom: 24px; left: 24px; z-index: 9999;
background: white; border: 1px solid #888; padding: 8px 10px;
font-size: 12px; line-height: 1.5;">
<b>Misroute map</b><br>
<span style="color:#1565c0;">●</span> Call / uncertainty circle<br>
<span style="color:#c62828;">━</span> Routed PSAP boundary and distance<br>
<span style="color:#2e7d32;">┄</span> Expected PSAP boundary and distance<br>
PSAP points are polygon representatives, not buildings
</div>
"""
misroute_map_v1.get_root().html.add_child(folium.Element(legend_html))

if map_extent_points:
    misroute_map_v1.fit_bounds(map_extent_points, padding=(20, 20))

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
misroute_map_v1.save(MAP_OUTPUT_PATH)

display(Markdown(
    f"## Top {len(map_calls):,} problematic calls: "
    f"{MAP_MIN_ROUTED_DISTANCE_M:,.0f}–"
    f"{MAP_MAX_ROUTED_DISTANCE_M:,.0f} m"
    if MAP_MAX_ROUTED_DISTANCE_M is not None else
    f"## Top {len(map_calls):,} problematic calls: >= "
    f"{MAP_MIN_ROUTED_DISTANCE_M:,.0f} m"
))
display(misroute_map_v1)
print(f"Saved interactive map: {MAP_OUTPUT_PATH}")
