"""Core geospatial pipeline: query Sentinel-2, compute averaged NDTI, classify."""
from __future__ import annotations

import geopandas as gpd
import numpy as np
import planetary_computer
import pystac_client
import rioxarray  # noqa: F401  (registers the .rio accessor)
import xarray as xr
from odc.stac import load as odc_load
from rasterio.enums import Resampling

STAC_URL = "https://planetarycomputer.microsoft.com/api/stac/v1"
COLLECTION = "sentinel-2-l2a"

WORLDCOVER_COLLECTION = "esa-worldcover"
WORLDCOVER_VERSION = "2.0.0"  # 2021 epoch; avoids mixing with the older 2020 v1.0.0 product
WORLDCOVER_GRASSLAND_CODE = 30

NATIVE_RESOLUTION = 20  # meters; native resolution of the SWIR bands used for NDTI
# Keeps peak memory in the few-hundred-MB range: a ~56 km^2 real ranch AOI at native
# resolution (~380k pixels) measured ~310MB peak RSS end-to-end, well under Streamlit
# Community Cloud's ~1GB ceiling.
DEFAULT_MAX_PIXELS = 600_000

# Sentinel-2 SCL classes kept as "clear": vegetation, bare soil, unclassified.
# Excludes cloud/shadow/cirrus/snow/water, which aren't meaningful for a tillage index.
CLEAR_SCL_CLASSES = [4, 5, 7]

DEFAULT_BINS = [-1.0, 0.15, 0.20, 0.25, 1.0]
DEFAULT_LABELS = [
    "Bare / high disturbance",
    "Low residue",
    "Moderate residue",
    "High residue / no-till",
]


def search_recent_scenes(
    aoi_gdf: gpd.GeoDataFrame,
    n_scenes: int = 3,
    max_cloud_cover: float = 40,
    max_lookback_items: int = 30,
    start_date=None,
    end_date=None,
):
    """Return the n most recent Sentinel-2 L2A STAC items intersecting the AOI.

    If start_date/end_date (date objects) are given, search is bounded to that window
    and "most recent" means the n most recent scenes within it, rather than overall.
    """
    aoi_wgs84 = aoi_gdf.to_crs(4326)
    geometry = aoi_wgs84.geometry.union_all()

    datetime_filter = None
    if start_date or end_date:
        start_str = start_date.isoformat() if start_date else ".."
        end_str = end_date.isoformat() if end_date else ".."
        datetime_filter = f"{start_str}/{end_str}"

    catalog = pystac_client.Client.open(STAC_URL, modifier=planetary_computer.sign_inplace)
    search = catalog.search(
        collections=[COLLECTION],
        intersects=geometry.__geo_interface__,
        datetime=datetime_filter,
        query={"eo:cloud_cover": {"lt": max_cloud_cover}},
        sortby=[{"field": "properties.datetime", "direction": "desc"}],
        max_items=max_lookback_items,
    )
    items = list(search.items())
    if not items:
        window = f" between {start_date} and {end_date}" if datetime_filter else ""
        raise ValueError(f"No Sentinel-2 scenes found for this area/cloud-cover threshold{window}.")
    return items[:n_scenes]


def estimate_resolution(aoi_gdf: gpd.GeoDataFrame, max_pixels: int = DEFAULT_MAX_PIXELS) -> int:
    """Pick a pixel size (meters) that keeps the AOI's pixel count under max_pixels.

    Stays at Sentinel-2's native 20m for normal-sized properties; coarsens (in 10m
    steps) for very large AOIs so memory use doesn't scale unbounded with area.
    """
    utm = aoi_gdf.to_crs(aoi_gdf.estimate_utm_crs())
    minx, miny, maxx, maxy = utm.total_bounds
    width, height = maxx - minx, maxy - miny

    native_pixels = (width / NATIVE_RESOLUTION) * (height / NATIVE_RESOLUTION)
    if native_pixels <= max_pixels:
        return NATIVE_RESOLUTION

    needed = NATIVE_RESOLUTION * (native_pixels / max_pixels) ** 0.5
    return int(np.ceil(needed / 10.0) * 10)


def load_ndti_stack(items, aoi_gdf: gpd.GeoDataFrame, resolution: int = 20) -> xr.DataArray:
    """Load B11/B12/SCL for the given items, clip to the AOI, return per-scene NDTI (nan where cloudy)."""
    aoi_wgs84 = aoi_gdf.to_crs(4326)
    bbox = tuple(aoi_wgs84.total_bounds)

    ds = odc_load(
        items,
        bands=["B11", "B12", "SCL"],
        bbox=bbox,
        resolution=resolution,
        chunks={},
    )

    clear_mask = ds.SCL.isin(CLEAR_SCL_CLASSES)
    ndti = ((ds.B11 - ds.B12) / (ds.B11 + ds.B12)).where(clear_mask)

    aoi_native = aoi_gdf.to_crs(ndti.rio.crs)
    ndti_clipped = ndti.rio.clip(aoi_native.geometry.values, aoi_native.crs, drop=True)
    return ndti_clipped  # dims: time, y, x


def load_grassland_mask(aoi_gdf: gpd.GeoDataFrame, match: xr.DataArray) -> xr.DataArray:
    """Load ESA WorldCover for the AOI and return a boolean mask (True = grassland),
    resampled onto `match`'s exact grid (nearest-neighbor, since land cover is categorical)."""
    aoi_wgs84 = aoi_gdf.to_crs(4326)
    bbox = tuple(aoi_wgs84.total_bounds)

    catalog = pystac_client.Client.open(STAC_URL, modifier=planetary_computer.sign_inplace)
    search = catalog.search(
        collections=[WORLDCOVER_COLLECTION],
        bbox=bbox,
        query={"esa_worldcover:product_version": {"eq": WORLDCOVER_VERSION}},
    )
    items = list(search.items())
    if not items:
        raise ValueError("No ESA WorldCover coverage found for this area.")

    ds = odc_load(items, bands=["map"], bbox=bbox, chunks={})
    landcover = ds["map"]
    if "time" in landcover.dims:
        # Adjacent WorldCover tiles are separate time steps; they don't spatially
        # overlap, and 0 (nodata) sorts below every real class code (10-100).
        landcover = landcover.max(dim="time")

    landcover_matched = landcover.rio.reproject_match(match, resampling=Resampling.nearest)
    return landcover_matched == WORLDCOVER_GRASSLAND_CODE


def average_ndti(ndti_stack: xr.DataArray) -> xr.DataArray:
    """Average NDTI across scenes, ignoring cloud-masked pixels."""
    mean = ndti_stack.mean(dim="time", skipna=True)
    mean = mean.rio.write_crs(ndti_stack.rio.crs)
    return mean


def equal_interval_bins(ndti_mean: xr.DataArray, n_classes: int = 4) -> list[float]:
    """Split this scene's actual NDTI range into n_classes equal-width bins."""
    values = ndti_mean.values
    vmin = float(np.nanmin(values))
    vmax = float(np.nanmax(values))
    return np.linspace(vmin, vmax, n_classes + 1).tolist()


def quantile_bins(ndti_mean: xr.DataArray, n_classes: int = 4) -> list[float]:
    """Split this scene's NDTI values into n_classes bins holding an equal pixel count each."""
    values = ndti_mean.values
    percentiles = np.linspace(0, 100, n_classes + 1)
    edges = np.nanpercentile(values, percentiles)
    # Guard against duplicate edges when many pixels share a value (e.g. large flat areas)
    edges = np.unique(edges)
    if len(edges) < 2:
        edges = np.array([float(np.nanmin(values)), float(np.nanmax(values))])
    return edges.tolist()


def range_labels(bins: list[float]) -> list[str]:
    """Generate 'low..high' legend labels from bin edges, e.g. for data-driven bins."""
    return [f"{bins[i]:.3f} – {bins[i + 1]:.3f}" for i in range(len(bins) - 1)]


def classify_ndti(ndti_mean: xr.DataArray, bins=None, labels=None) -> xr.DataArray:
    """Bin the averaged NDTI into discrete classes, preserving nodata as nan."""
    bins = bins if bins is not None else DEFAULT_BINS
    labels = labels if labels is not None else DEFAULT_LABELS

    values = ndti_mean.values
    classified = np.digitize(values, bins[1:-1]).astype(float)
    classified[np.isnan(values)] = np.nan

    out = ndti_mean.copy(data=classified)
    out = out.rio.write_crs(ndti_mean.rio.crs)
    out.attrs["bins"] = bins
    out.attrs["labels"] = labels
    return out
