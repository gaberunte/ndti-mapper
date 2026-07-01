"""Core geospatial pipeline: query Sentinel-2, compute averaged NDTI, classify."""
from __future__ import annotations

import geopandas as gpd
import numpy as np
import planetary_computer
import pystac_client
import rioxarray  # noqa: F401  (registers the .rio accessor)
import xarray as xr
from odc.stac import load as odc_load

STAC_URL = "https://planetarycomputer.microsoft.com/api/stac/v1"
COLLECTION = "sentinel-2-l2a"

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
):
    """Return the n most recent Sentinel-2 L2A STAC items intersecting the AOI."""
    aoi_wgs84 = aoi_gdf.to_crs(4326)
    geometry = aoi_wgs84.geometry.unary_union

    catalog = pystac_client.Client.open(STAC_URL, modifier=planetary_computer.sign_inplace)
    search = catalog.search(
        collections=[COLLECTION],
        intersects=geometry.__geo_interface__,
        query={"eo:cloud_cover": {"lt": max_cloud_cover}},
        sortby=[{"field": "properties.datetime", "direction": "desc"}],
        max_items=max_lookback_items,
    )
    items = list(search.items())
    if not items:
        raise ValueError("No Sentinel-2 scenes found for this area/cloud-cover threshold.")
    return items[:n_scenes]


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


def average_ndti(ndti_stack: xr.DataArray) -> xr.DataArray:
    """Average NDTI across scenes, ignoring cloud-masked pixels."""
    mean = ndti_stack.mean(dim="time", skipna=True)
    mean = mean.rio.write_crs(ndti_stack.rio.crs)
    return mean


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
