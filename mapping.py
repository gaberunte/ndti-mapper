"""Rendering: classified raster -> preview figure + true georeferenced PDF export."""
from __future__ import annotations

from pathlib import Path

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import rasterio
from matplotlib.colors import BoundaryNorm, ListedColormap
from matplotlib_scalebar.scalebar import ScaleBar
from rasterio.io import MemoryFile
from rasterio.shutil import copy as rio_copy

CLASS_COLORS = ["#8c510a", "#d8b365", "#80cdc1", "#01665e"]  # bare -> high residue


def render_preview(classified, aoi_gdf, title="NDTI (scene average)"):
    labels = classified.attrs["labels"]
    n = len(labels)
    cmap = ListedColormap(CLASS_COLORS[:n])
    norm = BoundaryNorm(np.arange(-0.5, n + 0.5, 1), cmap.N)

    fig, ax = plt.subplots(figsize=(8, 8))
    classified.plot.imshow(ax=ax, cmap=cmap, norm=norm, add_colorbar=False)
    aoi_gdf.to_crs(classified.rio.crs).boundary.plot(ax=ax, edgecolor="black", linewidth=1.2)

    handles = [mpatches.Patch(color=CLASS_COLORS[i], label=labels[i]) for i in range(n)]
    ax.legend(handles=handles, loc="lower left", fontsize=8, framealpha=0.9)
    ax.set_title(title)
    ax.set_axis_off()
    ax.add_artist(ScaleBar(1, location="lower right"))
    fig.tight_layout()
    return fig


def export_geotiff(classified, out_path):
    classified.rio.to_raster(out_path, dtype="float32")


def export_geopdf(classified, out_dir) -> Path:
    """Write the classified raster as a true georeferenced PDF via rasterio's bundled GDAL.

    GDAL's PDF driver only supports CreateCopy (not Create), so we build an
    in-memory GeoTIFF first and copy it into the PDF driver.
    """
    out_dir = Path(out_dir)
    pdf_path = out_dir / "ndti_classified.pdf"

    # PDF driver only supports 8-bit bands, so classes are cast to uint8
    # with 255 reserved as the nodata sentinel for masked-out pixels.
    nodata = 255
    data = np.where(np.isnan(classified.values), nodata, classified.values).astype("uint8")

    profile = dict(
        driver="GTiff",
        height=data.shape[0],
        width=data.shape[1],
        count=1,
        dtype="uint8",
        crs=classified.rio.crs,
        transform=classified.rio.transform(),
        nodata=nodata,
    )

    with MemoryFile() as memfile:
        with memfile.open(**profile) as mem:
            mem.write(data, 1)
        rio_copy(memfile.name, str(pdf_path), driver="PDF", GEO_ENCODING="ISO32000")

    return pdf_path
