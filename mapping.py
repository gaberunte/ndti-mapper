"""Rendering: classified raster -> preview figure + true georeferenced PDF export."""
from __future__ import annotations

import subprocess
from pathlib import Path

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import BoundaryNorm, ListedColormap
from matplotlib_scalebar.scalebar import ScaleBar

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
    """Write the classified raster as a GeoTIFF, then convert to a true georeferenced PDF via gdal_translate."""
    out_dir = Path(out_dir)
    tif_path = out_dir / "ndti_classified.tif"
    pdf_path = out_dir / "ndti_classified.pdf"
    export_geotiff(classified, tif_path)

    subprocess.run(
        [
            "gdal_translate",
            "-of", "PDF",
            "-co", "GEO_ENCODING=ISO32000",
            str(tif_path),
            str(pdf_path),
        ],
        check=True,
        capture_output=True,
    )
    return pdf_path
