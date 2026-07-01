"""Rendering: classified raster -> preview figure + true georeferenced PDF export."""
from __future__ import annotations

from pathlib import Path

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import rasterio
from matplotlib.colors import BoundaryNorm, ListedColormap, to_hex
from matplotlib_scalebar.scalebar import ScaleBar
from PIL import Image
from rasterio.io import MemoryFile
from rasterio.shutil import copy as rio_copy

# Diverging brown -> teal colormap (bare -> high residue), same family as the original
# fixed 4-color palette but sampled to fit however many classes are actually in use.
_BASE_CMAP = "BrBG"
NODATA_BYTE = 255


def _class_colors(n: int) -> list[str]:
    cmap = plt.get_cmap(_BASE_CMAP, n)
    return [to_hex(cmap(i)) for i in range(n)]


def _hex_to_rgb(hex_color: str) -> tuple[int, int, int]:
    hex_color = hex_color.lstrip("#")
    return tuple(int(hex_color[i : i + 2], 16) for i in (0, 2, 4))


def _class_colormap(n: int) -> dict:
    colors = _class_colors(n)
    cmap = {i: (*_hex_to_rgb(colors[i]), 255) for i in range(n)}
    cmap[NODATA_BYTE] = (255, 255, 255, 0)  # transparent
    return cmap


def render_preview(classified, aoi_gdf, title="NDTI (scene average)"):
    labels = classified.attrs["labels"]
    n = len(labels)
    colors = _class_colors(n)
    cmap = ListedColormap(colors)
    norm = BoundaryNorm(np.arange(-0.5, n + 0.5, 1), cmap.N)

    fig, ax = plt.subplots(figsize=(8, 8))
    classified.plot.imshow(ax=ax, cmap=cmap, norm=norm, add_colorbar=False)
    aoi_gdf.to_crs(classified.rio.crs).boundary.plot(ax=ax, edgecolor="black", linewidth=1.2)

    handles = [mpatches.Patch(color=colors[i], label=labels[i]) for i in range(n)]
    ax.legend(handles=handles, loc="lower left", fontsize=8, framealpha=0.9)
    ax.set_title(title)
    ax.set_axis_off()
    ax.add_artist(ScaleBar(1, location="lower right"))
    fig.tight_layout()
    return fig


def _legend_panel_rgb(labels, title, height_px, width_px=220, dpi=150):
    """Render a title + color-swatch legend as an RGB array, resized to exactly height_px tall."""
    n = len(labels)
    colors = _class_colors(n)
    fig = plt.figure(figsize=(width_px / dpi, height_px / dpi), dpi=dpi)
    fig.patch.set_facecolor("white")
    ax = fig.add_axes((0, 0, 1, 1))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    ax.text(0.08, 0.95, title, fontsize=10, fontweight="bold", va="top", wrap=True)

    handles = [mpatches.Patch(color=colors[i], label=labels[i]) for i in range(n)]
    ax.legend(handles=handles, loc="center left", bbox_to_anchor=(0.0, 0.55), frameon=False, fontsize=7.5)

    fig.canvas.draw()
    panel = np.asarray(fig.canvas.buffer_rgba())[:, :, :3]
    plt.close(fig)

    if panel.shape[0] != height_px:
        panel = np.array(Image.fromarray(panel).resize((panel.shape[1], height_px)))
    return panel


def _classified_to_rgb(classified):
    """Render the classified raster as an RGB array (white background outside the AOI)."""
    labels = classified.attrs["labels"]
    n = len(labels)
    colors = _class_colors(n)
    data = classified.values
    rgb = np.full((*data.shape, 3), 255, dtype="uint8")
    for i in range(n):
        rgb[data == i] = _hex_to_rgb(colors[i])
    return rgb


def export_geotiff(classified, out_path):
    n = len(classified.attrs["labels"])
    data = np.where(np.isnan(classified.values), NODATA_BYTE, classified.values).astype("uint8")

    profile = dict(
        driver="GTiff",
        height=data.shape[0],
        width=data.shape[1],
        count=1,
        dtype="uint8",
        crs=classified.rio.crs,
        transform=classified.rio.transform(),
        nodata=NODATA_BYTE,
        photometric="PALETTE",
    )
    with rasterio.open(out_path, "w", **profile) as dst:
        dst.write(data, 1)
        dst.write_colormap(1, _class_colormap(n))


def export_geopdf(classified, out_dir, title="NDTI (scene average)") -> Path:
    """Write the classified raster + legend as a true georeferenced, colored PDF via rasterio's bundled GDAL.

    GDAL's PDF driver only supports CreateCopy (not Create), so an in-memory GeoTIFF is
    built and copied into the PDF driver. The legend is rendered as a color-swatch panel
    and appended as extra columns of real RGB pixels alongside the map, so it travels
    inside the same georeferenced file (those legend pixels just carry extrapolated,
    meaningless coordinates past the map's real extent, which is harmless for a report).
    """
    out_dir = Path(out_dir)
    pdf_path = out_dir / "ndti_classified.pdf"
    labels = classified.attrs["labels"]

    map_rgb = _classified_to_rgb(classified)
    legend_rgb = _legend_panel_rgb(labels, title, height_px=map_rgb.shape[0])
    combined = np.hstack([map_rgb, legend_rgb])  # (rows, map_cols + legend_cols, 3)

    profile = dict(
        driver="GTiff",
        height=combined.shape[0],
        width=combined.shape[1],
        count=3,
        dtype="uint8",
        crs=classified.rio.crs,
        transform=classified.rio.transform(),
    )

    with MemoryFile() as memfile:
        with memfile.open(**profile) as mem:
            for band in range(3):
                mem.write(combined[:, :, band], band + 1)
        rio_copy(memfile.name, str(pdf_path), driver="PDF", GEO_ENCODING="ISO32000")

    return pdf_path
