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
from rasterio.features import rasterize
from rasterio.io import MemoryFile
from rasterio.shutil import copy as rio_copy
from rasterio.transform import array_bounds, from_bounds

# Diverging brown -> teal colormap (bare -> high residue), same family as the original
# fixed 4-color palette but sampled to fit however many classes are actually in use.
_BASE_CMAP = "BrBG"
NODATA_BYTE = 255
# Fill for pixels inside the AOI that have no valid class (cloud-masked, non-grassland
# under the land-cover mask, etc.) -- distinct from the plain white/transparent background
# outside the AOI, so "no data here" doesn't read the same as "not part of the property."
_MASKED_COLOR = (224, 224, 224)  # matplotlib "0.88" gray, matched in the raster export
_MASKED_GRAY_MPL = "0.88"
_LEGEND_DPI = 300  # render density for the legend panel's text -- unlike the map (bounded
# by the source imagery's real resolution), text has no such ceiling, so this is set high
# for print-quality glyphs rather than matching the map's pixel density.
_MIN_LEGEND_HEIGHT_IN = 6.0  # floor (in inches) for laying out the legend panel's text,
# see _legend_panel_rgb -- expressed in inches (not pixels) so it scales with _LEGEND_DPI.
_MAX_PDF_MAP_DIM = 2000  # cap on the PDF's map width/height in pixels, see export_geopdf


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

    aoi_proj = aoi_gdf.to_crs(classified.rio.crs)

    fig, ax = plt.subplots(figsize=(8, 8))
    # Gray AOI fill underneath: the raster's NaN pixels render transparent, so this
    # shows through for masked-out pixels (cloud, non-grassland, ...) inside the AOI.
    aoi_proj.plot(ax=ax, facecolor=_MASKED_GRAY_MPL, edgecolor="none", zorder=0)
    classified.plot.imshow(ax=ax, cmap=cmap, norm=norm, add_colorbar=False, zorder=1)
    aoi_proj.boundary.plot(ax=ax, edgecolor="black", linewidth=1.2, zorder=2)

    handles = [mpatches.Patch(color=colors[i], label=labels[i]) for i in range(n)]
    handles.append(mpatches.Patch(color=_MASKED_GRAY_MPL, label="No data (cloud/mask)"))
    ax.legend(handles=handles, loc="lower left", fontsize=8, framealpha=0.9)
    ax.set_title(title)
    ax.set_axis_off()
    ax.add_artist(ScaleBar(1, location="lower right"))
    fig.tight_layout()
    return fig


def _text_width_px(fig, text, fontsize, **text_kwargs):
    """Measure `text`'s rendered width in pixels at `fontsize`, using the real renderer
    (font metrics vary by weight/size/glyph, so a character count can't stand in for this)."""
    renderer = fig.canvas.get_renderer()
    probe = fig.text(0, 0, text, fontsize=fontsize, **text_kwargs)
    width = probe.get_window_extent(renderer=renderer).width
    probe.remove()
    return width


def _wrap_to_width(fig, text, fontsize, max_width_px, **text_kwargs):
    """Word-wrap `text` to fit `max_width_px`, measured with the actual font metrics."""
    lines = []
    current = ""
    for word in text.split():
        trial = f"{current} {word}".strip()
        width = _text_width_px(fig, trial, fontsize, **text_kwargs)
        if current and width > max_width_px:
            lines.append(current)
            current = word
        else:
            current = trial
    if current:
        lines.append(current)
    return lines or [text]


def _legend_panel_rgb(labels, title, height_px, metadata_lines=None, min_width_px=220, dpi=_LEGEND_DPI):
    """Render a title + metadata + color-swatch legend as an RGB array, resized to exactly height_px tall.

    Text is laid out on a canvas at least `_MIN_LEGEND_HEIGHT_IN` tall, then uniformly
    scaled to fit `height_px` (preserving aspect ratio, so nothing looks stretched) -- a
    raster with few rows (small AOI, coarse resolution) would otherwise give fixed-size
    fonts too little room, clipping the legend off the bottom. `dpi` defaults high (see
    `_LEGEND_DPI`) since this is the one part of the export not bounded by imagery
    resolution -- downscaling from a dense render gives properly antialiased, print-quality
    text rather than the comparatively blocky/soft result of rendering at screen density.

    The legend's own color-swatch rows are drawn manually rather than via matplotlib's
    `ax.legend()`, which doesn't know the panel's pixel width and would just let long
    labels (e.g. "No data (cloud/mask)") run past its edge and get clipped. The panel's
    width is instead measured from the actual content up front, so it's always wide
    enough for the widest legend label (title/metadata wrap onto multiple lines instead).
    """
    n = len(labels)
    colors = _class_colors(n)
    legend_labels = list(labels) + ["No data (cloud/mask)"]
    legend_colors = colors + [to_hex([c / 255 for c in _MASKED_COLOR])]
    render_px = max(height_px, round(_MIN_LEGEND_HEIGHT_IN * dpi))

    # Margins/swatch size are defined in points (a physical, dpi-independent unit, same
    # as fontsize) and converted to pixels for this dpi, so layout proportions stay the
    # same regardless of how dense the render is -- only converting to raw pixels here
    # would leave them a fixed, effectively shrinking size as dpi (and thus text) grows.
    left_margin_px = round(6.7 * dpi / 72)
    right_margin_px = round(6.7 * dpi / 72)
    swatch_w_px = round(12.5 * dpi / 72)
    swatch_gap_px = round(3.8 * dpi / 72)

    probe_fig = plt.figure(figsize=(1, 1), dpi=dpi)
    max_label_px = max(_text_width_px(probe_fig, lbl, 7.5) for lbl in legend_labels)
    plt.close(probe_fig)

    content_w_px = left_margin_px + swatch_w_px + swatch_gap_px + max_label_px + right_margin_px
    width_px = max(min_width_px, round(content_w_px))
    max_text_width_px = width_px - left_margin_px - right_margin_px

    fig = plt.figure(figsize=(width_px / dpi, render_px / dpi), dpi=dpi)
    fig.patch.set_facecolor("white")
    ax = fig.add_axes((0, 0, 1, 1))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    left_frac = left_margin_px / width_px

    # Line heights are computed as an axis-coordinate fraction (rather than a fixed
    # constant) so stacked text blocks don't collide regardless of the render height.
    def line_frac(fontsize, leading=1.35):
        return fontsize * dpi / 72 * leading / render_px

    y = 0.95
    title_lines = _wrap_to_width(fig, title, 10, max_text_width_px, fontweight="bold")
    ax.text(left_frac, y, "\n".join(title_lines), fontsize=10, fontweight="bold", va="top")
    y -= line_frac(10) * len(title_lines) + line_frac(10) * 0.6

    if metadata_lines:
        wrapped = [
            line
            for raw in metadata_lines
            for line in _wrap_to_width(fig, raw, 6, max_text_width_px, color="0.25")
        ]
        ax.text(left_frac, y, "\n".join(wrapped), fontsize=6, va="top", color="0.25", linespacing=1.6)
        y -= line_frac(6, leading=1.6) * len(wrapped) + line_frac(6) * 1.2

    y -= line_frac(7.5) * 0.6
    row_h = line_frac(7.5, leading=1.8)
    swatch_w_frac = swatch_w_px / width_px
    text_x_frac = (left_margin_px + swatch_w_px + swatch_gap_px) / width_px
    for color, label in zip(legend_colors, legend_labels):
        ax.add_patch(
            mpatches.Rectangle((left_frac, y - row_h * 0.75), swatch_w_frac, row_h * 0.5, facecolor=color)
        )
        ax.text(text_x_frac, y - row_h * 0.5, label, fontsize=7.5, va="center", ha="left")
        y -= row_h

    fig.canvas.draw()
    panel = np.asarray(fig.canvas.buffer_rgba())[:, :, :3]
    plt.close(fig)

    if panel.shape[0] != height_px:
        # Scale both dimensions by the same factor -- resizing only the height (as a
        # naive fit-to-height would) stretches everything non-uniformly. LANCZOS (rather
        # than the default filter) is what makes the high `dpi` render above pay off as
        # properly antialiased text once downscaled to the map's actual pixel height.
        scale = height_px / panel.shape[0]
        new_w = max(1, round(panel.shape[1] * scale))
        panel = np.array(Image.fromarray(panel).resize((new_w, height_px), Image.LANCZOS))
    return panel


def _classified_to_rgb(classified, aoi_mask=None):
    """Render the classified raster as an RGB array: white outside the AOI, gray for
    masked-out pixels (cloud, non-grassland, ...) within the AOI if `aoi_mask` is given."""
    labels = classified.attrs["labels"]
    n = len(labels)
    colors = _class_colors(n)
    data = classified.values
    rgb = np.full((*data.shape, 3), 255, dtype="uint8")
    if aoi_mask is not None:
        rgb[np.isnan(data) & aoi_mask] = _MASKED_COLOR
    for i in range(n):
        rgb[data == i] = _hex_to_rgb(colors[i])
    return rgb


def _aoi_mask(aoi_gdf, crs, transform, shape):
    """Boolean raster mask, True where inside the AOI polygon(s) (fill, not just boundary)."""
    aoi_proj = aoi_gdf.to_crs(crs)
    shapes = [geom for geom in aoi_proj.geometry if geom is not None and not geom.is_empty]
    if not shapes:
        return np.zeros(shape, dtype=bool)

    mask = rasterize(
        [(geom, 1) for geom in shapes],
        out_shape=shape,
        transform=transform,
        fill=0,
        dtype="uint8",
    )
    return mask.astype(bool)


def _draw_boundary(rgb, aoi_gdf, crs, transform, color=(0, 0, 0), width_px=2):
    """Burn the AOI boundary as a solid outline directly into an RGB raster array,
    so it survives export to a plain georeferenced raster/PDF (no matplotlib overlay)."""
    aoi_proj = aoi_gdf.to_crs(crs)
    px_size = abs(transform.a)
    outline = aoi_proj.boundary.buffer(px_size * width_px / 2)
    shapes = [geom for geom in outline if geom is not None and not geom.is_empty]
    if not shapes:
        return rgb

    mask = rasterize(
        [(geom, 1) for geom in shapes],
        out_shape=rgb.shape[:2],
        transform=transform,
        fill=0,
        dtype="uint8",
    )
    rgb = rgb.copy()
    rgb[mask == 1] = color
    return rgb


def export_geotiff(classified, out_dir) -> Path:
    """Write the classified raster as a single-band, palette-colored GeoTIFF.

    Pixel values are the class index (0..n-1); a colormap matching the preview/PDF is
    embedded, and pixels outside the AOI (or cloud-masked) carry the nodata value.
    """
    out_dir = Path(out_dir)
    tif_path = out_dir / "ndti_classified.tif"
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
        compress="deflate",
    )
    with rasterio.open(tif_path, "w", **profile) as dst:
        dst.write(data, 1)
        dst.write_colormap(1, _class_colormap(n))
    return tif_path


def export_ndti_geotiff(ndti_mean, out_dir) -> Path:
    """Write the continuous averaged NDTI as a single-band float32 GeoTIFF (nan = nodata).

    This is the raw data product -- unbinned NDTI values -- for further analysis in GIS,
    as opposed to the classified/palette raster from ``export_geotiff``.
    """
    out_dir = Path(out_dir)
    tif_path = out_dir / "ndti_continuous.tif"
    data = ndti_mean.values.astype("float32")

    profile = dict(
        driver="GTiff",
        height=data.shape[0],
        width=data.shape[1],
        count=1,
        dtype="float32",
        crs=ndti_mean.rio.crs,
        transform=ndti_mean.rio.transform(),
        nodata=float("nan"),
        compress="deflate",
    )
    with rasterio.open(tif_path, "w", **profile) as dst:
        dst.write(data, 1)
        dst.set_band_description(1, "NDTI (scene composite)")
    return tif_path


def export_geopdf(classified, aoi_gdf, out_dir, title="NDTI (scene average)", metadata_lines=None) -> Path:
    """Write the classified raster + legend as a true georeferenced, colored PDF via rasterio's bundled GDAL.

    GDAL's PDF driver only supports CreateCopy (not Create), so an in-memory GeoTIFF is
    built and copied into the PDF driver. The legend is rendered as a color-swatch panel
    and appended as extra columns of real RGB pixels alongside the map, so it travels
    inside the same georeferenced file (those legend pixels just carry extrapolated,
    meaningless coordinates past the map's real extent, which is harmless for a report).
    The AOI boundary is burned directly into the map pixels (there's no vector overlay
    in a flat raster PDF), so the property line stays visible in the exported map.
    `metadata_lines`, if given, is printed under the title (e.g. scene dates used, cloud
    cover threshold, resolution) so that context isn't lost once the PDF leaves the app.

    The GDAL PDF driver maps raster pixels to page points 1:1, with no notion of a
    target page size -- so a large AOI, or one with an oddly-shaped/elongated bounding
    box, can produce an enormous or absurdly elongated page that squeezes the legend's
    fixed pixel width down to nothing. The map is capped to `_MAX_PDF_MAP_DIM` on its
    longer side (nearest-neighbor, so class colors stay exact) before layout, which
    keeps the page a sane, predictable shape regardless of the AOI's real geometry; the
    full-resolution GeoTIFF exports are unaffected.
    """
    out_dir = Path(out_dir)
    pdf_path = out_dir / "ndti_classified.pdf"
    labels = classified.attrs["labels"]
    transform = classified.rio.transform()

    aoi_mask = _aoi_mask(aoi_gdf, classified.rio.crs, transform, classified.shape)
    map_rgb = _classified_to_rgb(classified, aoi_mask=aoi_mask)

    orig_h, orig_w = map_rgb.shape[:2]
    scale = _MAX_PDF_MAP_DIM / max(orig_h, orig_w)
    if scale < 1:
        # Downscale before drawing the boundary: a hairline burned in at full resolution
        # can be aliased away entirely by a large nearest-neighbor downsample, so it's
        # drawn fresh on the final grid instead, at a consistent width regardless of scale.
        new_w, new_h = max(1, round(orig_w * scale)), max(1, round(orig_h * scale))
        map_rgb = np.array(Image.fromarray(map_rgb).resize((new_w, new_h), Image.NEAREST))
        bounds = array_bounds(orig_h, orig_w, transform)
        transform = from_bounds(*bounds, new_w, new_h)

    map_rgb = _draw_boundary(map_rgb, aoi_gdf, classified.rio.crs, transform)
    legend_rgb = _legend_panel_rgb(labels, title, height_px=map_rgb.shape[0], metadata_lines=metadata_lines)
    combined = np.hstack([map_rgb, legend_rgb])  # (rows, map_cols + legend_cols, 3)

    profile = dict(
        driver="GTiff",
        height=combined.shape[0],
        width=combined.shape[1],
        count=3,
        dtype="uint8",
        crs=classified.rio.crs,
        transform=transform,
    )

    with MemoryFile() as memfile:
        with memfile.open(**profile) as mem:
            for band in range(3):
                mem.write(combined[:, :, band], band + 1)
        rio_copy(memfile.name, str(pdf_path), driver="PDF", GEO_ENCODING="ISO32000")

    return pdf_path
