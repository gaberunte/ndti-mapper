"""Streamlit app: draw/upload a polygon, get a binned-NDTI georeferenced PDF
averaged over the most recent cloud-free Sentinel-2 scenes."""
import datetime as dt
import tempfile
from pathlib import Path

import folium
import geopandas as gpd
import streamlit as st
from folium.plugins import Draw
from streamlit_folium import st_folium

from mapping import export_geopdf, render_preview
from ndti import (
    DEFAULT_LABELS,
    NATIVE_RESOLUTION,
    average_ndti,
    classify_ndti,
    equal_interval_bins,
    estimate_resolution,
    load_ndti_stack,
    quantile_bins,
    range_labels,
    search_recent_scenes,
)

st.set_page_config(page_title="Sentinel-2 NDTI Mapper", layout="wide")
st.title("Sentinel-2 NDTI Mapper")
st.caption(
    "Draw or upload a polygon to get a binned NDTI map, averaged over the most recent "
    "cloud-free Sentinel-2 scenes, as a georeferenced PDF."
)

with st.sidebar:
    st.header("Settings")
    max_cloud_cover = st.slider("Max scene cloud cover (%)", 0, 100, 40)
    n_scenes = st.number_input("Number of recent scenes to average", 1, 10, 3)

    st.markdown("**Imagery date range**")
    date_mode = st.radio(
        "Date range",
        ["Recent (default)", "Custom date range"],
        index=0,
        help=(
            "Recent uses whatever the most recent cloud-free scenes are, as of today. "
            "Custom date range instead searches only within a window you pick — useful "
            "for looking at a specific past period (e.g. a growing season) rather than "
            "always the latest imagery."
        ),
    )
    start_date = end_date = None
    if date_mode == "Custom date range":
        today = dt.date.today()
        picked = st.date_input(
            "Start and end date",
            value=(today - dt.timedelta(days=180), today),
            max_value=today,
        )
        if isinstance(picked, (tuple, list)) and len(picked) == 2:
            start_date, end_date = picked
        else:
            st.warning("Pick both a start and end date to search within that window.")

    st.markdown("**NDTI classes**")
    binning_mode = st.radio(
        "Binning mode",
        [
            "Equal-interval (equal NDTI range per class)",
            "Quantile (equal pixel count per class)",
            "Custom breaks",
        ],
        index=0,
        help=(
            "Equal-interval splits the observed NDTI range into evenly sized steps — "
            "intuitive, but a skewed distribution can leave some classes nearly empty. "
            "Quantile instead puts an equal number of pixels in each class, guaranteeing "
            "visual balance across the map; better for spotting relative differences "
            "within one AOI, though class boundaries are less 'round'."
        ),
    )
    if binning_mode == "Custom breaks":
        b1 = st.number_input("Bare / high disturbance <", value=0.15, step=0.01, format="%.2f")
        b2 = st.number_input("Low residue <", value=0.20, step=0.01, format="%.2f")
        b3 = st.number_input("Moderate residue <", value=0.25, step=0.01, format="%.2f")
        custom_bins = [-1.0, b1, b2, b3, 1.0]
    else:
        n_classes = st.number_input(
            "Number of classes",
            min_value=2,
            max_value=10,
            value=4,
            step=1,
            help=(
                "Quantile classification works for any number of classes, not just 4 — "
                "n=4/5/10 are traditionally called quartiles/quintiles/deciles, but it's "
                "the same equal-pixel-count method regardless of n."
            ),
        )

    st.markdown("**Resolution**")
    resolution_mode = st.radio(
        "Resolution mode",
        ["Auto (coarsens for very large AOIs)", "Custom (meters/pixel)"],
        index=0,
        help=(
            f"Sentinel-2's native resolution for these bands is {NATIVE_RESOLUTION}m/pixel. "
            "Auto keeps that for normal-sized properties, but coarsens the pixel size for "
            "very large AOIs so pixel count — and memory use — stays bounded regardless of "
            "how big the polygon is."
        ),
    )
    if resolution_mode == "Custom (meters/pixel)":
        custom_resolution = st.number_input(
            "Meters per pixel", min_value=10, value=NATIVE_RESOLUTION, step=10
        )

st.subheader("1. Define your area of interest")
upload = st.file_uploader(
    "Upload a polygon (GeoJSON, KML, or zipped Shapefile)", type=["geojson", "json", "kml", "zip"]
)

aoi_gdf = None
if upload is not None:
    with tempfile.NamedTemporaryFile(delete=False, suffix=Path(upload.name).suffix) as tmp:
        tmp.write(upload.getvalue())
        tmp_path = tmp.name
    aoi_gdf = gpd.read_file(tmp_path)
else:
    m = folium.Map(location=[36.0, -119.5], zoom_start=6)
    Draw(
        export=False,
        draw_options={
            "polygon": True,
            "rectangle": True,
            "marker": False,
            "circlemarker": False,
            "circle": False,
            "polyline": False,
        },
    ).add_to(m)
    drawn = st_folium(m, height=500, width=None, key="draw_map")
    if drawn and drawn.get("last_active_drawing"):
        aoi_gdf = gpd.GeoDataFrame.from_features([drawn["last_active_drawing"]], crs=4326)

if aoi_gdf is not None and not aoi_gdf.empty:
    st.success(f"AOI loaded: {len(aoi_gdf)} feature(s).")

    resolution = (
        custom_resolution if resolution_mode == "Custom (meters/pixel)" else estimate_resolution(aoi_gdf)
    )

    # Only the imagery pull (STAC search + band load) needs to be redone when the
    # AOI, scene count, cloud-cover threshold, date range, or resolution changes.
    # Re-binning is cheap, so it's cached separately to keep bin-toggle reruns fast.
    fetch_key = (
        aoi_gdf.geometry.union_all().wkb,
        n_scenes,
        max_cloud_cover,
        resolution,
        start_date,
        end_date,
    )

    date_range_ready = date_mode == "Recent (default)" or (start_date and end_date)

    if st.button("2. Run NDTI analysis", type="primary", disabled=not date_range_ready):
        if st.session_state.get("fetch_key") == fetch_key:
            st.info("Reusing already-fetched imagery for this AOI/settings — only re-binning.")
            mean_ndti = st.session_state["mean_ndti"]
        else:
            if resolution > NATIVE_RESOLUTION:
                st.info(
                    f"This AOI is large, so imagery is being loaded at {resolution}m/pixel "
                    f"(native is {NATIVE_RESOLUTION}m) to keep memory use bounded."
                )
            with st.spinner("Searching for Sentinel-2 scenes..."):
                items = search_recent_scenes(
                    aoi_gdf,
                    n_scenes=n_scenes,
                    max_cloud_cover=max_cloud_cover,
                    start_date=start_date,
                    end_date=end_date,
                )
            st.write(
                f"Using {len(items)} scene(s): "
                + ", ".join(i.properties["datetime"][:10] for i in items)
            )
            with st.spinner("Loading bands and computing NDTI..."):
                stack = load_ndti_stack(items, aoi_gdf, resolution=resolution)
                mean_ndti = average_ndti(stack)

            st.session_state["fetch_key"] = fetch_key
            st.session_state["mean_ndti"] = mean_ndti

        if binning_mode == "Custom breaks":
            bins, labels = custom_bins, DEFAULT_LABELS
        elif binning_mode.startswith("Quantile"):
            bins = quantile_bins(mean_ndti, n_classes=n_classes)
            labels = range_labels(bins)
        else:
            bins = equal_interval_bins(mean_ndti, n_classes=n_classes)
            labels = range_labels(bins)

        classified = classify_ndti(mean_ndti, bins=bins, labels=labels)

        with st.spinner("Building georeferenced PDF..."):
            out_dir = tempfile.mkdtemp()
            pdf_path = export_geopdf(classified, out_dir)
            pdf_bytes = pdf_path.read_bytes()

        # Stashed in session_state (rather than just rendered here) because Streamlit
        # reruns the whole script on any widget interaction -- e.g. redrawing the AOI
        # map -- and st.button() only reads True on the one rerun right after the
        # click. Rendering straight from this block would make the result vanish on
        # the very next unrelated interaction.
        st.session_state["result"] = {
            "classified": classified,
            "aoi_gdf": aoi_gdf,
            "pdf_bytes": pdf_bytes,
        }

    result = st.session_state.get("result")
    if result is not None:
        if result["aoi_gdf"].geometry.union_all().wkb != aoi_gdf.geometry.union_all().wkb:
            st.caption("Showing results from a previous AOI/run — click \"Run NDTI analysis\" to update.")
        fig = render_preview(result["classified"], result["aoi_gdf"])
        st.pyplot(fig)
        st.download_button(
            "Download georeferenced PDF", result["pdf_bytes"], file_name="ndti_map.pdf", mime="application/pdf"
        )
elif upload is None:
    st.info("Draw a polygon on the map above, or upload a file, then click \"Run NDTI analysis\".")
