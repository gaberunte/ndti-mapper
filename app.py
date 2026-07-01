"""Streamlit app: draw/upload a polygon, get a binned-NDTI georeferenced PDF
averaged over the most recent cloud-free Sentinel-2 scenes."""
import tempfile
from pathlib import Path

import folium
import geopandas as gpd
import streamlit as st
from folium.plugins import Draw
from streamlit_folium import st_folium

from mapping import export_geopdf, render_preview
from ndti import DEFAULT_LABELS, average_ndti, classify_ndti, load_ndti_stack, search_recent_scenes

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
    st.markdown("**NDTI class breaks**")
    b1 = st.number_input("Bare / high disturbance <", value=0.15, step=0.01, format="%.2f")
    b2 = st.number_input("Low residue <", value=0.20, step=0.01, format="%.2f")
    b3 = st.number_input("Moderate residue <", value=0.25, step=0.01, format="%.2f")
    bins = [-1.0, b1, b2, b3, 1.0]

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

    if st.button("2. Run NDTI analysis", type="primary"):
        with st.spinner("Searching for recent Sentinel-2 scenes..."):
            items = search_recent_scenes(aoi_gdf, n_scenes=n_scenes, max_cloud_cover=max_cloud_cover)
        st.write(
            f"Using {len(items)} scene(s): "
            + ", ".join(i.properties["datetime"][:10] for i in items)
        )

        with st.spinner("Loading bands and computing NDTI..."):
            stack = load_ndti_stack(items, aoi_gdf)
            mean_ndti = average_ndti(stack)
            classified = classify_ndti(mean_ndti, bins=bins, labels=DEFAULT_LABELS)

        fig = render_preview(classified, aoi_gdf)
        st.pyplot(fig)

        with st.spinner("Building georeferenced PDF..."):
            out_dir = tempfile.mkdtemp()
            pdf_path = export_geopdf(classified, out_dir)

        with open(pdf_path, "rb") as f:
            st.download_button("Download georeferenced PDF", f, file_name="ndti_map.pdf", mime="application/pdf")
elif upload is None:
    st.info("Draw a polygon on the map above, or upload a file, then click \"Run NDTI analysis\".")
