import streamlit as st
from PIL import Image
import numpy as np
import io
import zipfile
import cv2
from segmentation import detect_cells_from_image, draw_bboxes_on_image, crop_cells_from_image

st.set_page_config(page_title="PV EL Cell Segmentation", layout="wide")

st.title("PV EL Module → Cell Segmentation (Streamlit)")

st.markdown(
    """
Upload an electroluminescence (EL) image of a PV module. Tune the parameters on the sidebar,
press "Run segmentation", inspect the detected cells, and download cropped cells + masks as a ZIP.
"""
)

uploaded = st.file_uploader("Upload EL PV module image (jpg / png / tiff)", type=["jpg", "jpeg", "png", "tif", "tiff"])

st.sidebar.header("Segmentation parameters")
blur_ksize = st.sidebar.slider("Gaussian blur kernel (odd)", 1, 21, 5, step=2)
adaptive = st.sidebar.checkbox("Use adaptive threshold (else Otsu)", value=True)
th_blocksize = st.sidebar.slider("Adaptive blockSize (odd)", 3, 101, 35, step=2)
th_C = st.sidebar.slider("Adaptive C (constant subtracted)", -20, 40, 5)
morph_kernel = st.sidebar.slider("Morph kernel size", 1, 31, 5, step=2)
min_area = st.sidebar.slider("Minimum contour area (px)", 50, 50000, 1000)
max_area = st.sidebar.number_input("Maximum contour area (px, 0 = no limit)", value=0, step=100)
aspect_min = st.sidebar.slider("Min aspect ratio (w/h)", 0.1, 1.0, 0.4)
aspect_max = st.sidebar.slider("Max aspect ratio (w/h)", 1.0, 4.0, 1.6)
row_tol = st.sidebar.slider("Row tolerance (px) for grouping", 5, 200, 25)

run = st.button("Run segmentation")

if uploaded is not None:
    image = Image.open(uploaded).convert("RGB")
    image_np = np.array(image)[:, :, ::-1]  # PIL RGB -> BGR for OpenCV
    st.sidebar.image(image, caption="Uploaded image", use_column_width=True)

    if run:
        with st.spinner("Running segmentation..."):
            results = detect_cells_from_image(
                image_np,
                blur_ksize=blur_ksize,
                adaptive=adaptive,
                th_blocksize=th_blocksize,
                th_C=th_C,
                morph_kernel=morph_kernel,
                min_area=min_area,
                max_area=int(max_area) if max_area > 0 else None,
                aspect_min=aspect_min,
                aspect_max=aspect_max,
                row_tol=row_tol,
            )

        bboxes = results["bboxes"]
        masks = results["masks"]
        debug_img = draw_bboxes_on_image(image_np.copy(), bboxes)

        st.subheader(f"Detected cells: {len(bboxes)}")
        st.image(cv2.cvtColor(debug_img, cv2.COLOR_BGR2RGB), caption="Detections", use_column_width=True)

        # Show a small gallery of crops
        cols = st.columns(4)
        for i, (bbox, mask) in enumerate(zip(bboxes, masks)):
            if i >= 8:
                break
            x, y, w, h = bbox
            crop = crop_cells_from_image(image_np, bbox)
            crop_rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
            cols[i % 4].image(crop_rgb, caption=f"Cell {i+1}", use_column_width=True)

        # Offer download as zip
        buf = io.BytesIO()
        with zipfile.ZipFile(buf, "w", compression=zipfile.ZIP_DEFLATED) as zf:
            for i, (bbox, mask) in enumerate(zip(bboxes, masks)):
                crop = crop_cells_from_image(image_np, bbox)
                crop_rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
                pil_crop = Image.fromarray(crop_rgb)
                crop_bytes = io.BytesIO()
                pil_crop.save(crop_bytes, format="PNG")
                zf.writestr(f"cell_{i+1:03d}.png", crop_bytes.getvalue())

                # mask as PNG
                pil_mask = Image.fromarray((mask * 255).astype(np.uint8))
                mask_bytes = io.BytesIO()
                pil_mask.save(mask_bytes, format="PNG")
                zf.writestr(f"cell_{i+1:03d}_mask.png", mask_bytes.getvalue())

            # full debug image
            debug_rgb = cv2.cvtColor(debug_img, cv2.COLOR_BGR2RGB)
            pil_dbg = Image.fromarray(debug_rgb)
            dbg_bytes = io.BytesIO()
            pil_dbg.save(dbg_bytes, format="PNG")
            zf.writestr("debug_detections.png", dbg_bytes.getvalue())

        buf.seek(0)
        st.download_button("Download detected cells and masks (ZIP)", data=buf, file_name="cells_and_masks.zip", mime="application/zip")

else:
    st.info("Upload an image to start. You can tune segmentation parameters in the sidebar.")

st.markdown("----")
st.markdown("Notes: This uses classical CV heuristics that work well for regular PV modules with visible cell boundaries. For more robust segmentation on noisy / damaged images, consider training a U-Net style model and swapping the detection function.")
