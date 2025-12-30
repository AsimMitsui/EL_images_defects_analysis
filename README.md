# PV EL Cell Segmentation (Streamlit)

This small app segments PV EL module images into individual cells using classical computer vision methods.

## Quickstart

1. Create a virtual environment (recommended) and install:
   ```
   pip install -r requirements.txt
   ```

2. Run the app:
   ```
   streamlit run app.py
   ```

3. Upload a PV EL image, tweak parameters in the sidebar, press "Run segmentation", preview results, and download as ZIP.

## How it works (brief)
- Converts image to grayscale, applies CLAHE and Gaussian blur.
- Optionally crops to module boundary using edge detection.
- Uses adaptive thresholding (or Otsu) to separate bright cell regions.
- Morphological open/close to clean noise.
- Finds contours, filters by area and aspect ratio, then sorts bounding boxes into rows (a simple grid heuristic).
- Exports cropped cell images and binary masks.

## Tips for better results
- Increase `min area` if many small speckles are detected.
- Tweak `morph kernel` to better close cell regions.
- Increase `row tolerance` if rows are not detected properly (varies with image scale).
- For very noisy images or non-rectangular modules, consider training a segmentation network (U-Net) and replacing `detect_cells_from_image()`.

## Next improvements you might want
- Use Hough lines or morphological filtering to explicitly detect grid lines for robust grid extraction.
- Add an optional U-Net model loader for hardened segmentation when available.
- Auto-estimate typical cell size and default parameters based on image resolution.

License: MIT (use and modify as you like)