# Old Photo Restorer — Web App

A browser-based image restoration application built with Python (Flask + OpenCV) and vanilla HTML/CSS/JavaScript.

Restores and enhances old or degraded photographs using classical Digital Image Processing techniques — no AI or machine learning.

## Running the Web App

```bash
# Option 1: Use the startup script (auto-installs dependencies)
./run_web.sh

# Option 2: Manual
cd old-photo-restoration-pipeline
venv/bin/pip install flask flask-cors
PORT=8080 venv/bin/python server.py
```

Then open **http://localhost:8080** in your browser.

> **Note:** Port 5000 is used by macOS AirPlay Receiver. The app defaults to port 8080.  
> Override with: `PORT=9000 venv/bin/python server.py`

---

## Modes

### ⚡ Quick Restore
Fully automatic restoration. Select strength (Light / Medium / Strong) and style (Natural / Vintage / B&W), then click **Restore**. A restoration report shows exactly what was applied.

### 🔧 Prepare
Manual preprocessing: Histogram Equalization, CLAHE, Auto White Balance, Normalize, Resize.

### 🌫 Denoise
Manual denoising: Gaussian, Median, Bilateral, Non-Local Means — each with configurable parameters.

### 🔍 Deblur
Manual sharpening/deblurring: Unsharp Mask, Wiener Deconvolution, Lucy–Richardson Deconvolution.

### 🩹 Photo Repair
Interactive inpainting: paint a mask over damaged/scratched areas, then apply Telea or Navier-Stokes inpainting.

### ✨ Finish
Post-processing: Sharpen, Edge Enhancement, Brightness, Contrast, Gamma, Color Correction, Adaptive Threshold, B&W, Vintage.

### ⚙ Advanced Pipeline
Build a custom ordered pipeline from any combination of all available operations, configure parameters per step, then apply in one pass.

---

## Viewer Features

- **Zoom** — scroll wheel, or toolbar + / − buttons
- **Pan** — drag with mouse when repair mode is off
- **Fit / 1:1** — toolbar buttons
- **Before/After Compare** — split-view slider comparing original vs. current

## Toolbar

- **Open** — load any JPEG, PNG, TIFF, BMP, or WebP image
- **Export** — save as PNG, JPEG, or TIFF
- **Undo / Redo** — full non-destructive editing history
- **Reset** — return to original loaded image
- **Compare** — before/after split-view

---

## Processing Pipeline

```
Load → Analyze → Prepare → Restore → Finish → Compare → Export
```

All processing runs on the **Python backend** using NumPy/OpenCV arrays. The browser UI sends API requests and displays results — no image processing happens in the browser.

## Technology

| Layer | Technology |
|---|---|
| Backend | Python 3.12, Flask, OpenCV, NumPy, SciPy |
| Frontend | Vanilla HTML5, CSS3, JavaScript (no frameworks) |
| Image formats | JPEG, PNG, TIFF, BMP, WebP |

## Project Structure

```
old-photo-restoration-pipeline/
├── server.py              ← Flask web server + all API endpoints
├── templates/index.html   ← Full single-page web UI
├── run_web.sh             ← Startup script
├── core/                  ← Image state, pipeline, restoration engine
├── processing/            ← All DIP algorithms (preprocessing/restoration/postprocessing)
├── models/                ← ImageProfile, ProcessingConfig
├── utils/                 ← Utilities
└── tests/                 ← Unit tests
```
