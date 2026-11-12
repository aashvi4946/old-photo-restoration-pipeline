"""Flask backend for the Old Photo Restorer web application.

Replaces the PySide6 UI layer; all processing/core/models/utils code is used unchanged.
Single-user design: state is kept in module-level globals.
"""

from __future__ import annotations

import base64
import io
import os
import sys
import tempfile
from pathlib import Path

import cv2
import numpy as np
from flask import Flask, jsonify, request, send_file, render_template
from flask_cors import CORS

# ---------------------------------------------------------------------------
# Ensure imports resolve from the project root regardless of CWD
# ---------------------------------------------------------------------------
_PROJECT_ROOT = Path(__file__).parent.resolve()
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from core.image_state import ImageState
from core.pipeline import ProcessingPipeline, PipelineStep
from core.repair_mask_state import RepairMaskState
from core.restoration_engine import RestorationEngine
from models.image_profile import ImageProfile
from processing.analysis.analyze import analyze_image
from processing.postprocessing.black_white import convert_to_black_and_white
from processing.postprocessing.brightness import adjust_brightness
from processing.postprocessing.color_correction import apply_color_correction
from processing.postprocessing.contrast import adjust_contrast
from processing.postprocessing.edge_enhancement import enhance_edges
from processing.postprocessing.gamma import adjust_gamma
from processing.postprocessing.sharpen import sharpen_image
from processing.postprocessing.threshold import apply_adaptive_threshold
from processing.postprocessing.vintage import apply_vintage_style
from processing.preprocessing.clahe import apply_clahe
from processing.preprocessing.histogram import equalize_histogram
from processing.preprocessing.normalization import normalize_image
from processing.preprocessing.resize import resize_image
from processing.preprocessing.white_balance import auto_white_balance
from processing.restoration.deblur import (
    apply_lucy_richardson_deconvolution,
    apply_unsharp_mask,
    apply_wiener_deconvolution,
    create_motion_psf,
)
from processing.restoration.denoise import (
    apply_bilateral_denoise,
    apply_gaussian_denoise,
    apply_median_denoise,
    apply_non_local_means_denoise,
)
from processing.restoration.inpaint import apply_inpainting

# ---------------------------------------------------------------------------
# App setup
# ---------------------------------------------------------------------------
app = Flask(__name__, template_folder="templates")
CORS(app)

# ---------------------------------------------------------------------------
# Global single-user state
# ---------------------------------------------------------------------------
app_state: dict = {
    "image_state": ImageState(),
    "repair_mask_state": RepairMaskState(),
    "restoration_engine": RestorationEngine(),
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _image_to_b64(image: np.ndarray) -> str:
    """Encode a numpy image (BGR or grayscale) as a base64 PNG string."""
    success, buf = cv2.imencode(".png", image)
    if not success:
        raise RuntimeError("Failed to encode image to PNG.")
    return base64.b64encode(buf.tobytes()).decode("utf-8")


def _image_response(image_state: ImageState) -> dict:
    """Build the standard image response dict from the current ImageState."""
    img = image_state.current_image
    if img is None:
        return {"error": "No image loaded."}, 400

    if img.ndim == 2:
        h, w = img.shape
        channels = 1
        is_grayscale = True
    else:
        h, w, channels = img.shape
        is_grayscale = channels == 1 or (
            channels >= 3
            and np.array_equal(img[:, :, 0], img[:, :, 1])
            and np.array_equal(img[:, :, 1], img[:, :, 2])
        )

    return {
        "success": True,
        "width": w,
        "height": h,
        "channels": channels,
        "is_grayscale": bool(is_grayscale),
        "image_b64": _image_to_b64(img),
        "can_undo": image_state.can_undo,
        "can_redo": image_state.can_redo,
    }


def _require_image():
    """Return (image_state, None) or (None, error_response)."""
    image_state: ImageState = app_state["image_state"]
    if not image_state.has_image:
        return None, (jsonify({"error": "No image is currently loaded."}), 400)
    return image_state, None


def _mask_overlay_b64(image: np.ndarray, mask: np.ndarray) -> str:
    """Render binary mask as a clean RGBA PNG (transparent with red damaged regions)."""
    h, w = mask.shape[:2] if mask is not None else image.shape[:2]
    rgba = np.zeros((h, w, 4), dtype=np.uint8)
    if mask is not None:
        # BGRA in OpenCV: B=30, G=30, R=255, Alpha=255
        rgba[mask > 0] = [30, 30, 255, 255]
    success, buf = cv2.imencode(".png", rgba)
    if not success:
        raise RuntimeError("Failed to encode mask overlay.")
    return base64.b64encode(buf.tobytes()).decode("utf-8")


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.route("/")
def index():
    return render_template("index.html")


@app.route("/api/status", methods=["GET"])
def api_status():
    """Health-check endpoint used by the UI to confirm the server is running."""
    image_state: ImageState = app_state["image_state"]
    return jsonify({
        "ok": True,
        "has_image": image_state.has_image,
        "can_undo": image_state.can_undo,
        "can_redo": image_state.can_redo,
    })


# 1. /api/open
@app.route("/api/open", methods=["POST"])
def api_open():
    if "file" not in request.files:
        return jsonify({"error": "No file provided."}), 400

    file = request.files["file"]
    if file.filename == "":
        return jsonify({"error": "Empty filename."}), 400

    try:
        file_bytes = np.frombuffer(file.read(), dtype=np.uint8)
        image = cv2.imdecode(file_bytes, cv2.IMREAD_UNCHANGED)
        if image is None:
            return jsonify({"error": "Could not decode image file."}), 400

        # Convert RGBA → BGR for consistency (keep alpha if desired, but most
        # processing functions handle BGRA; we store as-is).
        image_state: ImageState = app_state["image_state"]
        image_state.set_image(image)

        repair_mask_state: RepairMaskState = app_state["repair_mask_state"]
        h = image.shape[0]
        w = image.shape[1]
        repair_mask_state.set_image_shape(h, w)

        return jsonify(_image_response(image_state))
    except Exception as exc:
        return jsonify({"error": str(exc)}), 500


# 2. /api/undo
@app.route("/api/undo", methods=["POST"])
def api_undo():
    image_state, err = _require_image()
    if err:
        return err
    try:
        image_state.undo()
        return jsonify(_image_response(image_state))
    except Exception as exc:
        return jsonify({"error": str(exc)}), 500


# 3. /api/redo
@app.route("/api/redo", methods=["POST"])
def api_redo():
    image_state, err = _require_image()
    if err:
        return err
    try:
        image_state.redo()
        return jsonify(_image_response(image_state))
    except Exception as exc:
        return jsonify({"error": str(exc)}), 500


# 4. /api/reset
@app.route("/api/reset", methods=["POST"])
def api_reset():
    image_state, err = _require_image()
    if err:
        return err
    try:
        image_state.reset()
        return jsonify(_image_response(image_state))
    except Exception as exc:
        return jsonify({"error": str(exc)}), 500


# 5. /api/export
@app.route("/api/export", methods=["POST"])
def api_export():
    image_state, err = _require_image()
    if err:
        return err

    data = request.get_json(silent=True) or {}
    fmt = data.get("format", "png").lower()

    ext_map = {"png": ".png", "jpg": ".jpg", "jpeg": ".jpg", "tiff": ".tiff", "tif": ".tiff"}
    mime_map = {"png": "image/png", "jpg": "image/jpeg", "jpeg": "image/jpeg",
                "tiff": "image/tiff", "tif": "image/tiff"}

    if fmt not in ext_map:
        return jsonify({"error": f"Unsupported format: {fmt}. Use png, jpg, or tiff."}), 400

    try:
        img = image_state.current_image
        ext = ext_map[fmt]
        success, buf = cv2.imencode(ext, img)
        if not success:
            return jsonify({"error": "Failed to encode image."}), 500

        return send_file(
            io.BytesIO(buf.tobytes()),
            mimetype=mime_map[fmt],
            as_attachment=True,
            download_name=f"restored{ext}",
        )
    except Exception as exc:
        return jsonify({"error": str(exc)}), 500


# 6. /api/analyze
@app.route("/api/analyze", methods=["POST"])
def api_analyze():
    image_state, err = _require_image()
    if err:
        return err
    try:
        profile: ImageProfile = analyze_image(image_state.current_image)
        return jsonify({
            "width": profile.width,
            "height": profile.height,
            "channels": profile.channels,
            "is_grayscale": profile.is_grayscale,
            "blur_score": profile.blur_score,
            "blur_level": profile.blur_level,
            "noise_score": profile.noise_score,
            "noise_level": profile.noise_level,
            "brightness": profile.brightness,
            "contrast": profile.contrast,
            "color_cast": profile.color_cast,
        })
    except Exception as exc:
        return jsonify({"error": str(exc)}), 500


# 7. /api/prepare
@app.route("/api/prepare", methods=["POST"])
def api_prepare():
    image_state, err = _require_image()
    if err:
        return err

    data = request.get_json(silent=True) or {}
    operation = data.get("operation", "")
    params = data.get("params", {})

    try:
        img = image_state.current_image
        result = None

        if operation == "histogram":
            result = equalize_histogram(img)
        elif operation == "clahe":
            clip_limit = float(params.get("clip_limit", 2.0))
            tgs = params.get("tile_grid_size", [8, 8])
            tile_grid_size = (int(tgs[0]), int(tgs[1]))
            result = apply_clahe(img, clip_limit=clip_limit, tile_grid_size=tile_grid_size)
        elif operation == "white_balance":
            result = auto_white_balance(img)
        elif operation == "normalize":
            new_min = int(params.get("new_min", 0))
            new_max = int(params.get("new_max", 255))
            result = normalize_image(img, new_min, new_max)
        elif operation == "resize":
            width = int(params.get("width", img.shape[1]))
            height = int(params.get("height", img.shape[0]))
            result = resize_image(img, width, height)
        else:
            return jsonify({"error": f"Unknown prepare operation: {operation}"}), 400

        image_state.apply_change(result)
        return jsonify(_image_response(image_state))
    except Exception as exc:
        return jsonify({"error": str(exc)}), 500


# 8. /api/denoise
@app.route("/api/denoise", methods=["POST"])
def api_denoise():
    image_state, err = _require_image()
    if err:
        return err

    data = request.get_json(silent=True) or {}
    operation = data.get("operation", "")
    params = data.get("params", {})

    try:
        img = image_state.current_image
        result = None

        if operation == "gaussian":
            kernel_size = int(params.get("kernel_size", 5))
            sigma = float(params.get("sigma", 0.0))
            result = apply_gaussian_denoise(img, kernel_size=kernel_size, sigma=sigma)
        elif operation == "median":
            kernel_size = int(params.get("kernel_size", 3))
            result = apply_median_denoise(img, kernel_size=kernel_size)
        elif operation == "bilateral":
            diameter = int(params.get("diameter", 9))
            sigma_color = float(params.get("sigma_color", 75.0))
            sigma_space = float(params.get("sigma_space", 75.0))
            result = apply_bilateral_denoise(img, diameter=diameter, sigma_color=sigma_color, sigma_space=sigma_space)
        elif operation == "non_local_means":
            h = float(params.get("h", 10.0))
            h_color = float(params.get("h_color", 10.0))
            template_window_size = int(params.get("template_window_size", 7))
            search_window_size = int(params.get("search_window_size", 21))
            result = apply_non_local_means_denoise(
                img,
                h=h,
                h_color=h_color,
                template_window_size=template_window_size,
                search_window_size=search_window_size,
            )
        else:
            return jsonify({"error": f"Unknown denoise operation: {operation}"}), 400

        image_state.apply_change(result)
        return jsonify(_image_response(image_state))
    except Exception as exc:
        return jsonify({"error": str(exc)}), 500


# 9. /api/deblur
@app.route("/api/deblur", methods=["POST"])
def api_deblur():
    image_state, err = _require_image()
    if err:
        return err

    data = request.get_json(silent=True) or {}
    operation = data.get("operation", "")
    params = data.get("params", {})

    try:
        img = image_state.current_image
        result = None

        if operation == "unsharp":
            radius = float(params.get("radius", 1.0))
            amount = float(params.get("amount", 1.0))
            threshold = int(params.get("threshold", 0))
            result = apply_unsharp_mask(img, radius=radius, amount=amount, threshold=threshold)
        elif operation == "wiener":
            psf_length = int(params.get("psf_length", 9))
            psf_angle = float(params.get("psf_angle", 0.0))
            balance = float(params.get("balance", 0.01))
            psf = create_motion_psf(psf_length, psf_angle)
            result = apply_wiener_deconvolution(img, psf, balance=balance)
        elif operation == "lucy_richardson":
            psf_length = int(params.get("psf_length", 9))
            psf_angle = float(params.get("psf_angle", 0.0))
            iterations = int(params.get("iterations", 10))
            psf = create_motion_psf(psf_length, psf_angle)
            result = apply_lucy_richardson_deconvolution(img, psf, iterations=iterations)
        else:
            return jsonify({"error": f"Unknown deblur operation: {operation}"}), 400

        image_state.apply_change(result)
        return jsonify(_image_response(image_state))
    except Exception as exc:
        return jsonify({"error": str(exc)}), 500


# 10. /api/repair/mask
@app.route("/api/repair/mask", methods=["POST"])
def api_repair_mask():
    image_state, err = _require_image()
    if err:
        return err

    data = request.get_json(silent=True) or {}
    action = data.get("action", "")
    repair_mask_state: RepairMaskState = app_state["repair_mask_state"]

    try:
        if action == "paint":
            points = data.get("points", [])
            brush_size = int(data.get("brush_size", 10))
            erase = bool(data.get("erase", False))

            repair_mask_state.begin_stroke()
            if len(points) >= 1:
                first_pt = (int(points[0][0]), int(points[0][1]))
                repair_mask_state.paint_line(first_pt, first_pt, brush_size, erase=erase)
            for i in range(len(points) - 1):
                start = (int(points[i][0]), int(points[i][1]))
                end = (int(points[i + 1][0]), int(points[i + 1][1]))
                repair_mask_state.paint_line(start, end, brush_size, erase=erase)
            repair_mask_state.end_stroke()

        elif action == "clear":
            repair_mask_state.clear_all()
        elif action == "undo":
            repair_mask_state.undo()
        elif action == "redo":
            repair_mask_state.redo()
        else:
            return jsonify({"error": f"Unknown mask action: {action}"}), 400

        mask_b64 = _mask_overlay_b64(image_state.current_image, repair_mask_state.mask)
        return jsonify({"mask_b64": mask_b64})
    except Exception as exc:
        return jsonify({"error": str(exc)}), 500


# 11. /api/repair/apply
@app.route("/api/repair/apply", methods=["POST"])
def api_repair_apply():
    image_state, err = _require_image()
    if err:
        return err

    data = request.get_json(silent=True) or {}
    method = data.get("method", "telea")
    radius = float(data.get("radius", 3.0))

    repair_mask_state: RepairMaskState = app_state["repair_mask_state"]
    if not repair_mask_state.has_damage:
        return jsonify({"error": "No repair mask painted. Paint a mask before applying."}), 400

    try:
        result = apply_inpainting(image_state.current_image, repair_mask_state.mask, method=method, radius=radius)
        image_state.apply_change(result)
        repair_mask_state.reset()
        return jsonify(_image_response(image_state))
    except Exception as exc:
        return jsonify({"error": str(exc)}), 500


# 12. /api/finish
@app.route("/api/finish", methods=["POST"])
def api_finish():
    image_state, err = _require_image()
    if err:
        return err

    data = request.get_json(silent=True) or {}
    operation = data.get("operation", "")
    params = data.get("params", {})

    try:
        img = image_state.current_image
        result = None

        if operation == "sharpen":
            amount = float(params.get("amount", 1.0))
            result = sharpen_image(img, amount=amount)
        elif operation == "edge_enhancement":
            strength = float(params.get("strength", 0.5))
            result = enhance_edges(img, strength=strength)
        elif operation == "brightness":
            value = float(params.get("value", 0))
            result = adjust_brightness(img, value=value)
        elif operation == "contrast":
            factor = float(params.get("factor", 1.0))
            result = adjust_contrast(img, factor=factor)
        elif operation == "gamma":
            gamma = float(params.get("gamma", 1.0))
            result = adjust_gamma(img, gamma=gamma)
        elif operation == "color_correction":
            blue_gain = float(params.get("blue_gain", 1.0))
            green_gain = float(params.get("green_gain", 1.0))
            red_gain = float(params.get("red_gain", 1.0))
            result = apply_color_correction(img, blue_gain=blue_gain, green_gain=green_gain, red_gain=red_gain)
        elif operation == "black_white":
            result = convert_to_black_and_white(img)
        elif operation == "vintage":
            intensity = float(params.get("intensity", 0.35))
            result = apply_vintage_style(img, intensity=intensity)
        elif operation == "adaptive_threshold":
            block_size = int(params.get("block_size", 11))
            constant = float(params.get("constant", 2.0))
            method = params.get("method", "gaussian")
            result = apply_adaptive_threshold(img, block_size=block_size, constant=constant, method=method)
        else:
            return jsonify({"error": f"Unknown finish operation: {operation}"}), 400

        image_state.apply_change(result)
        return jsonify(_image_response(image_state))
    except Exception as exc:
        return jsonify({"error": str(exc)}), 500


# 13. /api/quick_restore
@app.route("/api/quick_restore", methods=["POST"])
def api_quick_restore():
    image_state, err = _require_image()
    if err:
        return err

    data = request.get_json(silent=True) or {}
    # Accept both capitalised ("Medium") and lowercase ("medium") from client
    raw_strength = data.get("strength", "Medium")
    raw_style = data.get("style", "Natural")
    # Normalise to title-case for the engine
    strength = raw_strength.strip().title()
    # Map style values: "bw" → "B&W", others title-case
    _style_map = {"bw": "B&W", "b&w": "B&W", "natural": "Natural", "vintage": "Vintage"}
    style = _style_map.get(raw_style.strip().lower(), raw_style.strip().title())

    try:
        engine: RestorationEngine = app_state["restoration_engine"]
        profile = analyze_image(image_state.current_image)
        plan = engine.build_plan(profile, strength=strength, style=style)
        result = engine.execute(image_state.current_image, plan)
        image_state.apply_change(result)

        report = plan.report

        response = _image_response(image_state)
        response["report"] = report
        response["width"] = profile.width
        response["height"] = profile.height
        response["channels"] = profile.channels
        response["is_grayscale"] = profile.is_grayscale
        response["blur_score"] = profile.blur_score
        response["blur_level"] = profile.blur_level
        response["noise_score"] = profile.noise_score
        response["noise_level"] = profile.noise_level
        response["brightness"] = profile.brightness
        response["contrast"] = profile.contrast
        response["color_cast"] = profile.color_cast
        return jsonify(response)
    except Exception as exc:
        return jsonify({"error": str(exc)}), 500


# 14. /api/advanced
@app.route("/api/advanced", methods=["POST"])
def api_advanced():
    image_state, err = _require_image()
    if err:
        return err

    data = request.get_json(silent=True) or {}
    steps_data = data.get("steps", [])

    if not steps_data:
        return jsonify({"error": "No pipeline steps provided."}), 400

    try:
        pipeline = ProcessingPipeline()
        for step_def in steps_data:
            operation = step_def.get("operation", "")
            parameters = step_def.get("parameters", {})
            pipeline._steps.append(PipelineStep(operation=operation, parameters=parameters))

        repair_mask_state: RepairMaskState = app_state["repair_mask_state"]
        mask = repair_mask_state.mask if repair_mask_state.has_mask else None

        result = pipeline.execute(image_state.current_image, repair_mask=mask)
        image_state.apply_change(result)
        return jsonify(_image_response(image_state))
    except Exception as exc:
        return jsonify({"error": str(exc)}), 500


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    os.chdir(_PROJECT_ROOT)
    port = int(os.environ.get("PORT", 8080))
    app.run(debug=True, host="0.0.0.0", port=port)
