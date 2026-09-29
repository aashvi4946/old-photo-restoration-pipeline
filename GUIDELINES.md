# OldPhoto Restore — Project Guidelines

## 1. Project Overview

**OldPhoto Restore** is a macOS desktop application for restoring degraded historical/old photographs using **classical Digital Image Processing (DIP)** techniques.

The application helps a user take an old photograph that may contain:

- Blur
- Noise
- Low contrast
- Poor brightness
- Color casts/fading
- Scratches
- Small damaged/missing regions

and produce a cleaner, enhanced, and partially repaired version.

The project is intentionally **non-AI for the initial version**. No machine-learning or generative restoration models should be added unless explicitly requested later.

The architecture must, however, remain modular enough that AI-based restoration can be added later without redesigning the application.

---

## 2. Project Goal

The goal is not to create a collection of unrelated OpenCV demonstrations.

The goal is to create a **coherent image-restoration product** with:

1. Automatic image-quality analysis
2. Guided restoration
3. Manual/advanced restoration controls
4. Interactive damaged-area repair
5. Before/after comparison
6. Non-destructive editing with undo/redo
7. High-quality export

The user should be able to upload an old photograph, understand its detected problems, choose how much control they want, restore/refine the image, compare the result with the original, and export the final image.

---

# 3. Core User Experience

The user-facing application has three primary modes:

### A. Quick Restore

For users who do not want to understand individual image-processing algorithms.

Flow:

    Open Image
        ↓
    Automatic Analysis
        ↓
    Select Restoration Strength
        ↓
    Select Style
        ↓
    Restore
        ↓
    Compare Original / Restored
        ↓
    Optional Refinement
        ↓
    Export

Restoration strengths:

- Light
- Medium
- Strong

Styles:

- Natural
- Vintage
- B&W

The exact algorithms used are selected by a rule-based decision engine according to the image analysis results and selected restoration strength.

---

### B. Photo Repair

For photographs with visible scratches, tears, spots, or other localized damage.

Flow:

    Open Image
        ↓
    Analyze
        ↓
    Optional Denoising / Preparation
        ↓
    Enter Repair Mode
        ↓
    User paints over damaged regions
        ↓
    Application creates a binary damage mask
        ↓
    User chooses Telea or Navier-Stokes
        ↓
    Apply Inpainting
        ↓
    Inspect Result
        ↓
    Finish / Enhance
        ↓
    Export

The user should be able to control the repair mask with a brush, erase mistakes, clear the mask, undo strokes, and change brush size.

---

### C. Advanced

For users who want direct control over individual DIP operations.

The user can manually construct a processing pipeline such as:

    CLAHE
      ↓
    Non-Local Means
      ↓
    Unsharp Mask
      ↓
    Gamma
      ↓
    Contrast

Advanced mode should expose individual processing techniques and appropriate parameters without forcing those controls onto normal users.

---

# 4. Internal Processing Architecture

The user-facing modes are different ways of interacting with the same underlying processing engine.

Internally, the application follows:

    LOAD
      ↓
    ANALYZE
      ↓
    PREPARE
      ↓
    RESTORE
      ↓
    FINISH
      ↓
    COMPARE
      ↓
    EXPORT

### Prepare

Responsible for preparing the image and improving basic image quality.

Possible operations:

- Histogram Equalization
- CLAHE
- White Balance
- Resizing
- Appropriate normalization
- Optional mild preprocessing/denoising

### Restore

Responsible for recovering or improving degraded information.

Possible operations:

- Gaussian filtering
- Median filtering
- Bilateral filtering
- Non-Local Means denoising
- Unsharp Masking
- Wiener filtering
- Lucy-Richardson deconvolution
- Telea inpainting
- Navier-Stokes inpainting
- Controlled sharpening

### Finish

Responsible for final visual refinement.

Possible operations:

- Sharpening
- Edge enhancement
- Brightness
- Contrast
- Gamma correction
- Color correction
- B&W conversion
- Vintage styling
- Adaptive thresholding where appropriate

---

# 5. Image Analysis

Image analysis is a first-class component of the application.

When an image is loaded, the application should analyze it and create an `ImageProfile`.

The analysis should include, where applicable:

- Image dimensions
- Number of channels
- Color/grayscale status
- Blur score
- Blur classification
- Noise estimate
- Noise classification
- Brightness
- Contrast
- Histogram information
- Color cast

### Blur Detection

Use **Laplacian variance** as the primary classical blur metric.

Do not treat arbitrary thresholds as universal truth. Thresholds should be configurable and validated against representative sample images.

### Noise Analysis

Estimate image noise using a classical image-processing approach. The resulting value should be used for classification such as Low / Medium / High.

### Analysis Result

Conceptually:

    ImageProfile
        ├── width
        ├── height
        ├── channels
        ├── is_grayscale
        ├── blur_score
        ├── blur_level
        ├── noise_score
        ├── noise_level
        ├── brightness
        ├── contrast
        └── color_cast

The analysis layer must not directly manipulate the GUI.

---

# 6. Rule-Based Quick Restore

Quick Restore is **not AI**.

It uses deterministic, explainable rules based on the ImageProfile.

Example logic:

    IF noise is HIGH
        → use stronger denoising

    IF noise is MEDIUM
        → use moderate denoising

    IF blur is HIGH
        → use stronger deblurring

    IF blur is MEDIUM
        → use mild/moderate sharpening or deblurring

    IF contrast is LOW
        → use CLAHE or suitable contrast enhancement

    IF color cast is detected
        → use white balance/color correction

    AFTER restoration
        → apply controlled final sharpening if appropriate

The exact combinations and thresholds must be tested rather than assumed to be universally optimal.

The engine should produce an optional human-readable restoration report, e.g.:

    RESTORATION REPORT

    ✓ Noise reduction
      Non-Local Means

    ✓ Contrast enhancement
      CLAHE

    ✓ Color correction
      White Balance

    ✓ Detail enhancement
      Unsharp Mask

    Skipped:
    ○ Deblurring
      Blur level below configured threshold

This makes the automatic pipeline explainable.

---

# 7. Restoration Strengths

Restoration strength should be implemented through configuration rather than scattered hard-coded conditionals.

Conceptually:

    LIGHT
    MEDIUM
    STRONG

Each level defines:

- Which operations are enabled
- Their default strengths
- Their ordering
- Which analysis conditions activate them

Possible conceptual behavior:

### Light

- Mild color correction
- Mild contrast enhancement
- Mild sharpening

### Medium

- White balance when needed
- Moderate denoising
- CLAHE when appropriate
- Deblurring when needed
- Controlled sharpening
- Contrast correction

### Strong

- Stronger denoising
- Deblurring when needed
- CLAHE
- Color correction
- Sharpening
- Final contrast refinement

These are implementation starting points, not fixed scientific rules. Validate them with sample images.

---

# 8. Styles

Styles are applied after restoration/refinement.

### Natural

Preserve realistic color and tonal appearance.

### Vintage

Use controlled warm tones, saturation/contrast adjustments, and optional subtle effects.

### B&W

Convert to grayscale and apply suitable tonal/contrast refinement.

Styles must not destroy the restoration work that occurred before them.

---

# 9. Photo Repair / Inpainting

Inpainting is a guided operation, not something that should automatically run on every photograph.

The user identifies the damaged area.

The application creates a mask:

    0   = preserve
    255 = repair

Supported methods:

- Telea
- Navier-Stokes

The repair UI should support:

- Brush
- Brush size
- Eraser
- Clear mask
- Undo stroke
- Redo stroke where practical
- Zoom
- Pan
- Apply Inpainting
- Preview mask before applying

The mask itself should be maintained separately from the image state.

---

# 10. Advanced Manual Processing

Advanced mode exposes individual operations from the processing engine.

## Prepare

- Histogram Equalization
- CLAHE
- White Balance
- Resize
- Optional preprocessing operations

## Restore

- Gaussian
- Median
- Bilateral
- Non-Local Means
- Unsharp Mask
- Wiener
- Lucy-Richardson
- Inpainting

## Finish

- Sharpen
- Edge Enhancement
- Brightness
- Contrast
- Gamma
- Color Correction
- Adaptive Threshold
- B&W
- Vintage

The UI should use progressive disclosure:

- Show common controls first.
- Put complex technical parameters behind an advanced/parameter section.
- Do not overwhelm the Quick Restore workflow with algorithm-specific controls.

---

# 11. Pipeline Architecture

Processing algorithms must be implemented independently from the GUI.

A generic processing operation should conceptually follow:

    input image
        ↓
    operation
        ↓
    output image

The application should support constructing pipelines:

    pipeline = [
        WhiteBalance(...),
        NLM(...),
        CLAHE(...),
        UnsharpMask(...)
    ]

Then:

    result = pipeline.execute(image)

The same pipeline infrastructure must support:

- Quick Restore
- Photo Repair
- Advanced Manual mode

The GUI must not contain the implementation of image-processing algorithms.

---

# 12. Image State Management

The application must use non-destructive editing.

Maintain:

    original_image
    current_image
    undo_stack
    redo_stack

Rules:

1. Never overwrite the original image.
2. Before an operation modifies the current image, save the previous state.
3. A new operation clears the redo stack.
4. Undo moves the current state to redo and restores the previous state.
5. Redo restores a state from redo.
6. Reset returns to the original image and clears editing history.
7. Processing failures must not corrupt the last valid image.

Later, history entries can include operation names and parameters.

Example:

    Original
      ↓
    CLAHE
      ↓
    NLM
      ↓
    Unsharp Mask
      ↓
    Gamma 0.9

---

# 13. Image Viewer

The image viewer is responsible only for visualization.

It should support:

- Fit to window
- Zoom in
- Zoom out
- 100% view
- Pan
- Refresh
- Original/current display
- Before/after display
- Split comparison
- Before/after slider

The processing layer must not depend on GUI-specific image objects.

Keep image-processing images as NumPy/OpenCV-compatible arrays and convert them only at the display boundary.

---

# 14. Before/After Comparison

The comparison system should support:

### Side-by-side

    ORIGINAL | RESTORED

### Split view

    ORIGINAL | RESTORED
              ↑
          draggable divider

### Toggle

    Original ↔ Restored

The split slider is especially useful for demonstrations.

---

# 15. Performance Architecture

Large old photographs may be several thousand pixels wide.

The application should distinguish between:

### Preview image

Used for interactive UI updates.

Example:

    6000 × 4000 original
            ↓
    1200 × 800 preview

### Full-resolution image

Used for final export.

Interactive operations can use the preview image where appropriate.

When exporting:

    saved pipeline/settings
            ↓
    original full-resolution image
            ↓
    full-resolution processing
            ↓
    export

Do not permanently replace the original with a downscaled preview.

---

# 16. Background Processing

Expensive operations must not freeze the PySide6 UI.

Potentially expensive operations include:

- Non-Local Means
- Wiener filtering
- Lucy-Richardson
- Large-image inpainting
- Full-resolution pipelines

Use Qt worker/thread mechanisms for expensive processing.

The GUI should show a progress state:

    Restoring image...

    ████████░░ 80%

The processing worker must communicate results/errors back to the UI safely.

---

# 17. Caching

Where practical, avoid recomputing expensive operations unnecessarily.

For example:

    Original
      ↓
    NLM
      ↓
    CLAHE
      ↓
    Sharpen
          ↓
       cached

If only brightness changes afterward, the expensive earlier stages should not automatically rerun if the architecture can safely reuse their result.

Caching must never compromise correctness.

---

# 18. Export

Supported initial formats:

- JPEG
- PNG
- TIFF

JPEG should support a quality setting.

Export should preserve the selected final resolution unless the user explicitly requests resizing.

The export system should:

1. Validate the current image.
2. Apply the final full-resolution pipeline when required.
3. Validate output dimensions.
4. Encode using the requested format.
5. Save to the selected path.
6. Report success/failure clearly.

---

# 19. Error Handling

The application must gracefully handle:

- Invalid image files
- Unsupported formats
- Corrupt images
- No image loaded
- Invalid parameters
- Empty inpainting masks
- Processing failures
- Export failures
- Very large images
- Memory-related issues where detectable

Never allow an algorithm exception to silently corrupt the current image.

Prefer:

    processing failed
    previous valid state preserved

over:

    application crash / invalid image state

---

# 20. Project Structure

Recommended structure:

    old-photo-restorer/
    │
    ├── app.py
    ├── requirements.txt
    ├── README.md
    ├── .gitignore
    │
    ├── ui/
    │   ├── main_window.py
    │   ├── image_viewer.py
    │   ├── toolbar.py
    │   ├── analysis_panel.py
    │   ├── quick_restore_panel.py
    │   ├── repair_panel.py
    │   ├── advanced_panel.py
    │   └── widgets/
    │
    ├── core/
    │   ├── image_state.py
    │   ├── image_analyzer.py
    │   ├── pipeline.py
    │   ├── pipeline_manager.py
    │   └── restoration_engine.py
    │
    ├── processing/
    │   ├── preprocessing/
    │   │   ├── histogram.py
    │   │   ├── clahe.py
    │   │   ├── white_balance.py
    │   │   └── resize.py
    │   │
    │   ├── restoration/
    │   │   ├── denoise.py
    │   │   ├── deblur.py
    │   │   ├── wiener.py
    │   │   ├── lucy_richardson.py
    │   │   ├── sharpen.py
    │   │   └── inpaint.py
    │   │
    │   └── postprocessing/
    │       ├── brightness.py
    │       ├── contrast.py
    │       ├── gamma.py
    │       ├── color_correction.py
    │       ├── edge_enhancement.py
    │       └── threshold.py
    │
    ├── models/
    │   ├── image_profile.py
    │   └── processing_config.py
    │
    ├── utils/
    │   ├── image_utils.py
    │   ├── export.py
    │   └── constants.py
    │
    ├── tests/
    │
    ├── assets/
    │   ├── icons/
    │   └── styles/
    │
    ├── sample_images/
    │
    └── output/

Keep modules focused. Do not create a single large Python file containing the GUI and all processing algorithms.

---

# 21. Separation of Responsibilities

Follow these rules strictly.

### UI layer

Responsible for:

- Buttons
- Sliders
- Dialogs
- Display
- User input
- Progress state
- Error messages

The UI should NOT implement OpenCV algorithms.

### Core layer

Responsible for:

- Image state
- Analysis
- Pipeline execution
- Restoration decision logic
- Undo/redo coordination

### Processing layer

Responsible for:

- Actual image-processing algorithms

Processing modules should be callable without launching the GUI.

### Models

Responsible for structured data/configuration.

### Utilities

Responsible for reusable non-domain-specific helpers.

---

# 22. No AI in Initial Version

Do not introduce:

- LLMs
- neural networks
- diffusion models
- generative restoration
- face restoration models
- super-resolution models
- AI colorization

unless explicitly requested.

The initial project is a **classical Digital Image Processing project**.

Future AI functionality may be added as an optional restoration provider without changing the overall application architecture.

Potential future structure:

    processing/
        restoration/
            classical/
            ai/

But do not implement the AI portion now.

---

# 23. Development Methodology

This project is intentionally being developed **phase by phase with human supervision**.

The coding agent must NOT attempt to implement the entire project automatically.

The user will explicitly provide the phase/task to implement.

For each requested phase:

1. Inspect the existing project.
2. Understand the current architecture.
3. Implement only the requested scope.
4. Avoid prematurely implementing future phases.
5. Preserve existing functionality.
6. Run relevant tests/checks.
7. Report exactly what changed.
8. Report files created/modified.
9. Report commands/tests run.
10. Identify any issues or assumptions.
11. Stop after completing the requested phase.

Do not assume permission to redesign unrelated parts of the project.

---

# 24. Coding Agent Behavior

When asked to implement something:

### Before coding

- Inspect the repository.
- Inspect existing files relevant to the task.
- Identify the current architecture.
- Do not overwrite existing work without understanding it.
- Check whether the requested functionality already exists.

### During coding

- Follow the existing architecture.
- Keep processing logic separate from UI.
- Prefer small reusable functions/classes.
- Avoid unnecessary dependencies.
- Do not add AI.
- Do not introduce backend/database/cloud infrastructure.
- Do not implement future phases unless required by the current phase.
- Preserve existing functionality.

### After coding

Run appropriate checks.

At minimum, where applicable:

    python -m pytest

and/or an application launch check.

Report:

    Implemented:
    - ...

    Modified:
    - ...

    Tests:
    - ...

    Known issues:
    - ...

---

# 25. Phase Development Order

The project should be developed in this order.

## Phase 0 — Setup

- macOS Python environment
- virtual environment
- dependencies
- Git
- folder structure
- basic application launch

## Phase 1 — UI Foundation

- PySide6 main window
- menus
- toolbar
- navigation
- image viewer
- empty Quick Restore / Photo Repair / Advanced panels

## Phase 2 — Image State

- image loading
- validation
- original/current images
- undo
- redo
- reset
- operation state management

## Phase 3 — Image Analysis

- blur detection
- noise estimation
- brightness
- contrast
- histogram
- color cast
- ImageProfile
- analysis panel

## Phase 4 — Prepare

- histogram equalization
- CLAHE
- white balance
- resizing
- normalization where appropriate

## Phase 5 — Denoising

- Gaussian
- Median
- Bilateral
- Non-Local Means
- parameter handling
- denoising tests

## Phase 6 — Deblurring

- Unsharp Mask
- Wiener
- Lucy-Richardson
- PSF handling
- deblurring tests

## Phase 7 — Photo Repair

- repair canvas
- brush
- eraser
- mask
- mask undo/clear
- Telea
- Navier-Stokes
- repair workflow

## Phase 8 — Finish

- sharpening
- edge enhancement
- brightness
- contrast
- gamma
- color correction
- B&W
- Vintage

## Phase 9 — Quick Restore

- rule-based decision engine
- Light / Medium / Strong
- Natural / Vintage / B&W
- automatic pipeline construction
- restoration report

## Phase 10 — Advanced

- operation browser
- parameter controls
- manual pipeline construction
- pipeline ordering
- operation removal
- advanced parameters

## Phase 11 — UX

- before/after
- split slider
- zoom
- pan
- history display
- keyboard shortcuts
- improved status messages

## Phase 12 — Export

- JPEG
- PNG
- TIFF
- JPEG quality
- full-resolution export
- export validation

## Phase 13 — Reliability

- error handling
- validation
- automated tests
- regression tests

## Phase 14 — Performance

- preview resolution
- background workers
- progress indicators
- caching where appropriate
- memory handling

## Phase 15 — macOS Packaging

- PyInstaller
- `.app`
- application icon
- native testing
- clean-machine testing

## Phase 16 — Final Evaluation

- representative restoration cases
- screenshots
- before/after results
- algorithm comparison
- limitations
- project documentation
- presentation/demo preparation

---

# 26. Testing Philosophy

Do not judge algorithms only by whether the application runs.

For representative images, inspect:

- Noise reduction
- Detail preservation
- Edge preservation
- Contrast
- Color accuracy
- Haloing
- Oversharpening
- Artifacts
- Inpainting quality

Classical restoration involves trade-offs. A method that removes more noise can also remove details. A stronger sharpening operation can create halos. A deconvolution method can amplify noise.

The goal is not "maximum processing."

The goal is:

> **appropriate restoration with controlled artifacts.**

---

# 27. Important Design Principles

### Principle 1 — Non-destructive editing

Never destroy the original image.

### Principle 2 — Separation of concerns

GUI ≠ processing algorithm.

### Principle 3 — Explainable restoration

Quick Restore should be rule-based and explainable.

### Principle 4 — Progressive complexity

Simple users see simple controls. Advanced users can access technical controls.

### Principle 5 — Test before integrating

Every processing algorithm should work independently before being added to Quick Restore.

### Principle 6 — Don't over-process

Restoration should improve the image without unnecessarily changing its character.

### Principle 7 — Preserve resolution

Preview processing and final export processing should be treated separately.

### Principle 8 — Keep the architecture extensible

Future AI functionality should be able to plug into the restoration layer without rewriting the UI/core.

---

# 28. What the Project Is NOT

This project is not initially:

- A web application
- A mobile application
- A cloud service
- A database application
- An AI image generator
- A face-recognition system
- A general photo editor

It is:

> **A classical computer-vision/Digital Image Processing desktop application focused specifically on restoration of old/degraded photographs.**

---

# 29. Success Criteria

The project is considered successful when a user can:

1. Open an old photograph.
2. See automatic quality analysis.
3. Understand detected issues.
4. Run Quick Restore without understanding DIP algorithms.
5. Repair scratches/damaged regions manually.
6. Use Advanced mode for individual DIP techniques.
7. Undo/redo changes.
8. Compare original and restored images.
9. Refine the result.
10. Export a high-quality final image.
11. Run the application natively on macOS without the development environment.

The project should demonstrate the relationship between:

    Image Analysis
          ↓
    Digital Image Processing
          ↓
    Restoration
          ↓
    Visual Evaluation
          ↓
    Final Output

---

# 30. Current Scope Boundary

The coding agent should consider the above architecture and feature set as the **target architecture**, but it should only implement the phase explicitly requested by the user.

Do not jump ahead.

The user will guide implementation phase by phase.
