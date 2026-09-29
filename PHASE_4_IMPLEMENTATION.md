# OLD PHOTO RESTORER — PHASE 4 IMPLEMENTATION GUIDE

## Phase 4: Image Preprocessing / Preparation

---

## 1. Phase Overview

Phase 4 introduces the first real image-processing operations into **Old Photo Restorer**.

The goal is to build a clean, reusable **preprocessing / image preparation layer** that improves the input image before later restoration stages such as denoising, deblurring, inpainting, and final enhancement.

Operations introduced in this phase:

1. Histogram Equalization
2. CLAHE
3. Automatic White Balance
4. Basic normalization where appropriate
5. Explicit resizing support where required

These operations must remain independent of denoising, deblurring, inpainting, sharpening, finishing, Quick Restore, Advanced pipeline orchestration, and AI.

---

## 2. Product Context

Old Photo Restorer is being developed as:

> **A classical image-restoration engine wrapped in a polished desktop application.**

The overall processing model is:

```text
Load → Analyze → Prepare → Restore → Finish → Compare → Export
```

Phase 4 implements the **Prepare** stage.

The source project explicitly identifies histogram equalization, CLAHE, automatic white balance, and resizing as preprocessing operations. Blur and noise analysis were addressed in the previous analysis phase.

---

## 3. Phase Objective

By the end of this phase:

- Histogram equalization works.
- CLAHE works.
- Automatic white balance works.
- Safe normalization is available where justified.
- Resizing is implemented as a reusable processing operation.
- Operations are GUI-independent.
- Operations work on NumPy/OpenCV images.
- Operations integrate with `ImageState`.
- Successful operations create undoable states.
- Failed operations do not corrupt the current image.
- The UI can trigger these operations.
- Existing loading, analysis, undo/redo, reset, and display functionality remains intact.

The user should be able to load an image and manually apply preprocessing operations one at a time.

---

## 4. Prerequisites

Phase 4 begins only after Phases 0–3 are complete.

Existing functionality must include:

- PySide6 application launch
- image loading
- image display
- original/current image separation
- undo
- redo
- reset
- image analysis
- existing analysis panel behavior

Do not proceed if previous-phase functionality is broken.

---

## 5. Scope

### In Scope

**Preprocessing**
- Histogram Equalization
- CLAHE
- Automatic White Balance
- Basic intensity normalization
- Resizing

**Integration**
- Processing modules
- ImageState integration
- UI controls
- Undo/redo integration
- Error handling
- Basic operation feedback

**Testing**
- Unit tests
- Image property checks
- Regression checks
- Manual visual verification

### Explicitly Out of Scope

Do NOT implement:

- Gaussian denoising
- Median denoising
- Bilateral filtering
- Non-local means
- Wiener deblurring
- Lucy-Richardson deblurring
- Telea inpainting
- Navier-Stokes inpainting
- advanced sharpening
- Quick Restore
- automatic restoration decisions
- Advanced pipeline builder
- AI / ML / deep learning
- image generation
- cloud processing
- database functionality
- export workflow redesign
- macOS packaging
- later-phase performance architecture

---

## 6. Architecture

Processing functions must be independent of PySide6.

Bad:

```python
def apply_clahe(image, widget):
    ...
```

Good:

```python
def apply_clahe(image, clip_limit=2.0, tile_grid_size=(8, 8)):
    ...
```

The processing layer should only know about NumPy, OpenCV, and processing parameters.

The UI layer calls processing functions and then updates application state.

---

## 7. Expected Structure

Use the existing repository structure and do not restructure unrelated files.

Expected additions:

```text
processing/
└── preprocessing/
    ├── __init__.py
    ├── histogram.py
    ├── clahe.py
    ├── white_balance.py
    └── resize.py
```

If normalization is implemented separately:

```text
processing/preprocessing/normalization.py
```

Tests should follow the existing test organization, for example:

```text
tests/
├── test_histogram.py
├── test_clahe.py
├── test_white_balance.py
└── test_resize.py
```

Follow the repository's actual conventions if they differ.

---

## 8. Image Representation Contract

Maintain the representation established in Phase 2.

Typical color images:

```text
NumPy array
H × W × 3
BGR
uint8
```

Do not silently change the application's internal convention.

Processing should preserve:

- dimensions unless resizing is explicitly requested
- channel count unless intentionally changed
- valid dtype
- valid pixel range

The original image must never be modified in place.

---

## 9. General Processing Contract

Every preprocessing function should:

- accept a NumPy image
- return a new NumPy image
- not mutate its input
- preserve dimensions unless resizing
- preserve expected color representation
- preserve valid pixel ranges
- validate parameters
- raise meaningful errors for invalid input
- contain no GUI dependencies
- avoid unrelated processing
- be deterministic for the same input and parameters

Suggested pattern:

```python
result = processing_function(current_image)
image_state.apply(result)
```

---

# 10. Histogram Equalization

## Purpose

Histogram equalization improves global contrast by redistributing intensity values.

It is useful when an image's intensity distribution is concentrated in a limited range.

The source project explicitly identifies histogram equalization as a preprocessing operation.

### Color Images

Do NOT independently equalize BGR channels because this can produce unnatural colors.

Use a luminance/intensity-oriented approach:

```text
BGR
 ↓
luminance-oriented color space
 ↓
equalize luminance channel
 ↓
BGR
```

A YCrCb-based implementation operating on Y is appropriate.

### Grayscale Images

Use standard histogram equalization directly on the grayscale image.

### Suggested API

```python
def equalize_histogram(image: np.ndarray) -> np.ndarray:
    ...
```

### Validation

Test:

- grayscale
- color
- dark image
- low-contrast image
- already well-distributed image
- invalid input
- empty input where applicable

Verify:

- dimensions unchanged
- channel count appropriate
- dtype valid
- input unchanged

---

# 11. CLAHE

## Purpose

CLAHE means:

> Contrast Limited Adaptive Histogram Equalization

Unlike global histogram equalization, CLAHE operates on local regions and limits contrast amplification.

It is useful when illumination varies across the image.

The source project explicitly lists CLAHE as preprocessing.

### Parameters

Expose:

```text
clip_limit
tile_grid_size
```

Reasonable implementation defaults:

```text
clip_limit = 2.0
tile_grid_size = (8, 8)
```

These are defaults, not universal truths.

### Color Handling

Do not apply CLAHE independently to BGR channels.

Use:

```text
BGR
 ↓
luminance-oriented color space
 ↓
CLAHE on luminance
 ↓
BGR
```

For grayscale, apply CLAHE directly.

### Suggested API

```python
def apply_clahe(
    image: np.ndarray,
    clip_limit: float = 2.0,
    tile_grid_size: tuple[int, int] = (8, 8),
) -> np.ndarray:
    ...
```

Validate:

- `clip_limit > 0`
- tile dimensions are positive
- input is a valid image

### Important Distinction

Do not hide the difference from the user:

```text
Histogram Equalization = global contrast enhancement

CLAHE = local contrast enhancement + contrast limiting
```

Do not automatically apply both.

---

# 12. Automatic White Balance

## Purpose

Old photographs may contain color casts caused by aging, scanning, lighting, and degradation.

Automatic white balance attempts to correct overall color balance.

The source project explicitly includes automatic white balance.

### Initial Approach

Use a deterministic classical method such as the **gray-world assumption**.

Conceptually:

```text
mean_B = average(B)
mean_G = average(G)
mean_R = average(R)

target = average(mean_B, mean_G, mean_R)

scale_B = target / mean_B
scale_G = target / mean_G
scale_R = target / mean_R
```

Then apply the scales and clip to the valid range.

Protect against division by zero.

### Suggested API

```python
def auto_white_balance(image: np.ndarray) -> np.ndarray:
    ...
```

Requirements:

- color images supported
- grayscale images should return an unchanged copy or be explicitly rejected with a clear reason
- no in-place mutation
- output remains valid `uint8` if that is the application's standard
- clipping is handled safely

### Limitation

Do not claim white balance will correctly restore every old photograph.

Gray-world methods can perform poorly when the scene contains a strong dominant color.

---

# 13. Normalization

Treat normalization carefully.

Normalization should only be introduced where it has a clearly documented purpose.

For an 8-bit image:

```text
valid pixel range = 0–255
```

Do not unnecessarily normalize every image before every operation.

Prefer preserving the existing `uint8` representation unless another numerical range is genuinely required.

If implemented, a helper may be:

```python
def normalize_image(
    image: np.ndarray,
    new_min: int = 0,
    new_max: int = 255,
) -> np.ndarray:
    ...
```

Validate:

```text
new_min < new_max
```

Document that this is min-max intensity normalization if that is what is implemented.

Do not automatically call normalization before other preprocessing operations.

---

# 14. Resizing

## Purpose

Resizing belongs to preprocessing because some workflows require a specific working resolution.

However, resizing must be explicit.

Do not silently resize the user's original image merely because it is large.

### Suggested API

```python
def resize_image(
    image: np.ndarray,
    width: int,
    height: int,
    interpolation: int = cv2.INTER_AREA,
) -> np.ndarray:
    ...
```

Validate:

- width > 0
- height > 0

Do not modify the original.

Use appropriate interpolation. `INTER_AREA` is a suitable default for shrinking.

### Aspect Ratio

Avoid accidental distortion.

The UI should preferably provide:

```text
☑ Maintain aspect ratio
```

as the default.

If exact width and height are supported, allow intentional disabling of aspect-ratio preservation.

Do not silently distort photographs.

---

# 15. UI Integration

Phase 4 is the first phase where processing controls become functional.

Add a **Prepare / Preprocessing** section to the existing UI without redesigning the entire application.

Suggested controls:

```text
PREPARE

[ Histogram Equalization ]

[ CLAHE ]

[ Auto White Balance ]

[ Normalize ]

[ Resize... ]
```

The exact visual styling must match the existing application.

Avoid unrelated floating windows.

---

# 16. CLAHE Configuration

Because CLAHE has meaningful parameters, provide a compact configuration UI.

Example:

```text
CLAHE

Clip Limit
[ 2.0 ]

Tile Grid
[ 8 ] × [ 8 ]

[ Apply ]
[ Cancel ]
```

Validate parameters before processing.

Do not expose unnecessary OpenCV internals.

---

# 17. Resize Dialog

A resize dialog may contain:

```text
Width
[ 1200 ]

Height
[ 800 ]

☑ Maintain aspect ratio

[ Apply ]
[ Cancel ]
```

When aspect ratio is locked, changing one dimension should update the other.

Do not modify the image until the user confirms.

---

# 18. Operation Flow

Every preprocessing action should follow:

```text
User clicks operation
        ↓
Optional parameter dialog
        ↓
Validate parameters
        ↓
Run processing function
        ↓
Receive new NumPy image
        ↓
Commit to ImageState
        ↓
Update viewer
        ↓
Update analysis
        ↓
Update undo/redo state
```

The processing function itself must not commit state.

---

# 19. ImageState Integration

A successful operation should behave as one undoable action.

Example:

```text
Original
   ↓
CLAHE
   ↓
White Balance
   ↓
Undo
   ↓
CLAHE result
```

Undo must return the exact previous image.

Redo must restore the exact processed image.

New processing after undo must clear the redo history according to Phase 2 semantics.

---

# 20. Failure Handling

If processing fails:

```text
Current image remains unchanged.
```

Do not:

- replace current image with `None`
- push invalid images onto history
- corrupt original image
- unnecessarily clear redo history

The UI should display a concise error.

---

# 21. Re-analysis

After successful preprocessing:

```text
current image changes
        ↓
analysis refreshes
```

The analysis panel should represent the current image, not stale data.

Do not overwrite original-image analysis merely because the current image changed.

---

# 22. Display vs Processing Resolution

The viewer may use a scaled representation for display.

Processing must operate on the actual current image, not the viewer's display-scaled copy.

Do not accidentally process a preview-sized image and replace the full-resolution image.

---

# 23. Code Quality

Processing modules should:

- contain focused functions
- use type hints
- include concise docstrings
- validate inputs
- avoid duplicated logic
- avoid GUI imports
- avoid global mutable state
- use constants where useful
- remain easy to unit test

Do not create unnecessary abstractions.

---

# 24. Testing

Testing is mandatory.

Verify both:

```text
numerical / structural behavior
```

and:

```text
application behavior
```

Do not rely only on visual inspection.

### Histogram Tests

Test:

- grayscale
- color
- low-contrast image
- dark image
- input immutability
- invalid input

Verify:

- shape preserved
- dtype preserved
- valid range
- color channel structure preserved

### CLAHE Tests

Test:

- grayscale
- color
- default parameters
- custom parameters
- invalid clip limit
- invalid tile size
- input immutability

Verify:

```text
shape preserved
dtype preserved
valid pixel range
```

### White Balance Tests

Create a synthetic color-cast image.

Verify:

- output remains valid
- channel imbalance is reduced
- shape unchanged
- input unchanged

Do not require perfect color reconstruction from gray-world processing.

### Resize Tests

Test:

```text
100 × 100 → 50 × 50
```

Verify expected dimensions.

Also test:

- invalid width
- invalid height
- color image
- grayscale image
- input immutability

---

# 25. ImageState Integration Tests

Verify:

```text
load image
↓
apply preprocessing
↓
undo
↓
redo
```

Expected:

- result appears after apply
- undo restores previous image
- redo restores processed image

Also test:

```text
apply A
apply B
undo
apply C
```

Expected:

```text
redo history is cleared after C
```

---

# 26. Regression Tests

Verify that Phase 0–3 functionality remains intact:

- application starts
- image opens
- image displays
- reset works
- undo works
- redo works
- analysis works
- original image remains unchanged
- invalid images are handled
- existing UI remains usable

---

# 27. Visual Verification Dataset

Use a small local test set containing:

1. normal photograph
2. low-contrast photograph
3. dark photograph
4. color-cast photograph
5. grayscale photograph
6. high-resolution photograph
7. small photograph

Look for obvious artifacts:

- unnatural colors
- excessive contrast
- clipping
- strange halos
- image distortion
- unexpected grayscale conversion

The purpose is to catch obvious problems, not to prove universal algorithmic correctness.

---

# 28. Important Product Principle: Do Not Over-Process

Do NOT automatically run:

```text
Histogram Equalization
+
CLAHE
+
Normalization
+
White Balance
```

Each operation has a specific purpose.

The user or a later deterministic pipeline should decide which operations are appropriate.

This is especially important for old photographs because aggressive processing can create:

- excessive contrast
- clipped highlights/shadows
- unnatural colors
- amplified artifacts

---

# 29. No Quick Restore Yet

Do not add logic such as:

```python
if contrast_is_low:
    apply_clahe()
```

That belongs to the later **Quick Restore** phase.

Phase 4 provides reliable building blocks.

Later phases may use the analyzer to decide which operations are appropriate.

---

# 30. No AI

Absolutely no:

- LLM
- computer vision model
- neural network
- diffusion model
- cloud AI API
- AI image enhancement

Phase 4 is entirely classical image processing.

The architecture should remain compatible with future AI integration, but no AI implementation belongs here.

---

# 31. Coding Agent Workflow

Do NOT ask the agent to implement the entire phase blindly in one pass.

Recommended task order:

```text
Task 1  Inspect repository and Phases 0–3
Task 2  Create preprocessing module structure
Task 3  Implement histogram equalization
Task 4  Implement CLAHE
Task 5  Implement automatic white balance
Task 6  Implement normalization if justified
Task 7  Implement resizing
Task 8  Add unit tests
Task 9  Integrate UI controls
Task 10 Integrate ImageState
Task 11 Run full regression suite
Task 12 Perform manual visual verification
```

The agent must stop after Phase 4 and must not begin Phase 5.

---

# 32. Agent Inspection Requirement

Before writing code, inspect:

- repository tree
- `app.py`
- `core/image_state.py`
- image loading utilities
- image analyzer
- main window
- image viewer
- existing tests
- requirements
- styling

The agent must identify the actual implementation rather than assuming the files exactly match this guide.

---

# 33. Agent Preservation Requirement

Preserve existing functionality.

Do not rewrite Phase 1–3 code unnecessarily.

If an earlier file must be changed, keep the change minimal and report:

```text
File changed:
Reason:
What was preserved:
```

Do not introduce future-phase functionality.

---

# 34. Agent Reporting Format

After each task:

```text
TASK:
<task name>

CHANGED FILES:
- file
- file

IMPLEMENTED:
- item
- item

TESTS:
- command
- result

ISSUES:
- issue or "None"

NEXT:
<next Phase 4 task>
```

The agent must not silently continue through all tasks.

---

# 35. Suggested Agent Prompt — Initial Inspection

```text
You are working on Phase 4 of the Old Photo Restorer project.

Before writing code, inspect the existing repository and the implementation from Phases 0–3.

Do not implement anything yet.

Identify:
- current project structure
- ImageState implementation
- image loading and representation
- image analyzer
- main window architecture
- viewer architecture
- existing test structure
- current dependencies
- existing UI styling

Compare the repository with the Phase 4 implementation guide.

Report:
1. what already exists
2. what Phase 4 needs to add
3. which existing files need modification
4. any architectural conflicts

Do not begin Phase 5 or any AI implementation.
```

---

# 36. Suggested Agent Prompt — Processing Layer

```text
Implement only the Phase 4 preprocessing processing layer.

Start with histogram equalization and CLAHE.

Requirements:
- GUI-independent
- NumPy/OpenCV based
- no in-place mutation
- preserve image representation
- support grayscale and color appropriately
- validate parameters
- add type hints and docstrings
- add unit tests

Do not modify unrelated architecture.

Do not implement denoising, deblurring, inpainting, Quick Restore, Advanced mode, or AI.

Run the relevant tests and report changed files and results.
```

---

# 37. Suggested Agent Prompt — White Balance and Resize

```text
Continue Phase 4 only.

Implement:
1. automatic white balance using a deterministic classical method
2. resize operation with validation

Follow the existing image representation and processing architecture.

Do not modify the original image in place.

Add tests for:
- color and grayscale handling where applicable
- invalid parameters
- shape preservation for white balance
- expected dimensions for resize
- input immutability

Do not implement any later-phase features.
```

---

# 38. Suggested Agent Prompt — UI Integration

```text
Integrate the completed Phase 4 preprocessing operations into the existing PySide6 UI.

Add a Prepare/Preprocessing section using the existing application layout and style.

Provide controls for:
- Histogram Equalization
- CLAHE
- Auto White Balance
- Normalize if implemented
- Resize

CLAHE and Resize may use compact dialogs for parameters.

Every successful operation must commit through ImageState so that undo/redo works correctly.

Refresh current-image analysis after processing.

Do not redesign unrelated parts of the application.

Do not implement Quick Restore or later restoration operations.
```

---

# 39. Human Review Checkpoint 1

After processing functions are complete, inspect:

```text
processing/preprocessing/
```

Check:

- functions are independent of PySide6
- inputs are not mutated
- color operations are handled correctly
- parameters are validated
- code is simple
- there are no unnecessary abstractions
- no hidden automatic operations exist

Do not proceed to UI integration until this is satisfactory.

---

# 40. Human Review Checkpoint 2

After UI integration manually test:

```text
Open image
→ Histogram Equalization
→ Undo
→ Redo
→ CLAHE
→ Undo
→ White Balance
→ Undo
→ Resize
→ Reset
```

Verify every operation behaves as one history step.

---

# 41. Human Review Checkpoint 3

Load different photographs and inspect:

- contrast improvement
- natural colors
- clipping
- artifacts
- channel changes
- resizing distortion

Remember:

> The goal is controlled preprocessing, not maximum visual intensity.

---

# 42. Git Strategy

Use the project's existing Git workflow.

Suggested commits:

```text
feat: add preprocessing module structure
feat: add histogram equalization
feat: add CLAHE preprocessing
feat: add automatic white balance
feat: add image normalization
feat: add resize preprocessing
test: add preprocessing unit tests
feat: integrate preprocessing controls
test: add preprocessing integration tests
```

Review:

```bash
git status
```

before the final Phase 4 commit.

Do not commit generated caches or temporary output.

---

# 43. Final Test Command

Run the project's configured test suite, typically:

```bash
pytest
```

Report:

- total tests
- passed
- failed
- skipped
- errors

Do not claim completion if tests fail without documenting why.

---

# 44. Phase 4 Acceptance Checklist

## Processing

- [ ] Histogram equalization implemented
- [ ] CLAHE implemented
- [ ] Automatic white balance implemented
- [ ] Normalization implemented only if justified
- [ ] Resize implemented
- [ ] Processing is GUI-independent
- [ ] Inputs are not mutated
- [ ] Outputs have valid dtype/range
- [ ] Color images handled correctly
- [ ] Grayscale images handled correctly

## Image State

- [ ] Successful operations create history entries
- [ ] Undo works
- [ ] Redo works
- [ ] New operation after undo clears redo
- [ ] Reset works
- [ ] Original image remains unchanged
- [ ] Processing failures preserve previous valid image

## UI

- [ ] Prepare section exists
- [ ] Histogram Equalization works
- [ ] CLAHE works
- [ ] White Balance works
- [ ] Normalize exists only if implemented
- [ ] Resize works
- [ ] Parameter dialogs validate input
- [ ] Existing UI remains functional

## Analysis

- [ ] Current-image analysis refreshes after processing
- [ ] Original-image state is not accidentally overwritten

## Testing

- [ ] Unit tests added
- [ ] Integration tests added
- [ ] Regression tests pass
- [ ] Manual visual checks completed

## Scope

- [ ] No denoising
- [ ] No deblurring
- [ ] No inpainting
- [ ] No Quick Restore
- [ ] No Advanced pipeline
- [ ] No AI
- [ ] No unnecessary dependencies

---

# 45. Expected End State

The application should now support:

```text
OPEN IMAGE
     ↓
ANALYZE
     ↓
PREPARE
 ┌───────────────┐
 │ Histogram Eq. │
 │ CLAHE         │
 │ White Balance │
 │ Normalize     │
 │ Resize        │
 └───────────────┘
     ↓
UNDO / REDO
```

The application is **not** expected to perform complete photo restoration yet.

The user should now have a stable preprocessing foundation on which later restoration algorithms can operate.

---

# 46. Phase 4 → Phase 5 Boundary

Phase 4 ends here.

Phase 5 will introduce:

> **Denoising**

Potential operations include:

- Gaussian filtering
- Median filtering
- Bilateral filtering
- Non-local means

Phase 5 must consume the clean preprocessing layer created here.

Do not implement any Phase 5 functionality during Phase 4.

---

# 47. Final Principle

The most important architectural rule for this phase is:

> **Build reliable processing primitives first; build automatic restoration decisions later.**

Phase 4 should not attempt to decide what an old photograph needs.

It should provide safe, deterministic preprocessing operations that later phases can compose into the larger restoration pipeline.

**Do not over-process.**
