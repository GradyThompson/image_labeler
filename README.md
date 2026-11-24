# Confocal Image Labeler

An interactive Python application for semi-automated
segmentation and annotation of confocal microscopy images.

The application combines automated image processing
with manual annotation to support the creation and
refinement of segmentation masks.

## Features

- Loading Leica Image Format (LIF) microscopy images.
- Maximum-intensity projection across Z-slices.
- Automated initial segmentation using intensity
  thresholding and connected-component analysis.
- Distance-transform-based handling of small gaps.
- Interactive mask editing with brush and lasso tools.
- Loading and saving PNG segmentation masks.
- Cached image processing for responsive interaction.

## Installation

Requires Python 3.11 or later and Poetry.

```bash
git clone https://github.com/GradyThompson/image_labeler.git
cd image_labeler
poetry install
```

Create directories for input images and output labels:

```bash
mkdir -p images labels
```

## Usage

Place a compatible LIF image in the `images/` directory.

Launch the application:

```bash
poetry run streamlit run src/image_labeler/app.py -- example.lif
```

To continue editing an existing segmentation mask:

```bash
poetry run streamlit run src/image_labeler/app.py -- example.lif --label example_label.png
```

Existing label files should be placed in `labels/`.

### Annotation Workflow

1. Load and process the microscopy image.
2. Generate an initial segmentation or load an existing mask.
3. Refine the mask using brush or lasso selection.
4. Add or remove labeled regions.
5. Save the resulting segmentation mask.

Labels are saved to `labels/<image_name>_label.png`.

Saving overwrites any existing label with the same filename.

## Segmentation Method

The initial segmentation applies intensity thresholding
to identify foreground pixels. A Euclidean distance
transform allows nearby regions to be connected across
small gaps.

Connected-component analysis identifies candidate
regions, after which the largest foreground component
is retained and enclosed holes are filled.

Users can subsequently refine this automated
segmentation through the interactive interface.

## Repository Structure

| File | Purpose |
|------|---------|
| `src/image_labeler/app.py` | Streamlit interface and annotation tools |
| `src/image_labeler/analysis.py` | Initial segmentation algorithms |
| `src/image_labeler/image_manager.py` | Image loading, projection and mask handling |
| `tests/` | Test directory |
| `pyproject.toml` | Project and dependency configuration |

## Technologies

- Python
- NumPy
- SciPy
- OpenCV
- Pillow
- Streamlit
- Streamlit Drawable Canvas
- readlif
- Poetry

## Acknowledgments

The LIF maximum-intensity projection implementation
draws on the following project:

[RADICAL-UBC/neurites](https://github.com/RADICAL-UBC/neurites/blob/main/lif_max_projection.py)
