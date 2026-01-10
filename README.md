# Confocal Image Labeller

## Process

1. Image is loaded and processed
2. Existing label or initial label is overlayed onto image
3. User can add or remove labelling as needed
4. User can save label to the disk

## Ruinning code

### Command to run application
poetry run streamlit run src/image_labeler/app.py -- --image <image name> --label <optional label file name>

### Example

poetry run streamlit run src/image_labeler/app.py -- 16012025_IL_G3_80_GSH10_chip1_explant3_confocal.lif

### Dependencies

dependencies are managed through poetry, so should be installed using "poetry install"

## Technology
- Streamlit + Streamlit-Drawable-Canvas
- Scipy for image processing

## Notes for modification
- analysis.py contains algorithms for initial labeling of images
  - Can add other initial label code
- app.py contains main streamlit application
  - Can change what algorithms are run on initialization of app
  - Insure caching is used to avoid excessive image load times
- image_manager.py contains image loading and processing code