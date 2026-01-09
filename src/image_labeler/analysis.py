import numpy as np
import image_manager
from PIL import Image

"""
Creates a blank label of the specified shape and saves the the specified path
"""
def blank_label(image_path, label_path):
    """
        Create and save a blank (all-zero) label PNG matching the image size.

        Parameters
        ----------
        image_path : str
            Path to the source image.
        label_path : str
            Path where the label PNG will be saved.
        """
    # Get image projection (H, W, ...) → use first channel
    arr = image_manager.create_max_projection(image_path, image_index=1)[0]

    height = arr.shape[0]
    width = arr.shape[1]

    # Create zero label
    label = np.zeros((height, width), dtype=np.uint8)

    # Save using the previously defined safe writer
    image_manager.save_label_png(label, label_path)

"""
Perform initial guess on image

Args:
    image: 2D np array
"""
def initial_guess(image:np.typing.NDArray[np.float64]):
    return -1