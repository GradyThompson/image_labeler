import numpy as np
import image_manager
from scipy import ndimage as ndi

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

def initial_guess(image:np.typing.NDArray[np.float64],
                  threshold=None,
                  max_gap=2):
    """
       Perform initial guess on image by extracting the connected
       foreground region corresponding to the cell.

       Args:
           image: 2D np array

       Returns:
           mask: 2D boolean array of the initial cell guess
       """

    if threshold is None:
        threshold = np.mean(image)
        threshold = 200

        # Binary foreground
    binary = image > threshold

    # Allow growth across small gaps
    tolerant = ndi.distance_transform_edt(~binary) <= max_gap

    # Label connected regions
    labels, num = ndi.label(tolerant)

    if num == 0:
        raise ValueError("No regions found")

    # Select largest connected region
    sizes = ndi.sum(binary, labels, index=range(1, num + 1))
    largest = np.argmax(sizes) + 1

    mask = labels == largest

    # Optional cleanup
    mask = ndi.binary_fill_holes(mask)

    return mask