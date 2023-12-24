# average.py
from PIL import Image
import numpy as np

def averaging(original_image_path, processed_image_widget=None):
    # Read the image
    original_image = np.array(Image.open(original_image_path))

    # Check if the image is grayscale or color
    if len(original_image.shape) == 2:  # Grayscale image
        m, n = original_image.shape
        channels = 1
        original_image = original_image[:, :, np.newaxis]  # Add a third dimension for a single channel
    elif len(original_image.shape) == 3:  # Color image
        m, n, channels = original_image.shape
    else:
        raise ValueError("Unsupported image format")

    # Develop an Averaging filter (3, 3) mask
    mask = np.ones([3, 3], dtype=int)
    mask = mask / 9

    # Convolve the 3x3 mask over the image
    averaged_image = np.zeros([m, n, channels], dtype=np.uint8)

    for c in range(channels):
        for i in range(1, m - 1):
            for j in range(1, n - 1):
                temp = 0
                for k in range(3):
                    for l in range(3):
                        temp += original_image[i - 1 + k, j - 1 + l, c] * mask[k, l]

                averaged_image[i, j, c] = temp.astype(np.uint8)

    if channels == 1:
        # If the input image was grayscale, remove the third dimension
        averaged_image = np.squeeze(averaged_image, axis=-1)

    return averaged_image
