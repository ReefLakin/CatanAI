# This class is used to preprocess the state information passed to the select_action method.
# It takes the complex lists of state information and converts them into a list of integers.

# Import the numpy library
import numpy as np


# Class definition
class PixelPreprocessor:
    # Method for normalising the pixel arrays
    def normalise_pixel_array(self, pixel_array):
        # Convert to numpy array
        pixel_array = np.array(pixel_array)

        # Convert pixel values to floats
        pixel_array = pixel_array.astype(np.float32)

        # Scale pixel values to the range [0, 1]
        pixel_array = pixel_array / 255.0

        return pixel_array
