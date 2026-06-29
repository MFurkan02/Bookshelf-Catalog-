"""
🗂️ PROJECT SANDBOX / EXPERIMENTAL UTILITY
Course Project: Bookshelf Catalog System

PURPOSE:
Isolated unit testing environment for the LAB-color space image brightness normalization algorithm. 
Used to benchmark luminance threshold clipping adjustments (`cv2.cvtColor`, `np.clip`) on test images.

STATUS: 
Archived in Sandbox. Core function (`adjust_brightness_if_needed`) has been cleanly 
embedded inside the main production entry point (`book.py`).
"""


import cv2
import numpy as np
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def adjust_brightness_if_needed(img, target_brightness=120, dark_threshold=90, bright_threshold=140):
    if img is None:
        return None
    try:
        lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        avg_brightness = np.mean(l)
        logger.info(f"Original brightness: {avg_brightness}")

        if avg_brightness < dark_threshold or avg_brightness > bright_threshold:
            # Compute brightness shift value
            diff = target_brightness - avg_brightness
            l = np.clip(l + diff, 0, 255).astype(np.uint8)
            adjusted_lab = cv2.merge((l, a, b))
            adjusted_img = cv2.cvtColor(adjusted_lab, cv2.COLOR_LAB2BGR)
            logger.info(f"Adjusted brightness to target: {target_brightness}")
            return adjusted_img
        else:
            return img
    except Exception as e:
        logger.error(f"Brightness adjustment error: {str(e)}")
        return img

# === TEST CODE ===

# Load your image here (replace 'input.jpg' with your actual file path)
input_path = 'bookshelf/ev-bright.jpg'
output_path = 'adjusted_output.jpg'

image = cv2.imread(input_path)

if image is None:
    logger.error("Image could not be loaded. Check the file path.")
else:
    adjusted = adjust_brightness_if_needed(image)
    cv2.imwrite(output_path, adjusted)
    logger.info(f"Saved adjusted image to {output_path}")

    # Display the original and adjusted images side-by-side
    cv2.imshow("Brightness Adjusted Image", adjusted)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
