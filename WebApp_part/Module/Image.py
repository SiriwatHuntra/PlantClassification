from PIL import Image
from rembg import remove
import io

import numpy as np
import cv2

def pil_to_cv2(pil_image):
    """Convert PIL.Image to OpenCV format."""
    return cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)

def crop_object(image):
    # Convert image to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

    # Apply Gaussian blur to reduce noise
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    # Apply thresholding to segment the object
    _, thresh = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # Perform morphological operations to remove small noise
    kernel = np.ones((5, 5), np.uint8)
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)

    # Find contours
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if not contours:
        print("No object found!")
        return image  # Return original if no object found

    # Filter out small contours by setting a minimum area threshold
    min_area = 1000  # Adjust based on your dataset
    large_contours = [cnt for cnt in contours if cv2.contourArea(cnt) > min_area]

    if not large_contours:
        print("No significant object found!")
        return image

    # Get the largest contour (assumed to be the object)
    largest_contour = max(large_contours, key=cv2.contourArea)

    # Get bounding box
    x, y, w, h = cv2.boundingRect(largest_contour)

    # Compute the square size
    max_dim = max(w, h)
    center_x, center_y = x + w // 2, y + h // 2

    # Define the square box around the object
    half_size = max_dim // 2
    x1, y1 = center_x - half_size, center_y - half_size
    x2, y2 = center_x + half_size, center_y + half_size

    # Ensure the crop stays within image bounds
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(image.shape[1], x2), min(image.shape[0], y2)

    # Crop the object with a square aspect ratio
    cropped = image[y1:y2, x1:x2]

    # Resize the image to 512x512 pixels
    cropped = cv2.resize(cropped, (512, 512))
  
    return cropped


def background_removal(image_file):
    """
    Remove the background from an image file.

    Parameters:
        image_file (str or file-like object): Path to the image file or a file-like object.

    Returns:
        PIL.Image: Image with the background removed (RGB format).
    """
    try:
        # Open the image using Pillow
        img = Image.open(image_file).convert('RGB')  # Ensure the image is in RGB format

        # Convert the image to a byte array for compatibility with rembg
        img_byte_array = io.BytesIO()
        img.save(img_byte_array, format='PNG')
        img_byte_array.seek(0)

        # Remove the background
        output_image_data = remove(img_byte_array.getvalue())

        # Convert the output back to a PIL image
        output_image_io = io.BytesIO(output_image_data)
        result_img = Image.open(output_image_io).convert('RGB')
        result_img_pil = result_img.resize((512, 512))
        result_img_npArray = pil_to_cv2(result_img_pil)
        return result_img_pil, result_img_npArray

    except Exception as e:
        raise RuntimeError(f"Error during background removal: {e}")


