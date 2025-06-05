from rembg import remove
from PIL import Image
import io
import numpy as np
from Util.Image_IO import get_labeled_image_paths, ReadFile, SaveFile
import os
from tqdm import tqdm
import traceback
import cv2

import cv2
import numpy as np

def crop_object(image):
    """
    Localize/Crop 
    """
    # Convert image to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

    # Apply Gaussian blur to reduce noise
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    # Apply adaptive thresholding for better segmentation
    _, thresh = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # Perform morphological operations to remove small noise
    kernel = np.ones((5, 5), np.uint8)
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)

    # Find contours
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if not contours:
        print("No object found!")
        return cv2.resize(image, (512, 512))  # Return resized original image

    # Filter out small contours by setting a minimum area threshold
    min_area = 1000  # Adjust based on your dataset
    large_contours = [cnt for cnt in contours if cv2.contourArea(cnt) > min_area]

    if not large_contours:
        print("No significant object found!")
        return cv2.resize(image, (512, 512))

    # Get the largest contour (assumed to be the object)
    largest_contour = max(large_contours, key=cv2.contourArea)

    # Compute Convex Hull for a more precise bounding area
    hull = cv2.convexHull(largest_contour)

    # Get bounding box from Convex Hull
    x, y, w, h = cv2.boundingRect(hull)

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

def BG_RM(image):
    """
    Removes the background of an input image using the rembg library.

    Args:
        image (PIL.Image.Image or numpy.ndarray): Input image to process.

    Returns:
        numpy.ndarray: Image with the background removed.
    """
    if isinstance(image, np.ndarray):
        # Convert numpy array to PIL Image if needed
        image = Image.fromarray(image)
    elif not isinstance(image, Image.Image):
        raise ValueError("Input must be a PIL.Image.Image or numpy.ndarray.")

    # Convert image to bytes for rembg
    image_bytes = io.BytesIO()
    image.save(image_bytes, format='PNG')  # PNG supports transparency
    image_bytes = image_bytes.getvalue()

    # Remove background
    result_bytes = remove(image_bytes)

    # Convert bytes back to PIL Image
    result_image = Image.open(io.BytesIO(result_bytes)).convert('RGBA')

    # Convert back to numpy array
    return np.array(result_image)

def Segment(folder_path, output_directory):
    """
    Processes images for segmentation and saves outputs into class-specific subfolders.
    
    Args:
        folder_path (str): Path to the main folder containing subfolders of images.
        output_directory (str): The root output directory where segmented images will be saved.
    
    Returns:
        None
    """
    try:
        # Get labeled image paths
        labeled_image_paths = get_labeled_image_paths(folder_path)

        for class_images in tqdm(labeled_image_paths, desc="Processing Classes", unit="class"):
            if not class_images:
                continue  # Skip empty class lists

            try:
                # Extract class label from the first image path's folder name
                first_image_path = class_images[0]
                class_label = os.path.basename(os.path.dirname(first_image_path))

                # Create class-specific subfolder in the output directory
                class_output_dir = os.path.join(output_directory, class_label)
                os.makedirs(class_output_dir, exist_ok=True)

                # Process each image and save to the class subfolder
                for idx, image_path in enumerate(tqdm(class_images, desc=f"Processing {class_label}", unit="image", leave=False), start=1):
                    try:
                        # Read and preprocess the image
                        image = ReadFile(image_path, 'RGB')
                        if image is None:
                            print(f"Warning: Could not read image {image_path}. Skipping.")
                            continue

                        # Perform segmentation
                        image = crop_object(image)
                        segmented_image = BG_RM(image)

                        # Generate filename and save the processed image
                        filename = f"{idx}.jpg"
                        SaveFile(segmented_image, class_output_dir, filename)
                    
                    except Exception as e:
                        print(f"Error processing image {image_path}: {e}")
                        traceback.print_exc()
                        continue

            except Exception as e:
                print(f"Error processing class images in folder {folder_path}: {e}")
                traceback.print_exc()

        print("Segmentation and saving completed!")

    except Exception as e:
        print(f"Critical error during segmentation: {e}")
        traceback.print_exc()

