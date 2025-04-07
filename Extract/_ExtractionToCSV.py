import os, re
import pandas as pd
from tqdm import tqdm
import cv2

from Feature import extract_features

def natural_sort_key(s):
    return [int(text) if text.isdigit() else text.lower() for text in re.split('(\d+)', s)]

def get_labeled_image_paths(folder_path):
    """
    Fetch labeled image paths from a folder.
    """
    if not os.path.exists(folder_path):
        raise FileNotFoundError(f"The folder path {folder_path} does not exist.")

    labeled_image_paths = []
    labels = []

    for subfolder_name in sorted(os.listdir(folder_path), key=natural_sort_key):
        subfolder_path = os.path.join(folder_path, subfolder_name)

        if not os.path.isdir(subfolder_path):
            continue

        image_paths = [
            os.path.join(subfolder_path, file_name)
            for file_name in os.listdir(subfolder_path)
            if file_name.lower().endswith((".png", ".jpg", ".jpeg", ".bmp", ".tiff"))
        ]
        
        if image_paths:
            labeled_image_paths.extend(image_paths)
            labels.extend([subfolder_name] * len(image_paths))

    return labeled_image_paths, labels

def extract_and_save_features(image_paths, labels, output_csv_path):
    """
    Extract features from images and save them to a CSV file.
    """
    if not image_paths or not labels:
        raise ValueError("Image paths and labels cannot be empty.")

    os.makedirs(os.path.dirname(output_csv_path), exist_ok=True)

    data = []

    for image_path, label in tqdm(zip(image_paths, labels), total=len(image_paths), desc="Processing Images"):
        try:
            features = extract_features(cv2.imread(image_path))
            if not features:
                print(f"No features extracted for {image_path}, skipping.")
                continue

            image_name = os.path.basename(image_path)
            row = [f"{image_name}-class-{label}", label] + list(features.values())
            data.append(row)
        except Exception as e:
            print(f"Error processing image {image_path}: {e}")

    try:
        columns = ["Image", "Label"] + list(features.keys())
        df = pd.DataFrame(data, columns=columns)
        df.to_csv(output_csv_path, index=False)
        print(f"Feature data saved to {output_csv_path}")
    except Exception as e:
        print(f"Error saving data to CSV: {e}")

# Example Usage
if __name__ == "__main__":
    try:
        # List of folders to iterate through (can add more augmentations)
        # folder_paths = [
        #     "ClassificationModel/Augmented_Out/blur",
        #     "ClassificationModel/Augmented_Out/damage",
        #     "ClassificationModel/Augmented_Out/speckle",
        #     "ClassificationModel/Augmented_Out/brightness",
        #     "ClassificationModel/Augmented_Out/perspective"
        # ]

        folder_paths = ["ClassificationModel/Image_Output_2"]

        # Iterate over each folder
        for folder_path in folder_paths:
            print(f"\nProcessing folder: {folder_path}")

            # Derive output CSV path based on augmentation folder
            augmentation_type = os.path.basename(folder_path)
            # output_csv_path = f"ClassificationModel/Extract/Extract_output/Plant_{augmentation_type}.csv"
            output_csv_path = f"ClassificationModel/Extract/Extract_output/Plant2.csv"
            # Extract paths and labels, then process them
            image_paths, labels = get_labeled_image_paths(folder_path)
            extract_and_save_features(image_paths, labels, output_csv_path)

            print(f"Features extracted and saved to {output_csv_path}")

    except Exception as e:
        print(f"Critical error: {e}")

