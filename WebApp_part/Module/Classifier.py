import numpy as np
import joblib
import json
import os
from .ImageExtract import extract_features

# Load the model
MODEL_PATH = "Module/Plant.pkl"
PLANT_DATA_PATH = "Module/Plant_Data.json"

model = None
expected_features = None

def load_model():
    global model, expected_features
    try:
        if not os.path.exists(MODEL_PATH):
            raise FileNotFoundError(f"Model file not found at {MODEL_PATH}")

        # 🔹 Load model dictionary
        model_data = joblib.load(MODEL_PATH)
        
        if not isinstance(model_data, dict) or "pipeline" not in model_data:
            raise ValueError("Invalid model format. Expected a dictionary with 'pipeline'.")

        model = model_data["pipeline"]  # 🔹 Extract pipeline
        best_params = model_data.get("best_params", {})  # Optional: Load best parameters

        print("Model loaded successfully.")

        # Retrieve expected number of features
        if hasattr(model, "feature_names_in_"):
            expected_features = len(model.feature_names_in_)
        elif hasattr(model, "n_features_in_"):
            expected_features = model.n_features_in_
        else:
            raise AttributeError("Model does not contain feature shape attributes.")

        print(f"Expected number of features: {expected_features}")
    except Exception as e:
        print(f"Error loading model: {e}")
        model = None
        expected_features = None

# Load plant data from JSON
plant_data = {}
def load_plant_data():
    global plant_data
    try:
        if not os.path.exists(PLANT_DATA_PATH):
            raise FileNotFoundError(f"Plant data file not found at {PLANT_DATA_PATH}")

        with open(PLANT_DATA_PATH, "r", encoding="utf-8") as json_file:
            plant_data = json.load(json_file)
        print("Plant data loaded successfully.")
    except json.JSONDecodeError:
        print("Error: Invalid JSON format in Plant_Data.json.")
        plant_data = {}
    except Exception as e:
        print(f"Error loading plant data: {e}")
        plant_data = {}

# Fetch plant details
def get_plant_data(label):
    """
    Retrieve plant information based on the predicted label.
    """
    return plant_data.get(
        str(label),
        {"common_name": "Unknown", "scientific_name": "N/A", "details": "No details available."}
    )

# Ensure extracted features match the model's expected input
def align_features(features, expected_size):
    """
    Aligns extracted features to the model's expected size.

    Args:
        features (list or np.ndarray): Extracted features from an image.
        expected_size (int): Expected number of features from the model.

    Returns:
        np.ndarray: Adjusted feature vector.
    """
    try:
        features = np.array(features, dtype=float)  # Convert to NumPy array
        current_size = features.shape[0]

        if current_size > expected_size:
            # Trim excess features
            features = features[:expected_size]
        elif current_size < expected_size:
            # Pad missing features with zeros
            features = np.pad(features, (0, expected_size - current_size), mode='constant')

        return features
    except Exception as e:
        print(f"Error aligning features: {e}")
        raise

# Classify an image
def image_classify(image):
    """
    Classifies a given image and returns plant information.
    """
    try:
        if model is None:
            return {"error": "Model not loaded. Cannot classify image."}

        # Extract features from the image
        features = extract_features(image)

        print(features)
        if not features:
            return {"error": "Feature extraction failed."}

        print(f"Extracted {len(features)} features.")

        # Align features
        # features_array = align_features(features, expected_features)
        # features_array = features_array.reshape(1, -1)

        feature_values = []
        for value in features.values():
            if isinstance(value, list):  # Handle array-based features
                feature_values.append(np.mean(value))  # Use mean or flatten as needed
            else:
                feature_values.append(value)

        feature_array = np.array(feature_values).reshape(1, -1)

        # Predict label
        prediction = model.predict(feature_array)
        if prediction is None or len(prediction) == 0:
            return {"error": "Prediction failed."}

        label = prediction[0]
        print(f"Predicted label: {label}")

        # Retrieve plant details
        plant_info = get_plant_data(label)
        return plant_info

    except Exception as e:
        print(f"Error in image classification: {e}")
        return {"error": str(e)}

# Load everything on import
load_model()
load_plant_data()
