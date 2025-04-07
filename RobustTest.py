import pandas as pd
import joblib, os
import numpy as np
import ast
import logging
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Utility to plot confusion matrix and model metrics
import seaborn as sns
import matplotlib.pyplot as plt

def plot_confusion_matrix(cm, class_names, csv):
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=class_names, yticklabels=class_names)
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title("Confusion Matrix " + csv)
    plt.show()

# Load dataset
def load_data(file_path):
    try:
        data = pd.read_csv(file_path)
        logging.info(f"Data loaded successfully from {file_path}.")
        return data
    except FileNotFoundError as e:
        logging.error(f"File not found: {e}")
        raise
    except pd.errors.EmptyDataError as e:
        logging.error(f"No data: {e}")
        raise
    except Exception as e:
        logging.error(f"Error loading data: {e}")
        raise

# Data preprocessing
def preprocess_data(data):
    feature_columns = [col for col in data.columns if col not in ['Image', 'Label']]
    X = data[feature_columns]

    # Convert lists of features to numeric
    for col in X.columns:
        if isinstance(X[col].iloc[0], str) and X[col].iloc[0].startswith('['):
            try:
                X[col] = X[col].apply(lambda x: np.mean(ast.literal_eval(x)) if x else np.nan)
            except Exception as e:
                logging.error(f"Error processing column {col}: {e}")
                raise

    y = data['Label']
    return X, y

# Evaluate model robustness
def evaluate_model(model_path, noisy_data_path):
    # Load the saved model
    # Load the saved model correctly
    model_data = joblib.load(model_path)

    # If model_data is a dictionary, extract the pipeline
    if isinstance(model_data, dict):
        model = model_data.get("pipeline", None)
    else:
        model = model_data  # Directly a model object

    # Ensure model is correctly loaded
    if model is None or not hasattr(model, "predict"):
        raise ValueError("Loaded model is invalid or does not support 'predict'")

    csv_name = os.path.basename(noisy_data_path)
    # Load and preprocess noisy data
    noisy_data = load_data(noisy_data_path)
    X_noise, y_noise = preprocess_data(noisy_data)

    # Make predictions
    y_pred = model.predict(X_noise)

    # Calculate accuracy
    accuracy = accuracy_score(y_noise, y_pred)
    logging.info(f"Accuracy on noisy data: {accuracy:.4f}")

    # Display classification report
    print("Classification Report:")
    print(classification_report(y_noise, y_pred))

    # Confusion matrix
    cm = confusion_matrix(y_noise, y_pred)
    # plot_confusion_matrix(cm, np.unique(y_noise), csv_name)

    return accuracy

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

    # Paths to model and noisy dataset
    # model_path = "ClassificationModel/Extract/Model/Best_Classifier_pipeline.pkl"
    # noisy_data_path = ["ClassificationModel/Extract/Extract_output/PlantData.csv",
    #                     "ClassificationModel/Extract/Extract_output/PlantData_brightness.csv",
    #                     "ClassificationModel/Extract/Extract_output/PlantData_brightness.csv",
    #                     "ClassificationModel/Extract/Extract_output/PlantData_brightness.csv",
    #                     "ClassificationModel/Extract/Extract_output/PlantData_brightness.csv",
    #                     "ClassificationModel/Extract/Extract_output/PlantData_brightness.csv"]

    model_path = "ClassificationModel/Extract/Model/Plant.pkl"
    noisy_data_path = ["ClassificationModel/Extract/Extract_output/Plant.csv",
                        "ClassificationModel/Extract/Extract_output/Plant_blur.csv",
                        "ClassificationModel/Extract/Extract_output/Plant_brightness.csv",
                        "ClassificationModel/Extract/Extract_output/Plant_damage.csv",
                        #"ClassificationModel/Extract/Extract_output/Plant_perspective.csv",
                        "ClassificationModel/Extract/Extract_output/Plant_speckle.csv"]
    
    Acc_logs = []
    CSV_list = []
    try:
        # Evaluate model on noisy dataset
        for i in noisy_data_path:
            csv_name = os.path.basename(i)
            accuracy = evaluate_model(model_path, i)
            Acc_logs.append(accuracy)
            CSV_list.append(csv_name)

        for i in range(len(Acc_logs)):
            print(f"\nAccuracy on {CSV_list[i]}: {Acc_logs[i]:.4f}")
    
    except Exception as e:
        logging.error(f"Error during robustness evaluation: {e}")
