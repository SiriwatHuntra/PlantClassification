import os
import pandas as pd
import joblib
import logging
import numpy as np
import optuna
import shap
import seaborn as sns
import matplotlib.pyplot as plt
from datetime import datetime

from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.ensemble import HistGradientBoostingClassifier, GradientBoostingClassifier, RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from lightgbm import LGBMClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report, confusion_matrix
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from imblearn.over_sampling import SMOTE
from sklearn.svm import SVC

# ===== Utility Functions =====
# Configure logging to log into a txt file
# === Dynamic path generator ===
def generate_paths(model_name):
    """
    Generate timestamped log and model file paths based on model name.
    :param model_name: Name of the model
    :return: Tuple of (log_txt_path, model_filename)
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Create directories if they don't exist
    log_dir = "ClassificationModel/Extract/Logs"
    model_dir = "ClassificationModel/Extract/Model"
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(model_dir, exist_ok=True)
    
    # Construct file paths
    log_txt_path = os.path.join(log_dir, f"{model_name}_.txt")
    model_filename = os.path.join(model_dir, f"{model_name}_Plant.pkl")
    
    return log_txt_path, model_filename

def load_data(file_path):
    """Load dataset from CSV file."""
    data = pd.read_csv(file_path)
    logging.info(f"Data loaded successfully from {file_path}.")
    return data

def preprocess_data(data):
    """Preprocess dataset by handling missing values and separating features/labels."""
    feature_columns = [col for col in data.columns if col not in ['Image', 'Label']]
    X = data[feature_columns]
    y = data['Label']
    
    return X, y

def balance_data(X, y):
    """Apply SMOTE to balance the dataset."""
    smote = SMOTE(sampling_strategy="auto", random_state=42)
    X_resampled, y_resampled = smote.fit_resample(X, y)
    logging.info("Applied SMOTE to balance the dataset.")
    return X_resampled, y_resampled

# ===== Model Training & Optimization =====
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor, GradientBoostingRegressor
from sklearn.svm import SVR

# ===== Model Training & Optimization =====
def objective(trial, X_train, y_train, model_name):
    """Objective function for Optuna hyperparameter tuning with multiple models."""
    
    model_mapping = {
        # Classification Models
        "HistGradientBoosting": HistGradientBoostingClassifier,
        "RandomForest": RandomForestClassifier,
        "GradientBoosting": GradientBoostingClassifier,
        "SVM": SVC,
        "LightGBM": LGBMClassifier,

        # Regression Models
        "HistGradientBoostingRegressor": HistGradientBoostingRegressor,
        "RandomForestRegressor": RandomForestRegressor,
        "GradientBoostingRegressor": GradientBoostingRegressor,
        "SVR": SVR,
        "MLP": MLPClassifier

    }

    if model_name not in model_mapping:
        raise ValueError(f"Model '{model_name}' is not recognized.")

    params = {}

    # Classification Models
    if model_name == "HistGradientBoosting":
        params = {
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.2),
            'max_depth': trial.suggest_int('max_depth', 3, 10),
            'max_iter': trial.suggest_int('max_iter', 100, 300),
            'l2_regularization': trial.suggest_float('l2_regularization', 0.0, 0.1)  # Overfitting Reduction
        }
    elif model_name == "RandomForest":
        params = {
            'n_estimators': trial.suggest_int('n_estimators', 50, 300),
            'max_depth': trial.suggest_int('max_depth', 3, 20),
            'min_samples_split': trial.suggest_int('min_samples_split', 2, 10),  # Reduce Overfitting
            'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 5),
            'max_features': trial.suggest_categorical('max_features', ['sqrt', 'log2', None])  # Reduce Overfitting
        }
    elif model_name == "GradientBoosting":
        params = {
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.2),
            'n_estimators': trial.suggest_int('n_estimators', 50, 300),
            'max_depth': trial.suggest_int('max_depth', 3, 10),
            'subsample': trial.suggest_float('subsample', 0.5, 1.0),  # Overfitting Control
            'min_samples_split': trial.suggest_int('min_samples_split', 2, 10)
        }
    elif model_name == "SVM":
        params = {
            'C': trial.suggest_float('C', 0.1, 10.0),
            'gamma': trial.suggest_float('gamma', 0.0001, 1.0),
            'kernel': trial.suggest_categorical('kernel', ['linear', 'rbf', 'poly'])
        }

    # Regression Models
    elif model_name == "HistGradientBoostingRegressor":
        params = {
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.2),
            'max_depth': trial.suggest_int('max_depth', 3, 10),
            'max_iter': trial.suggest_int('max_iter', 100, 300),
            'l2_regularization': trial.suggest_float('l2_regularization', 0.0, 0.1)
        }
    elif model_name == "RandomForestRegressor":
        params = {
            'n_estimators': trial.suggest_int('n_estimators', 50, 300),
            'max_depth': trial.suggest_int('max_depth', 3, 20),
            'min_samples_split': trial.suggest_int('min_samples_split', 2, 10),
            'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 5),
            'max_features': trial.suggest_categorical('max_features', ['sqrt', 'log2', None])
        }

    elif model_name == "LightGBM":
        params = {
            'n_estimators': trial.suggest_int('n_estimators', 50, 300),
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3),
            'max_depth': trial.suggest_int('max_depth', 3, 10),
            'num_leaves': trial.suggest_int('num_leaves', 20, 300),
            'subsample': trial.suggest_float('subsample', 0.5, 1.0),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
        }


    elif model_name == "MLP":
        params = {
            'hidden_layer_sizes': trial.suggest_categorical('hidden_layer_sizes', [(100,), (50, 50), (100, 100)]),
            'activation': trial.suggest_categorical('activation', ['relu', 'tanh']),
            'solver': trial.suggest_categorical('solver', ['adam', 'sgd']),
            'alpha': trial.suggest_float('alpha', 0.0001, 0.1),
        }

    elif model_name == "SVR":
        params = {
            'C': trial.suggest_float('C', 0.1, 10.0),
            'gamma': trial.suggest_float('gamma', 0.0001, 1.0),
            'kernel': trial.suggest_categorical('kernel', ['linear', 'rbf', 'poly'])
        }

    model = model_mapping[model_name](random_state=42, **params)

    pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="mean")),
        ("scaler", StandardScaler()),
        ("classifier", model)
    ])

    cv_score = cross_val_score(pipeline, X_train, y_train, cv=5, scoring='accuracy').mean()
    logging.info(f"Optuna Trial - CV Accuracy: {cv_score:.4f} with params: {params} for {model_name}")
    
    return cv_score


def tune_hyperparameters(X_train, y_train, model_name):
    """Optimize hyperparameters using Optuna for different models."""
    
    model_mapping = {
        "HistGradientBoosting": HistGradientBoostingClassifier,
        "RandomForest": RandomForestClassifier,
        "GradientBoosting": GradientBoostingClassifier,
        "SVM": SVC,
        "LightGBM": LGBMClassifier, 
        "MLP": MLPClassifier         
    }


    if model_name not in model_mapping:
        raise ValueError(f"Model '{model_name}' is not recognized.")

    study = optuna.create_study(direction='maximize')
    study.optimize(lambda trial: objective(trial, X_train, y_train, model_name), n_trials=10)

    logging.info(f"Best params for {model_name}: {study.best_params}")

    # ใช้ pipeline และ train ที่นี่
    best_model = model_mapping[model_name](random_state=42, **study.best_params)

    best_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="mean")),
        ("scaler", StandardScaler()),
        ("classifier", best_model)
    ])
    
    best_pipeline.fit(X_train, y_train)  # 🚀 ใช้ pipeline train อย่างถูกต้อง
    
    return best_pipeline, study.best_params


def cross_validate_model(X, y, best_pipeline, n_splits=5):
    """Perform cross-validation and evaluate the model."""
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    accuracy_scores = cross_val_score(best_pipeline, X, y, cv=skf, scoring='accuracy')
    
    mean_accuracy = np.mean(accuracy_scores)
    logging.info(f"Cross-Validation Accuracy (Mean): {mean_accuracy:.4f}")

    best_pipeline.fit(X, y)
    
    try:
        explainer = shap.Explainer(best_pipeline.predict, X)
        shap_values = explainer(X)
        shap.summary_plot(shap_values, X)
    except Exception as e:
        logging.warning(f"SHAP feature importance not computed: {e}")

    return mean_accuracy

# ===== Model Evaluation =====
def evaluate_model(pipeline, X_test, y_test):
    """Evaluate trained pipeline on the test dataset."""

    y_pred = pipeline.predict(X_test)  

    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average='weighted')
    recall = recall_score(y_test, y_pred, average='weighted')
    f1 = f1_score(y_test, y_pred, average='weighted')

    class_report = classification_report(y_test, y_pred, output_dict=True)
    cm = confusion_matrix(y_test, y_pred)

    logging.info(f"Confusion Matrix:\n{cm}")
    logging.info(f"Classification Report:\n{classification_report(y_test, y_pred)}")

    return accuracy, precision, recall, f1


def plot_confusion_matrix(cm, class_names):
    """Plot confusion matrix."""
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=class_names, yticklabels=class_names)
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title("Confusion Matrix")
    plt.show()

# ===== Model Saving =====
def save_best_model(model, best_params, file_path="Plant.pkl"):
    """Save the best trained model."""
    joblib.dump({
        "pipeline": model,
        "best_params": best_params
    }, file_path)
    logging.info(f"Best pipeline saved successfully to {file_path}.")

# ===== Main Execution =====
import logging
import pandas as pd

# เก็บค่า Log ในรูปแบบ List
log_data = []

import os

# Define log CSV path
log_csv_path = "ClassificationModel/Extract/Logs/model_comparison_log.csv"

# Check if log directory exists, if not, create it
os.makedirs(os.path.dirname(log_csv_path), exist_ok=True)

# ===== Main Execution =====
if __name__ == "__main__":
    dataset_path = "ClassificationModel/Extract/Extract_output/Plant.csv"
    data = load_data(dataset_path)
    if data is None:
        exit(1)

    X, y = preprocess_data(data)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y)

    #X_train, y_train = balance_data(X_train, y_train)

    # Define models to compare
    # models = ["SVM", "HistGradientBoosting", "GradientBoosting", "RandomForest", "LightGBM", "MLP"] 
    models = ["SVM", "LightGBM", "MLP"] 



    for model_name in models:
        log_txt_path, model_filename = generate_paths(model_name)

        for handler in logging.root.handlers[:]:
            logging.root.removeHandler(handler)
        logging.basicConfig(
            filename=log_txt_path,
            level=logging.INFO,
            format="%(asctime)s - %(levelname)s - %(message)s",
            filemode="w"
        )

        logging.info(f"\n=== Training and tuning model: {model_name} ===\n")

        start_time = datetime.now()
        logging.info(f"Start Time: {start_time.strftime('%Y-%m-%d %H:%M:%S')}")

        best_pipeline, best_params = tune_hyperparameters(X_train, y_train, model_name)

        cv_acc = cross_validate_model(X_train, y_train, best_pipeline)

        test_accuracy, precision, recall, f1 = evaluate_model(best_pipeline, X_test, y_test)

        end_time = datetime.now()
        elapsed_time = (end_time - start_time).total_seconds()

        save_best_model(best_pipeline, best_params, model_filename)

        logging.info(f"\n===== Model {model_name} Performance =====")
        logging.info(f"Start Time: {start_time.strftime('%Y-%m-%d %H:%M:%S')}")
        logging.info(f"End Time: {end_time.strftime('%Y-%m-%d %H:%M:%S')}")
        logging.info(f"Total Training Time: {elapsed_time:.2f} seconds")
        logging.info(f"Cross Validation Accuracy: {cv_acc:.4f}")
        logging.info(f"Test Accuracy: {test_accuracy:.4f}")
        logging.info(f"Precision: {precision:.4f}")
        logging.info(f"Recall: {recall:.4f}")
        logging.info(f"F1-score: {f1:.4f}")

        print(f"\n=== {model_name} training completed in {elapsed_time:.2f} seconds! ===")
        print(f"Start: {start_time.strftime('%Y-%m-%d %H:%M:%S')}, End: {end_time.strftime('%Y-%m-%d %H:%M:%S')}")
        print("=== Model Training Log updated and saved to TXT ===\n")




