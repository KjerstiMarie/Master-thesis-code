import time
from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, IsolationForest
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import RobustScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    fbeta_score,
    confusion_matrix,
)


# SETTINGS

BASE_DIR = Path(__file__).resolve().parent
DATA_FOLDER = BASE_DIR / "CSE-CIC-IDS2018"
USE_FRACTION = 0.3
RANDOM_STATE = 42



# PRINT FUNCTIONS

def print_section(title):
    """
    Prints a clear section title in the terminal.
    """
    print("\n" + "=" * 60)
    print(title)
    print("=" * 60)


def print_summary_line(name, metrics):
    """
    Prints a short summary line for one model.

    The line shows the most important evaluation results, such as accuracy,
    precision, recall, F1-score, false positives and false negatives.
    """
    print(
        f"{name:<22} "
        f"Accuracy: {metrics['accuracy']:.4f}, "
        f"Precision: {metrics['precision']:.4f}, "
        f"Recall: {metrics['recall']:.4f}, "
        f"F1: {metrics['f1']:.4f}, "
        f"F2: {metrics['f2']:.4f}, "
        f"FP: {metrics['fp']}, "
        f"FN: {metrics['fn']}, "
        f"FPR: {metrics['fpr']:.6f}, "
        f"FNR: {metrics['fnr']:.6f}"
    )


#Load data

def load_dataset(folder_path, use_fraction=None):
    """
    Loads all CSV files from the dataset folder and combines them into one dataset.

    If use_fraction is given, only a part of each file is used.
    """
    all_data = []

    folder_path = Path(folder_path)

    if not folder_path.exists():
        raise FileNotFoundError(f"The folder does not exist: {folder_path}")

    csv_files = list(folder_path.glob("*.csv"))

    if len(csv_files) == 0:
        raise ValueError(f"No CSV files found in folder: {folder_path}")

    print(f"Looking for CSV files in: {folder_path}")
    print(f"Found {len(csv_files)} CSV files.")

    for file_path in csv_files:
        print("Loading:", file_path.name)

        df = pd.read_csv(file_path, low_memory=False)

        if use_fraction is not None:
            df = df.sample(frac=use_fraction, random_state=RANDOM_STATE)

        all_data.append(df)

    data = pd.concat(all_data, ignore_index=True)

    print("Combined shape:", data.shape)

    return data



# Finds the label column


def find_label_column(df):
    """
    Finds the column that contains the labels in the dataset.
    """

    for col in df.columns:
        if col.strip().lower() == "label":
            return col

    raise ValueError(
        "Could not find a label column named 'Label'. "
        f"Other columns are: {df.columns.tolist()}"
    )


#Preprocessing and cleaning of the data

def prepare_data(df):
    """
    Prepares the dataset for machine learning.

    The labels are first changed into two classes:
    benign traffic is given the value 0, and attack traffic is given the value 1.

    Columns that should not be used directly by the model are removed, such as
    IP addresses, ports and timestamps. Finally, only numeric columns are kept,
    because the models need numbers as input.
    """
    df = df.copy()

    df.columns = [col.strip() for col in df.columns]

    label_col = find_label_column(df)

    labels = df[label_col].astype(str).str.strip().str.lower()

    # benign = 0, attack = 1
    y = (labels != "benign").astype(int)

    X = df.drop(columns=[label_col])

    X = X.replace([np.inf, -np.inf], np.nan)
    
    #Drop these if they exists
    columns_to_drop = [
        "id",
        "ID",
        "Flow ID",
        "Src IP",
        "Dst IP",
        "Source IP",
        "Destination IP",
        "Timestamp",
        "Src Port",
        "Dst Port",
        "Source Port",
        "Destination Port",
        "Attempted Category",
        "Attack",
        "attack",
        "Class",
        "ICMP Code",
        "ICMP Type",
    ]

    existing_columns_to_drop = [
        col for col in columns_to_drop if col in X.columns
    ]

    X = X.drop(columns=existing_columns_to_drop)

    # If it is text columns convert to numeric 
    for col in X.columns:
        if X[col].dtype == "object":
            X[col] = pd.to_numeric(X[col], errors="coerce")

    # Keep only numeric columns
    X = X.select_dtypes(include=[np.number])

    print("Number of features used:", X.shape[1])

    return X, y



# Evaluation of the models

def evaluate_model(y_true, y_pred):
    """
    Calculates how well the model performs.

    The function compares the true labels with the model predictions.
    It calculates common metrics such as accuracy, precision, recall,
    F1-score.

    It also calculates the false positive rate and false negative rate.
    """
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])

    tn, fp, fn, tp = cm.ravel()

    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0
    fnr = fn / (fn + tp) if (fn + tp) > 0 else 0.0

    metrics = {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
        "f2": fbeta_score(y_true, y_pred, beta=2, zero_division=0),
        "confusion_matrix": cm,
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "tp": tp,
        "fpr": fpr,
        "fnr": fnr,
    }

    return metrics


def print_metrics(title, metrics):
    """
    Prints the evaluation results for a model in a readable way.

    This includes both the main evaluation metrics and the confusion matrix,
    so it is easier to understand what kind of mistakes the model makes.
    """
    print("\n" + "=" * 60)
    print(title)
    print("=" * 60)

    print(f"Accuracy : {metrics['accuracy']:.4f}")
    print(f"Precision: {metrics['precision']:.4f}")
    print(f"Recall   : {metrics['recall']:.4f}")
    print(f"F1-score : {metrics['f1']:.4f}")
    print(f"F2-score : {metrics['f2']:.4f}")

    print("\nConfusion Matrix:")
    print("Rows: actual [benign, attack]")
    print("Cols: predicted [benign, attack]")
    print(metrics["confusion_matrix"])

    print("\nValues:")
    print("TN:", metrics["tn"])
    print("FP:", metrics["fp"])
    print("FN:", metrics["fn"])
    print("TP:", metrics["tp"])

    print("\nError rates:")
    print(f"False Positive Rate: {metrics['fpr']:.6f}")
    print(f"False Negative Rate: {metrics['fnr']:.6f}")



#Random Forest model with gridsearch

def tune_random_forest(X_train, y_train, X_val, y_val, model_name):
    """
    Trains and tests several versions of a Random Forest model.

    The function tries different parameter values and different decision
    thresholds. The goal is to find the combination that gives the best result
    on the validation set.

    The threshold decides how certain the model must be before it classifies
    traffic as an attack.
    """
    print_section(f"{model_name} GRID SEARCH")

    # Small grid search
    param_grid = [
        {
            "n_estimators": 100,
            "max_depth": 10,
            "min_samples_leaf": 5,
            "class_weight": "balanced",
        },
        {
            "n_estimators": 100,
            "max_depth": 20,
            "min_samples_leaf": 5,
            "class_weight": "balanced",
        },
        {
            "n_estimators": 200,
            "max_depth": 20,
            "min_samples_leaf": 5,
            "class_weight": "balanced",
        },
        {
            "n_estimators": 100,
            "max_depth": 20,
            "min_samples_leaf": 10,
            "class_weight": "balanced",
        },
    ]

    thresholds = [0.10, 0.20, 0.30, 0.40, 0.50]

    best_model = None
    best_params = None
    best_threshold = None
    best_metrics = None
    best_f1 = -1

    print(f"Number of parameter combinations: {len(param_grid)}")
    print(f"Thresholds tested: {thresholds}")

    for i, params in enumerate(param_grid, start=1):
        print(f"\nTesting combination {i}/{len(param_grid)}")
        print(params)

        model = RandomForestClassifier(
            n_estimators=params["n_estimators"],
            max_depth=params["max_depth"],
            min_samples_split=10,
            min_samples_leaf=params["min_samples_leaf"],
            max_features="sqrt",
            class_weight=params["class_weight"],
            random_state=RANDOM_STATE,
            n_jobs=2,
        )

        model.fit(X_train, y_train)

        probabilities = model.predict_proba(X_val)[:, 1]

        for threshold in thresholds:
            y_pred = (probabilities >= threshold).astype(int)
            metrics = evaluate_model(y_val, y_pred)

            print(
                f"threshold={threshold:.2f} | "
                f"precision={metrics['precision']:.4f} | "
                f"recall={metrics['recall']:.4f} | "
                f"f1={metrics['f1']:.4f} | "
                f"f2={metrics['f2']:.4f} | "
                f"fpr={metrics['fpr']:.6f} | "
                f"fnr={metrics['fnr']:.6f}"
            )

            if metrics["f1"] > best_f1:
                best_f1 = metrics["f1"]
                best_model = model
                best_params = params
                best_threshold = threshold
                best_metrics = metrics

    print("\nBest parameters:")
    print(best_params)
    print("Best threshold:", best_threshold)
    print(f"Best validation F1: {best_f1:.4f}")

    return best_model, best_params, best_threshold, best_metrics


#Isolation Forest model with grid search

def tune_isolation_forest(X_train, y_train, X_val, y_val):
    """
    Tunes and evaluates the Isolation Forest model.

    The function selects benign training samples, tests different parameter
    combinations, and calculates anomaly scores on the validation data.

    Several percentile-based thresholds are tested. For each threshold, the
    validation predictions are evaluated using the selected metrics.

    The model and threshold with the best validation F1-score are returned.
    """
    print_section("ISOLATION FOREST GRID SEARCH")

    X_train_benign = X_train[y_train.values == 0]

    if len(X_train_benign) == 0:
        X_train_benign = X_train

    
    param_grid = [
        {
            "n_estimators": 100,
            "max_samples": 512,
            "contamination": "auto",
        },
        {
            "n_estimators": 100,
            "max_samples": 1024,
            "contamination": "auto",
        },
        {
            "n_estimators": 200,
            "max_samples": 1024,
            "contamination": "auto",
        },
    ]

    percentiles = [5, 10, 15, 20, 25, 30]

    best_model = None
    best_params = None
    best_threshold = None
    best_percentile = None
    best_metrics = None
    best_f1 = -1

    print(f"Number of parameter combinations: {len(param_grid)}")
    print(f"Percentiles tested: {percentiles}")

    for i, params in enumerate(param_grid, start=1):
        print(f"\nTesting combination {i}/{len(param_grid)}")
        print(params)

        actual_max_samples = min(params["max_samples"], len(X_train_benign))

        model = IsolationForest(
            n_estimators=params["n_estimators"],
            contamination=params["contamination"],
            max_samples=actual_max_samples,
            random_state=RANDOM_STATE,
            n_jobs=2,
        )

        model.fit(X_train_benign)

        scores = model.decision_function(X_val)

        for percentile in percentiles:
            threshold = np.percentile(scores, percentile)
            y_pred = (scores <= threshold).astype(int)

            metrics = evaluate_model(y_val, y_pred)

            print(
                f"percentile={percentile} | "
                f"precision={metrics['precision']:.4f} | "
                f"recall={metrics['recall']:.4f} | "
                f"f1={metrics['f1']:.4f} | "
                f"f2={metrics['f2']:.4f} | "
                f"fpr={metrics['fpr']:.6f} | "
                f"fnr={metrics['fnr']:.6f}"
            )

            if metrics["f1"] > best_f1:
                best_f1 = metrics["f1"]
                best_model = model
                best_params = {
                    "n_estimators": params["n_estimators"],
                    "max_samples": actual_max_samples,
                    "contamination": params["contamination"],
                    "trained_on": "benign training samples only",
                }
                best_threshold = threshold
                best_percentile = percentile
                best_metrics = metrics

    print("\nBest Isolation Forest parameters:")
    print(best_params)
    print("Best percentile:", best_percentile)
    print("Best threshold:", best_threshold)
    print(f"Best validation F1: {best_f1:.4f}")

    return best_model, best_params, best_threshold, best_percentile, best_metrics


#Prediciton helper functions

def predict_random_forest(model, X, threshold):
    """
    Makes predictions with a trained Random Forest model.

    The model first gives a probability that each sample is an attack.
    If the probability is higher than the chosen threshold, the sample is
    classified as an attack.
    """    
    probabilities = model.predict_proba(X)[:, 1]
    predictions = (probabilities >= threshold).astype(int)
    return predictions


def predict_isolation_forest(model, X, threshold):
    """
    Makes predictions with a trained Isolation Forest model.

    Samples with anomaly scores below the chosen threshold are classified
    as attacks. The rest are classified as benign traffic.
    """
    scores = model.decision_function(X)
    predictions = (scores <= threshold).astype(int)
    return predictions


def get_isolation_features(model, X, threshold):
    """
    Creates extra features based on the Isolation Forest model.

    For each sample, the function adds two new values:
    one that says whether the sample was marked as anomalous, and one anomaly
    score. These new features are later used by the hybrid Random Forest model.
    """
    scores = model.decision_function(X)

    anomaly_flag = (scores <= threshold).astype(int)
    anomaly_score = -scores

    anomaly_flag = anomaly_flag.reshape(-1, 1)
    anomaly_score = anomaly_score.reshape(-1, 1)

    return np.hstack([X, anomaly_flag, anomaly_score])



# MAIN


def main():
    """
    Runs the full experiment from start to finish.

    The function loads the dataset, prepares the data, splits it into training,
    validation and test sets, trains the models, evaluates them and prints a
    final summary of the results.

    The three tested approaches are:
    1. Random Forest
    2. Isolation Forest
    3. Hybrid Random Forest with Isolation Forest features
    """
    
   #1. Load data
    print_section("LOAD DATA")

    raw_df = load_dataset(
        folder_path=DATA_FOLDER,
        use_fraction=USE_FRACTION,
    )

    label_col = find_label_column(raw_df)

    print("\nOriginal label distribution:")
    print(raw_df[label_col].value_counts())

   #2.Prepare the data

    print_section("PREPARE DATA")

    X, y = prepare_data(raw_df)

    print("\nBinary label distribution:")
    print(y.value_counts())

    #3. Split the data 70,15,15

    print_section("SPLIT DATA 70 / 15 / 15")

    X_train, X_temp, y_train, y_temp = train_test_split(
        X,
        y,
        test_size=0.30,
        random_state=RANDOM_STATE,
        stratify=y,
    )

    X_val, X_test, y_val, y_test = train_test_split(
        X_temp,
        y_temp,
        test_size=0.50,
        random_state=RANDOM_STATE,
        stratify=y_temp,
    )

    print("Train rows:", len(X_train))
    print("Validation rows:", len(X_val))
    print("Test rows:", len(X_test))

    print("\nTrain label distribution:")
    print(y_train.value_counts())

    print("\nValidation label distribution:")
    print(y_val.value_counts())

    print("\nTest label distribution:")
    print(y_test.value_counts())

    # 4.Preprocess the data
    print_section("PREPROCESSING")

    preprocessor = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", RobustScaler()),
        ]
    )

    X_train_t = preprocessor.fit_transform(X_train)
    X_val_t = preprocessor.transform(X_val)
    X_test_t = preprocessor.transform(X_test)

    print("X_train:", X_train_t.shape)
    print("X_val  :", X_val_t.shape)
    print("X_test :", X_test_t.shape)

    #5. Random forest with gridsearch and timer

    rf_start = time.perf_counter()

    rf_model, rf_params, rf_threshold, rf_val_metrics = tune_random_forest(
        X_train_t,
        y_train,
        X_val_t,
        y_val,
        model_name="RANDOM FOREST",
    )

    rf_train_pred = predict_random_forest(
        rf_model,
        X_train_t,
        rf_threshold,
    )

    rf_test_pred = predict_random_forest(
        rf_model,
        X_test_t,
        rf_threshold,
    )

    rf_train_metrics = evaluate_model(y_train, rf_train_pred)
    rf_test_metrics = evaluate_model(y_test, rf_test_pred)

    rf_runtime = time.perf_counter() - rf_start

    print_metrics("Random Forest - train", rf_train_metrics)
    print_metrics("Random Forest - validation", rf_val_metrics)
    print_metrics("Random Forest - test", rf_test_metrics)

   #6. Isoaltion forest with gridsearch and timer
    if_start = time.perf_counter()

    if_model, if_params, if_threshold, if_percentile, if_val_metrics = tune_isolation_forest(
        X_train_t,
        y_train,
        X_val_t,
        y_val,
    )

    if_train_pred = predict_isolation_forest(
        if_model,
        X_train_t,
        if_threshold,
    )

    if_test_pred = predict_isolation_forest(
        if_model,
        X_test_t,
        if_threshold,
    )

    if_train_metrics = evaluate_model(y_train, if_train_pred)
    if_test_metrics = evaluate_model(y_test, if_test_pred)

    if_runtime = time.perf_counter() - if_start

    print_metrics("Isolation Forest - train", if_train_metrics)
    print_metrics("Isolation Forest - validation", if_val_metrics)
    print_metrics("Isolation Forest - test", if_test_metrics)

  #7. Hybrid forest using the same grid as the random forest
    hybrid_start = time.perf_counter()

    print_section("CREATE HYBRID FEATURES")

    X_train_h = get_isolation_features(
        if_model,
        X_train_t,
        if_threshold,
    )

    X_val_h = get_isolation_features(
        if_model,
        X_val_t,
        if_threshold,
    )

    X_test_h = get_isolation_features(
        if_model,
        X_test_t,
        if_threshold,
    )

    print("X_train_h:", X_train_h.shape)
    print("X_val_h  :", X_val_h.shape)
    print("X_test_h :", X_test_h.shape)

    hybrid_model, hybrid_params, hybrid_threshold, hybrid_val_metrics = tune_random_forest(
        X_train_h,
        y_train,
        X_val_h,
        y_val,
        model_name="HYBRID RANDOM FOREST",
    )

    hybrid_train_pred = predict_random_forest(
        hybrid_model,
        X_train_h,
        hybrid_threshold,
    )

    hybrid_test_pred = predict_random_forest(
        hybrid_model,
        X_test_h,
        hybrid_threshold,
    )

    hybrid_train_metrics = evaluate_model(y_train, hybrid_train_pred)
    hybrid_test_metrics = evaluate_model(y_test, hybrid_test_pred)

    hybrid_runtime = time.perf_counter() - hybrid_start

    print_metrics("Hybrid - train", hybrid_train_metrics)
    print_metrics("Hybrid - validation", hybrid_val_metrics)
    print_metrics("Hybrid - test", hybrid_test_metrics)

    #8. Summary of the results
    print_section("SUMMARY RESULTS")

    print("\nBest Random Forest parameters:")
    print(rf_params)
    print("Selected RF threshold:", rf_threshold)

    print("\nBest Isolation Forest parameters:")
    print(if_params)
    print("Selected IF percentile:", if_percentile)
    print("Selected IF threshold:", if_threshold)

    print("\nBest Hybrid parameters:")
    print(hybrid_params)
    print("Selected Hybrid threshold:", hybrid_threshold)
    print("Hybrid extra features: Isolation Forest anomaly_flag and anomaly_score")

    print("\nPerformance summary:")
    print_summary_line("RF train", rf_train_metrics)
    print_summary_line("RF validation", rf_val_metrics)
    print_summary_line("RF test", rf_test_metrics)

    print_summary_line("IF train", if_train_metrics)
    print_summary_line("IF validation", if_val_metrics)
    print_summary_line("IF test", if_test_metrics)

    print_summary_line("Hybrid train", hybrid_train_metrics)
    print_summary_line("Hybrid validation", hybrid_val_metrics)
    print_summary_line("Hybrid test", hybrid_test_metrics)

    print("\nRuntime summary:")
    print(f"Random Forest    : {rf_runtime:.2f} seconds ({rf_runtime / 60:.2f} minutes)")
    print(f"Isolation Forest : {if_runtime:.2f} seconds ({if_runtime / 60:.2f} minutes)")
    print(f"Hybrid           : {hybrid_runtime:.2f} seconds ({hybrid_runtime / 60:.2f} minutes)")


if __name__ == "__main__":
    main()