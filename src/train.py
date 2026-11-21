"""
AIPI 510: Module 3 Project: Data/ML Engineering
- src/train.py
Hasan Al-Quaid
=================================================
Trains the diamond price model:
  1. Downloads the cleaned data from Google Cloud Storage (or reads it locally)
  2. Splits it into train / validation / test sets (70 / 15 / 15)
  3. Builds the pipeline: ordinal encoding for cut, color, and clarity + a random forest
  4. Trains on the training set, then evaluates on the validation and test sets
  5. Logs the settings, scores, and model to MLflow
  6. Saves the model locally and uploads it to GCS (for the API)

All settings come from the config file (config.yaml by default).

Output:
    models/model.joblib
    gs://<bucket>/models/model.joblib
    an MLflow run (stored in mlflow.db and the mlruns/ folder)
"""
import argparse
import os
import tempfile
import joblib
import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
import yaml
from google.cloud import storage
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OrdinalEncoder

# global values
DATA_FILE = os.path.join(tempfile.gettempdir(), "diamonds_clean.csv")


def load_data(cfg):
    """
    Download the cleaned data from GCS, or read the local CSV when use_gcs is false

    Parameters:
        cfg (dict): Full config

    Returns:
        pd.DataFrame: Cleaned data
    """
    print("-------------------------- Start: Load data ---------------------------")
    st = cfg["storage"]
    if st["use_gcs"]:
        try:
            blob = storage.Client(project=st["project"]).bucket(st["bucket"]).blob(st["data_path"])
            blob.download_to_filename(DATA_FILE)
        except Exception as e:
            raise RuntimeError(f"Problem downloading the data from the cloud: {e}")
        print(f"Downloaded gs://{st['bucket']}/{st['data_path']}")
        df = pd.read_csv(DATA_FILE)
    else:
        raise ValueError("use_gcs is false in config.yaml, but this pipeline needs the data from the cloud")
    
    print(f"Rows: {df.shape[0]}  |  Columns: {df.shape[1]}")
    print("-------------------------- End: Load data -----------------------------\n")

    return df


def split_data(X, y, split_ratio, seed):
    """
    Split into train / validation / test sets

    Parameters:
        X (pd.DataFrame): Features
        y (pd.Series): Target (price)
        split_ratio (list): [train, val, test] fractions, e.g. [0.7, 0.15, 0.15]
        seed (int): Random seed I chose 

    Returns:
        tuple: (X_train, X_val, X_test, y_train, y_val, y_test)
    """
    print("-------------------------- Start: Split data --------------------------")
    # Split off the train set first, then split the rest in half into validation and test.
    # The seed makes the split the same every time the script runs
    train, val, test = split_ratio
    
    # first split: train vs. everything else (val + test = 30%)
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y,
        test_size = val + test,
        random_state = seed
        )

    # second split: cut the remaining 30% in half
    # 15% for validation and 15% for testing
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, 
        y_temp,
        test_size = test / (val + test),
        random_state = seed
        )

    total = len(X_train) + len(X_val) + len(X_test)
    print(f"Train:      {len(X_train)} rows ({len(X_train) / total:.0%})")
    print(f"Validation: {len(X_val)} rows ({len(X_val) / total:.0%})")
    print(f"Test:       {len(X_test)} rows ({len(X_test) / total:.0%})")
    print(f"{len(X_train)} + {len(X_val)} + {len(X_test)} = {len(X)}: {total == len(X)}")
    print("-------------------------- End: Split data ----------------------------\n")

    return X_train, X_val, X_test, y_train, y_val, y_test


def correlation_matrix(df):
    """
    Print the correlation matrix of all numeric columns.

    Parameters:
        df (pd.DataFrame): The training rows (with all columns)

    Returns:
        None
    """
    print("-------------------------- Start: Correlation matrix ------------------")
    # Chapter 2, section 7: remove one of any pair above a threshold (e.g., 0.8).
    corr = df.select_dtypes("number").corr()
    print(corr.round(2).to_string())

    print("\nThe following are above 0.8, and are redundant for the model:")
    print("x, y, z vs carat:")
    for col in ["x", "y", "z"]:
        print(f"  carat vs {col}: {corr.loc['carat', col]:.2f}  (above 0.8: {corr.loc['carat', col] > 0.8})")
    print("-------------------------- End: Correlation matrix --------------------\n")

def feature_engineering(cfg):
    """
    Feature engineering:
      - Feature selection: use carat, depth, table, cut, color, and clarity.
        x, y, and z are left out because they are redundant with carat, as shown in the correlation matrix.
      - Ordinal encoding: cut, color, and clarity have a natural order
        (e.g. Fair < Good < Ideal), so they are encoded worst -> best as 0, 1, 2, ...
      - Numeric columns are passed through unchanged.

    Parameters:
        cfg (dict): Full config

    Returns:
        ColumnTransformer: Unfitted feature engineering step
    """
    print("-------------------------- Start: Feature engineering -----------------")
    features = cfg["dataset"]["features"]
    ordinal = features["ordinal"]

    print(f"Numeric features (used as is): {features['numeric']}")
    print("Ordinal features (encoded worst -> best as 0, 1, 2, ...):")
    for col, order in ordinal.items():
        print(f"  {col}:")
        for i, value in enumerate(order):
            print(f"    {value} = {i}")
    print("Not used: x, y, z (see the correlation matrix)")

    preprocess = ColumnTransformer([
        ("ordinal", OrdinalEncoder(categories=list(ordinal.values())), list(ordinal.keys())),
        ("numeric", "passthrough", features["numeric"]),
    ])
    print("-------------------------- End: Feature engineering -------------------\n")

    return preprocess

def build_model(cfg):
    """
    Build the full pipeline: feature engineering + random forest. Keeping both in
    one pipeline means the API applies the exact same feature steps to new diamonds.

    Parameters:
        cfg (dict): Full config

    Returns:
        Pipeline: Unfitted scikit-learn pipeline
    """
    print("-------------------------- Start: Build model -------------------------")
    model_cfg = cfg["model"]
    rfr = RandomForestRegressor(
        random_state = cfg["reproducibility"]["seed"],
        n_jobs = -1,
        **model_cfg["hyperparameters"]
    )
    
    print(f"Model: random forest {model_cfg['hyperparameters']}")
    print("-------------------------- End: Build model ---------------------------\n")

    return Pipeline([("preprocess", feature_engineering(cfg)), ("model", rfr)])


def evaluate(model, X, y, metric_names):
    """
    Calculate the regression metrics listed in the config

    Parameters:
        model (Pipeline): Fitted pipeline
        X (pd.DataFrame): Features
        y (pd.Series): True prices
        metric_names (list): Which metrics to return, from mae, rmse, r2

    Returns:
        dict: Metric name -> value (MAE and RMSE are in US dollars)
    """
    pred = model.predict(X)
    all_metrics = {
        "mae": mean_absolute_error(y, pred),
        "rmse": float(np.sqrt(mean_squared_error(y, pred))),
        "r2": r2_score(y, pred),
    }
    return {name: all_metrics[name] for name in metric_names}


def train_and_evaluate(model, cfg, X_train, X_val, X_test, y_train, y_val, y_test):
    """
     Train the final model on the training set, then score it on the validation and test sets.
    Hyperparameters are picked by tune.py (validation only), so the test set is used only here, once.

    Parameters:
        model (Pipeline): Unfitted pipeline
        cfg (dict): Full config
        X_train, X_val, X_test (pd.DataFrame): Features for each set
        y_train, y_val, y_test (pd.Series): Prices for each set

    Returns:
        dict: {"val": {...}, "test": {...}} metrics for each set
    """
    print("-------------------------- Start: Train and evaluate ------------------")
    model.fit(X_train, y_train)
    print(f"Trained on {len(X_train)} diamonds")

    # The test set is only used here, once, at the end
    results = {}
    for name, X_set, y_set in [("val", X_val, y_val), ("test", X_test, y_test)]:
        results[name] = evaluate(model, X_set, y_set, cfg["evaluation"]["metrics"])
        print(f"\n{name}:")
        for metric, value in results[name].items():
            print(f"  {metric.upper():5s} {value:,.3f}")
    print("-------------------------- End: Train and evaluate --------------------\n")
    return results

def log_to_mlflow(cfg, model, results, config_path):
    """
    Log the settings, scores, config file, and trained model to MLflow

    Parameters:
        cfg (dict): Full config
        model (Pipeline): Fitted pipeline
        results (dict): Metrics for the validation and test sets
        config_path (str): Path to the config file (saved with the run)

    Returns:
        str: The MLflow run ID
    """
    print("-------------------------- Start: Log to MLflow -----------------------")
    tracking = cfg["tracking"]
    mlflow.set_tracking_uri(tracking["mlflow_tracking_uri"])
    mlflow.set_experiment(tracking["mlflow_experiment"])

    with mlflow.start_run(run_name=cfg["experiment_id"], description=cfg["description"]) as run:
        mlflow.set_tags({f"tag_{t}": "true" for t in cfg["tags"]} | {"dataset": cfg["dataset"]["name"]})
        mlflow.log_params({
            **cfg["model"]["hyperparameters"],
            "architecture": cfg["model"]["architecture"],
            "split_ratio": str(cfg["dataset"]["split_ratio"]),
            "seed": cfg["reproducibility"]["seed"],
            "features": str(list(cfg["dataset"]["features"]["ordinal"]) + cfg["dataset"]["features"]["numeric"]),
        })
        for split, metrics in results.items():
            mlflow.log_metrics({f"{split}_{k}": v for k, v in metrics.items()})
        mlflow.log_artifact(config_path)

        # MLflow checks saved models for safety and asks you to confirm the random forest's
        # internal tree type. This model was trained here, so that one type is marked as trusted
        mlflow.sklearn.log_model(model, name="model", skops_trusted_types=["sklearn.tree._tree.Tree"])

        print(f"Experiment: {tracking['mlflow_experiment']}  |  Run: {cfg['experiment_id']}")
        print(f"Run ID: {run.info.run_id}")
        print("Logged: parameters, metrics, config file, and model")
    print("View it with: mlflow ui --backend-store-uri " + tracking["mlflow_tracking_uri"])
    print("-------------------------- End: Log to MLflow -------------------------\n")

    return run.info.run_id


def save_model(model, cfg):
    """
    Save the model locally (compressed) and upload it to GCS for the API

    Parameters:
        model (Pipeline): Fitted pipeline
        cfg (dict): Full config

    Returns:
        None
    """
    print("-------------------------- Start: Save model --------------------------")
    output_path, st = cfg["model"]["output_path"], cfg["storage"]
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # compress = 3 works like zipping the file: same model, much smaller file
    joblib.dump(model, output_path, compress = 3)
    print(f"Saved {output_path}  ({os.path.getsize(output_path) / 1_000_000:.1f} MB)")

    if st["use_gcs"]:
        blob = storage.Client(project=st["project"]).bucket(st["bucket"]).blob(st["model_path"])
        blob.upload_from_filename(output_path)
        print(f"Uploaded model to gs://{st['bucket']}/{st['model_path']}")
        print(f"Public URL: {blob.public_url}")

        # verify the model is now in the bucket, should be True
        print(f"File exists in bucket (should be True): {blob.exists()}")
    print("-------------------------- End: Save model ----------------------------\n")


def main():
    parser = argparse.ArgumentParser(description="Train the diamond price model.")
    parser.add_argument("--config", default="config.yaml", help="Path to the experiment config file")
    args = parser.parse_args()
    cfg = yaml.safe_load(open(args.config))
    ds = cfg["dataset"]
    print(f"Experiment {cfg['experiment_id']}: {cfg['name']}\n")

    # load the clean data from the cloud
    df = load_data(cfg)

    # pick the inputs (features) and the answer (price), then split the data
    feature_cols = list(ds["features"]["ordinal"].keys()) + ds["features"]["numeric"]
    X = df[feature_cols]        # X: carat, depth, table, cut, color, clarity
    y = df[ds["target"]]        # y: price
    X_train, X_val, X_test, y_train, y_val, y_test = split_data(X, y, ds["split_ratio"], cfg["reproducibility"]["seed"])

    # check why x, y, and z are left out, using only the training rows
    correlation_matrix(df.loc[X_train.index])

    # build, train, and score the model
    model = build_model(cfg)
    results = train_and_evaluate(model, cfg, X_train, X_val, X_test, y_train, y_val, y_test)

    # track the run, then save the model for the API
    log_to_mlflow(cfg, model, results, args.config)

    # save and upload the final model for the API
    save_model(model, cfg)


if __name__ == "__main__":
    main()
