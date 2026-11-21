"""
AIPI 510: Module 3 Project: Data/ML Engineering
- src/tune.py
Hasan Al-Quaid
=================================================
Grid search for the random forest's hyperparameters:
  1. Loads and splits the data the same way as train.py (same seed, same split)
  2. Trains one model for every combination in the grid from config.yaml
  3. Scores each one on the VALIDATION set only (the test set is never used here)
  4. Logs every combination to MLflow, grouped under one "grid_search" run
  5. Prints the ranking and the best settings to copy into config.yaml

After this finishes, I will put the best settings in config.yaml,
and run train.py once to train, test, and upload the final model.
"""
import argparse
import contextlib
import io
import os
import tempfile
import mlflow
import pandas as pd
import yaml
from sklearn.model_selection import ParameterGrid

# reuse the same steps as train.py, so tuning and the final model are built the same way
from train import load_data, split_data, build_model, evaluate 


def make_grid(cfg):
    """
    Build every combination of the hyperparameters listed in config.yaml

    Parameters:
        cfg (dict): Full config

    Returns:
        list: One dict of hyperparameters per combination
    """
    print("-------------------------- Start: Grid --------------------------------")
    grid_cfg = cfg["tuning"]["grid"]
    grid = list(ParameterGrid(grid_cfg))

    for name, values in grid_cfg.items():
        print(f"  {name}: {values}")
    sizes = [len(values) for values in grid_cfg.values()]

    print(f"\nCombinations: {' x '.join(str(s) for s in sizes)} = {len(grid)}")
    print("-------------------------- End: Grid ----------------------------------\n")

    return grid


def run_grid_search(cfg, grid, X_train, X_val, y_train, y_val):
    """
    Train and score one model per combination, logging each one to MLflow

    Parameters:
        cfg (dict): Full config
        grid (list): Hyperparameter combinations from make_grid()
        X_train, X_val (pd.DataFrame): Features for the training and validation sets
        y_train, y_val (pd.Series): Prices for the training and validation sets

    Returns:
        pd.DataFrame: One row per combination with its validation scores, best first
    """
    print("\n-------------------------- Start: Grid search -------------------------")
    # Every model is trained on the training set and scored on the validation set.
    # The test set is not used here, so it stays an honest check for the final model
    results = []
    total = len(grid)
    run_num = 1

    for params in grid:
        # use this combination's settings
        cfg["model"]["hyperparameters"] = params

        # build the model (its prints are hidden so they don't repeat 27 times)
        with contextlib.redirect_stdout(io.StringIO()):
            model = build_model(cfg)

        # train on the training set, score on the validation set
        model.fit(X_train, y_train)
        scores = evaluate(model, X_val, y_val, cfg["evaluation"]["metrics"])

        # rename the scores to val_mae, val_rmse, val_r2
        val_scores = {}
        for metric, value in scores.items():
            val_scores["val_" + metric] = value

        # log this combination as its own run inside the grid_search run
        with mlflow.start_run(run_name=f"grid_{run_num:02d}", nested=True):
            mlflow.log_params(params)
            mlflow.log_metrics(val_scores)

        # save one row: the settings plus their scores
        row = {}
        row.update(params)
        row.update(val_scores)
        results.append(row)

        print(f"[{run_num:2d}/{total}] {params}  ->  val MAE {scores['mae']:,.1f}")
        run_num += 1
    print("-------------------------- End: Grid search ---------------------------\n")

    # put the results in a table, lowest validation MAE (best) first
    results_df = pd.DataFrame(results)
    results_df = results_df.sort_values("val_mae").reset_index(drop=True)
    return results_df


def report_best(results, grid_keys):
    """
    Print the top combinations and the best settings to copy into config.yaml

    Parameters:
        results (pd.DataFrame): Grid search results, best first
        grid_keys (list): Names of the hyperparameters in the grid

    Returns:
        dict: The best hyperparameters
    """
    print("-------------------------- Start: Results -----------------------------")
    print("Top 5 combinations (lowest validation MAE first):")
    print(results.head(5).round(3).to_string(index=False))

    # Take the best settings from the first row (results are sorted best first)
    best = {}
    for name in grid_keys:
        best[name] = int(results.loc[0, name])

    print(f"\nBest validation MAE: ${results.loc[0, 'val_mae']:,.2f}")
    print("\nCopy these into config.yaml, then run train.py:")
    print("  hyperparameters:")

    for k, v in best.items():
        print(f"    {k}: {v}")
    print("-------------------------- End: Results -------------------------------\n")

    return best


def main():
    parser = argparse.ArgumentParser(description="Grid search for the random forest's hyperparameters.")
    parser.add_argument("--config", default="config.yaml", help="Path to the config file")
    args = parser.parse_args()
    cfg = yaml.safe_load(open(args.config))
    ds = cfg["dataset"]

    # load and split the data exactly like train.py (same seed -> same split)
    df = load_data(cfg)
    feature_cols = list(ds["features"]["ordinal"].keys()) + ds["features"]["numeric"]
    X = df[feature_cols]
    y = df[ds["target"]]
    X_train, X_val, X_test, y_train, y_val, y_test = split_data(X, y, ds["split_ratio"], cfg["reproducibility"]["seed"])

    # build the grid of hyperparameter combinations
    grid = make_grid(cfg)

    # train and score every combination, grouped under one MLflow run
    mlflow.set_tracking_uri(cfg["tracking"]["mlflow_tracking_uri"])
    mlflow.set_experiment(cfg["tracking"]["mlflow_experiment"])

    # one parent run called "grid_search"; each combination is logged inside it
    with mlflow.start_run(run_name="grid_search", description="Grid search over the random forest's hyperparameters"):
        mlflow.log_param("combinations", len(grid))

        # train and score every combination
        results = run_grid_search(cfg, grid, X_train, X_val, y_train, y_val)

        # find and print the best settings
        grid_keys = list(cfg["tuning"]["grid"].keys())
        best = report_best(results, grid_keys)

        # log the best settings, named best_n_estimators, best_max_depth, best_min_samples_leaf
        for name, value in best.items():
            mlflow.log_param("best_" + name, value)

        # log the best validation MAE
        best_val_mae = results.loc[0, "val_mae"]
        mlflow.log_metric("best_val_mae", best_val_mae)

        # save the full results table as a CSV and attach it to the run
        results_file = os.path.join(tempfile.gettempdir(), "grid_search_results.csv")
        results.to_csv(results_file, index=False)
        mlflow.log_artifact(results_file)

    print("View it with: mlflow ui --backend-store-uri " + cfg["tracking"]["mlflow_tracking_uri"])


if __name__ == "__main__":
    main()
