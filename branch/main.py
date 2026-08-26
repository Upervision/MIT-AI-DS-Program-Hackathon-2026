"""End-to-end pipeline: load data -> engineer features -> train both models
-> cross-validate -> ensemble -> write submission.csv.

Usage:
    python main.py                 # looks for train.csv/test.csv in cwd
    python main.py --data-dir path/to/data

Requires `lightgbm` installed. Run `python -m src.eda` first if you want
to see the leakage-trap analysis that justifies the feature choices here.
"""
import argparse
import pandas as pd
import numpy as np

from src.data_loading import load_data
from src.feature_engineering import engineer
from src.train_lightgbm import cross_validate_lightgbm, fit_final_lightgbm
from src.train_mlp import cross_validate_mlp, fit_final_mlp
from src.ensemble import combine_predictions
from src.config import ENSEMBLE_CONFIDENCE_THRESHOLD


def main(data_dir="."):
    print("Loading data...")
    train, test = load_data(data_dir)

    print("Engineering features...")
    d_train, feat_cols, ref_medians = engineer(train)
    d_test, _, _ = engineer(test, ref_medians=ref_medians)
    X, y = d_train[feat_cols], train["target"].values
    X_test = d_test[feat_cols]
    classes_ = np.sort(np.unique(y))

    print("\nCross-validating LightGBM...")
    _, lgb_metrics = cross_validate_lightgbm(X, y)
    print(f"  LightGBM CV Accuracy={lgb_metrics['accuracy']:.4f}  Macro F1={lgb_metrics['macro_f1']:.4f}")

    print("\nCross-validating MLP...")
    _, mlp_metrics = cross_validate_mlp(X, y)
    print(f"  MLP CV Accuracy={mlp_metrics['accuracy']:.4f}  Macro F1={mlp_metrics['macro_f1']:.4f}")

    print("\nFitting final models on full training data...")
    final_lgb = fit_final_lightgbm(X, y)
    proba_lgb_test = final_lgb.predict_proba(X_test)

    final_mlp, scaler = fit_final_mlp(X, y)
    proba_mlp_test = final_mlp.predict_proba(scaler.transform(X_test))

    print(f"\nCombining predictions (confidence threshold={ENSEMBLE_CONFIDENCE_THRESHOLD})...")
    final_pred, override_mask = combine_predictions(
        proba_lgb_test, proba_mlp_test, classes_, threshold=ENSEMBLE_CONFIDENCE_THRESHOLD
    )
    print(f"  Rows overridden by MLP: {override_mask.sum()} / {len(final_pred)}")

    submission = pd.DataFrame({"row_id": test["row_id"], "target": final_pred.astype(int)})
    submission.to_csv("submission.csv", index=False)
    print("\nSaved submission.csv")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default=".", help="Directory containing train.csv/test.csv")
    args = parser.parse_args()
    main(args.data_dir)
