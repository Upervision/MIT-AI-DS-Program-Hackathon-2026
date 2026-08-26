"""Exploratory analysis that justifies dropping shift_reliability_index and
motion_elegance_score. Run directly to reproduce the findings:

    python -m src.eda
"""
import pandas as pd
import numpy as np
from scipy import stats

from .data_loading import load_data


def feature_target_correlation(train, feature_cols):
    """Spearman correlation of each feature with the ordinal target.
    A single feature scoring far above the rest is a red flag worth
    investigating before trusting it (see check_train_test_shift below)."""
    corrs = {}
    for c in feature_cols:
        valid = train[c].notna()
        corrs[c] = stats.spearmanr(train.loc[valid, c], train.loc[valid, "target"])[0]
    return pd.Series(corrs).sort_values(key=abs, ascending=False)


def check_train_test_shift(train, test, feature_cols):
    """Two-sample KS test per feature. A feature whose train/test
    distributions differ far more than the rest is a sign it may not carry
    the same meaning (or reliability) in the hidden test set."""
    rows = []
    for c in feature_cols:
        a, b = train[c].dropna(), test[c].dropna()
        ks_stat, p_value = stats.ks_2samp(a, b)
        rows.append((c, ks_stat, p_value))
    return pd.DataFrame(rows, columns=["feature", "ks_stat", "p_value"]).sort_values(
        "ks_stat", ascending=False
    )


def single_feature_accuracy(train, feature, n_splits=3, random_state=42):
    """How much can ONE feature alone predict the 7-class target? A
    legitimate sensor reading combined with several others might explain a
    lot; a single feature explaining more than a well-built composite of
    several honest sensors is suspicious on its own."""
    from sklearn.model_selection import StratifiedKFold
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.metrics import accuracy_score

    X = train[[feature]].values
    y = train["target"].values
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    accs = []
    for tr_idx, val_idx in cv.split(X, y):
        clf = RandomForestClassifier(n_estimators=150, max_depth=10, random_state=random_state, n_jobs=-1)
        clf.fit(X[tr_idx], y[tr_idx])
        accs.append(accuracy_score(y[val_idx], clf.predict(X[val_idx])))
    return float(np.mean(accs))


def run_eda(data_dir="."):
    train, test = load_data(data_dir)
    feature_cols = [c for c in train.columns if c not in ["row_id", "target"]]

    print("=== Spearman correlation with target ===")
    corr = feature_target_correlation(train, feature_cols)
    print(corr.round(3))

    print("\n=== Train/test distribution shift (KS test) ===")
    shift = check_train_test_shift(train, test, feature_cols)
    print(shift.to_string(index=False))

    print("\n=== Single-feature predictive power (sanity check on the leak) ===")
    for feat in ["shift_reliability_index", "motion_elegance_score"]:
        acc = single_feature_accuracy(train, feat)
        print(f"  {feat} ALONE: {acc:.4f} accuracy on a 7-class problem")

    print(
        "\nConclusion: shift_reliability_index has both the highest target "
        "correlation (0.96) and by far the largest train/test shift "
        "(10-25x every other feature) -- a leakage trap. "
        "motion_elegance_score has ~0 correlation with target -- pure noise. "
        "Both are excluded in src/feature_engineering.py."
    )
    return corr, shift


if __name__ == "__main__":
    run_eda()
