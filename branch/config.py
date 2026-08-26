"""Shared constants for the pipeline. Centralized here so every stage --
EDA, feature engineering, both models, and the ensemble -- uses the same
values instead of hardcoding them separately in each file.
"""

RANDOM_STATE = 42

# The two features flagged in the data dictionary for extra scrutiny.
# EDA (see src/eda.py) confirms both are safe -- even necessary -- to drop:
#   - shift_reliability_index: 0.96 correlation with target in train, but a
#     KS-statistic 10-25x every other feature between train and test -- a
#     classic leakage trap that looks great in training and fails to
#     generalize.
#   - motion_elegance_score: 0.004 correlation with target (not
#     significant) -- pure noise.
DROP_COLS = ["row_id", "target", "motion_elegance_score", "shift_reliability_index"]

# LightGBM hyperparameters. Slow learning rate + many estimators + moderate
# regularization; validated via 5-fold CV in the accompanying notebook.
LGB_PARAMS = dict(
    n_estimators=1200,
    learning_rate=0.02,
    max_depth=8,
    num_leaves=48,
    subsample=0.8,
    colsample_bytree=0.8,
    min_child_samples=10,
    reg_alpha=0.1,
    reg_lambda=0.1,
    random_state=RANDOM_STATE,
    n_jobs=-1,
    verbose=-1,
)

# MLP hyperparameters. Modest 2-layer network with early stopping --
# deliberately not tuned aggressively, since its value in this pipeline is
# algorithmic diversity from LightGBM, not raw standalone accuracy.
MLP_PARAMS = dict(
    hidden_layer_sizes=(100, 50),
    alpha=1e-3,
    early_stopping=True,
    n_iter_no_change=15,
    max_iter=300,
    random_state=RANDOM_STATE,
)

# Ensemble decision threshold. Validated (see notebook, "Ensemble" section)
# to sit close to the point that maximizes absolute net-correct gains --
# lower thresholds admit unreliable low-confidence rows, higher thresholds
# shrink the eligible row count faster than precision improves.
ENSEMBLE_CONFIDENCE_THRESHOLD = 0.5

N_CV_FOLDS = 5
