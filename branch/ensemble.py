"""Confidence-gated ensemble: combine LightGBM and MLP predictions.

Rule (validated in the accompanying notebook against several looser and
tighter alternatives, including full model-replacement and additional
voting models -- all of which tested worse):

    Use LightGBM's prediction everywhere by default.
    Override it ONLY where the MLP disagrees AND both models' own
    top-class confidence exceeds ENSEMBLE_CONFIDENCE_THRESHOLD.

Requiring agreement isn't quite what's implemented here -- this override
fires whenever the MLP disagrees with LightGBM and both are individually
confident in their own (different) top classes, which is the exact rule
that produced the validated 0.8578 -> 0.8583 result.
"""
import numpy as np

from .config import ENSEMBLE_CONFIDENCE_THRESHOLD


def combine_predictions(proba_lgb, proba_mlp, classes, threshold=ENSEMBLE_CONFIDENCE_THRESHOLD):
    """Combine two models' probability arrays into final class predictions.

    Parameters
    ----------
    proba_lgb, proba_mlp : np.ndarray, shape (n_rows, n_classes)
    classes : array of class labels, in the same column order as proba_*
    threshold : float, confidence bar both models must individually clear

    Returns
    -------
    final_pred : np.ndarray of predicted class labels
    override_mask : boolean array, True where the MLP's prediction was used
    """
    pred_lgb = classes[np.argmax(proba_lgb, axis=1)]
    pred_mlp = classes[np.argmax(proba_mlp, axis=1)]
    conf_lgb = proba_lgb.max(axis=1)
    conf_mlp = proba_mlp.max(axis=1)

    override_mask = (pred_mlp != pred_lgb) & (conf_mlp > threshold) & (conf_lgb > threshold)

    final_pred = pred_lgb.copy()
    final_pred[override_mask] = pred_mlp[override_mask]
    return final_pred, override_mask


if __name__ == "__main__":
    # Small smoke test with synthetic data -- exercises the logic without
    # needing trained models.
    classes = np.array([1, 2, 3])
    proba_lgb = np.array([[0.7, 0.2, 0.1], [0.3, 0.3, 0.4], [0.9, 0.05, 0.05]])
    proba_mlp = np.array([[0.7, 0.2, 0.1], [0.2, 0.2, 0.6], [0.4, 0.55, 0.05]])
    pred, mask = combine_predictions(proba_lgb, proba_mlp, classes, threshold=0.5)
    print("Predictions:", pred)
    print("Overridden:", mask)
    # Row 0: both models agree (class 1) -> no override.
    # Row 1: both models agree (class 3) -> no override.
    # Row 2: models disagree (1 vs 2) and both exceed 0.5 confidence -> overridden to MLP's class 2.
    assert list(pred) == [1, 3, 2], "smoke test failed"
    assert list(mask) == [False, False, True], "smoke test failed"
    print("Smoke test passed.")
