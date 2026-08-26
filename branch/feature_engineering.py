"""Data-quality fixes and engineered features. See src/eda.py for why
motion_elegance_score and shift_reliability_index are excluded.
"""
from .config import DROP_COLS


def engineer(df, ref_medians=None):
    """Apply data-quality fixes and add engineered features.

    Parameters
    ----------
    df : pd.DataFrame
        Raw train or test dataframe.
    ref_medians : dict or None
        Median values to use for imputation. Pass the dict returned when
        engineering the TRAINING set when engineering the TEST set, so test
        rows are imputed with train statistics (no leakage).

    Returns
    -------
    d : pd.DataFrame with engineered columns added
    feature_cols : list of column names to use as model input
    ref_medians : dict (pass this into the next call, for test data)
    """
    d = df.copy()
    raw_features = [c for c in df.columns if c not in DROP_COLS]

    # --- data quality: battery readings above 100% are physically impossible ---
    d["battery_over_flag"] = (d["battery_health_pct"] > 100).astype(int)
    d["battery_health_pct"] = d["battery_health_pct"].clip(0, 100)

    # --- missingness in joint_torque_variance may itself carry information ---
    d["joint_missing_flag"] = d["joint_torque_variance"].isna().astype(int)
    if ref_medians is None:
        ref_medians = {"joint_torque_variance": d["joint_torque_variance"].median()}
    d["joint_torque_variance"] = d["joint_torque_variance"].fillna(
        ref_medians["joint_torque_variance"]
    )

    # --- composite health score (0-1, higher = healthier) ---
    d["health_composite"] = (
        d["battery_health_pct"] / 100
        + d["sensor_calibration_score"]
        + d["self_diagnostic_score"] / 100
        + d["load_capacity_pct"]
        + d["task_completion_rate"]
        + d["vision_accuracy"]
        + (1 - d["joint_torque_variance"].clip(0, 0.8) / 0.8)
    ) / 7

    # --- rate features: ratios are hard for axis-aligned tree splits to
    # reconstruct from the raw columns alone ---
    d["error_rate_per_shift_hr"] = d["error_count_7d"] / (d["shift_hours_last_7d"] + 1)
    d["error_rate_per_uptime_hr"] = d["error_count_7d"] / (d["uptime_hrs"] + 0.5)
    d["maintenance_urgency"] = d["last_maintenance_days"] / (d["uptime_hrs"] + 1)
    d["cycle_efficiency"] = d["task_completion_rate"] / d["avg_cycle_time_sec"]
    d["avg_daily_shift_hrs"] = d["shift_hours_last_7d"] / 7.0
    d["workload_intensity"] = d["uptime_hrs"] / (d["avg_daily_shift_hrs"] + 0.5)

    # --- count of simultaneous "red flag" conditions ---
    d["red_flag_count"] = (
        (d["error_count_7d"] > 10).astype(int)
        + (d["motor_temperature_c"] > 65).astype(int)
        + (d["joint_torque_variance"] > 0.3).astype(int)
        + (d["sensor_calibration_score"] < 0.6).astype(int)
        + (d["task_completion_rate"] < 0.6).astype(int)
        + (d["last_maintenance_days"] > 45).astype(int)
    )

    feature_cols = raw_features + [
        "battery_over_flag",
        "joint_missing_flag",
        "health_composite",
        "error_rate_per_shift_hr",
        "error_rate_per_uptime_hr",
        "maintenance_urgency",
        "cycle_efficiency",
        "avg_daily_shift_hrs",
        "workload_intensity",
        "red_flag_count",
    ]
    return d, feature_cols, ref_medians


if __name__ == "__main__":
    from .data_loading import load_data

    train, test = load_data()
    d_train, feat_cols, med = engineer(train)
    d_test, _, _ = engineer(test, ref_medians=med)
    print(f"Engineered {len(feat_cols)} features from {len(train.columns) - 2} raw columns.")
    print(feat_cols)
