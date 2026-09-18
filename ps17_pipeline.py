"""
PS 17 - Smart Waste Collection Priority Predictor
End-to-end pipeline: EDA -> Feature Engineering -> Baseline Model -> Main Model -> Evaluation -> Priority List

Run: python ps17_pipeline.py
Expects: waste_collection_dataset.csv in the same folder (or edit DATA_PATH below)
"""

import os
import warnings
warnings.filterwarnings("ignore")

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.metrics import (
    mean_absolute_error, mean_squared_error, r2_score,
    f1_score, precision_score, recall_score, classification_report, confusion_matrix
)
import joblib

DATA_PATH = "waste_collection_dataset.csv"
PLOTS_DIR = "plots"
os.makedirs(PLOTS_DIR, exist_ok=True)

pd.set_option("display.width", 120)


# ============================================================
# STEP 1: LOAD DATA
# ============================================================
def load_data(path):
    df = pd.read_csv(path, parse_dates=["timestamp", "last_collection"])
    df = df.sort_values(["bin_id", "timestamp"]).reset_index(drop=True)
    print(f"[STEP 1] Loaded {df.shape[0]} rows, {df.shape[1]} columns")
    print(df.head())
    return df


# ============================================================
# STEP 2: EXPLORATORY DATA ANALYSIS (EDA)
# ============================================================
def run_eda(df):
    print("\n[STEP 2] Running EDA...")

    # 2.1 Basic info
    print("\n-- dtypes --")
    print(df.dtypes)
    print("\n-- missing values --")
    print(df.isnull().sum())
    print("\n-- summary stats --")
    print(df.describe())

    # 2.2 Class balance for overflow
    print("\n-- overflow class balance --")
    print(df["overflow"].value_counts(normalize=True))

    # 2.3 Fill level distribution
    plt.figure(figsize=(8, 5))
    sns.histplot(df["fill_level"], bins=30, kde=True)
    plt.title("Distribution of Bin Fill Level")
    plt.xlabel("Fill Level (%)")
    plt.savefig(f"{PLOTS_DIR}/01_fill_level_distribution.png", bbox_inches="tight")
    plt.close()

    # 2.4 Fill level by zone
    plt.figure(figsize=(8, 5))
    sns.boxplot(data=df, x="location_zone", y="fill_level")
    plt.title("Fill Level by Zone")
    plt.savefig(f"{PLOTS_DIR}/02_fill_level_by_zone.png", bbox_inches="tight")
    plt.close()

    # 2.5 Fill level by day type
    plt.figure(figsize=(6, 5))
    sns.boxplot(data=df, x="day_type", y="fill_level")
    plt.title("Fill Level: Weekday vs Weekend")
    plt.savefig(f"{PLOTS_DIR}/03_fill_level_by_daytype.png", bbox_inches="tight")
    plt.close()

    # 2.6 Time series for a sample bin (shows fill-and-reset pattern)
    sample_bin = df["bin_id"].unique()[0]
    sample = df[df["bin_id"] == sample_bin]
    plt.figure(figsize=(12, 4))
    plt.plot(sample["timestamp"], sample["fill_level"], marker="o", markersize=2)
    plt.title(f"Fill Level Over Time - {sample_bin}")
    plt.xlabel("Time")
    plt.ylabel("Fill Level (%)")
    plt.xticks(rotation=45)
    plt.savefig(f"{PLOTS_DIR}/04_sample_bin_timeseries.png", bbox_inches="tight")
    plt.close()

    # 2.7 Correlation heatmap (numeric features only)
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    plt.figure(figsize=(8, 6))
    sns.heatmap(df[numeric_cols].corr(), annot=True, fmt=".2f", cmap="coolwarm")
    plt.title("Correlation Heatmap")
    plt.savefig(f"{PLOTS_DIR}/05_correlation_heatmap.png", bbox_inches="tight")
    plt.close()

    print(f"\n[STEP 2] EDA plots saved to '{PLOTS_DIR}/'")


# ============================================================
# STEP 3: FEATURE ENGINEERING
# ============================================================
def engineer_features(df):
    print("\n[STEP 3] Engineering features...")
    df = df.copy()

    # 3.1 Time-since-last-collection (hours) -- key predictive signal
    df["hours_since_collection"] = (
        df["timestamp"] - df["last_collection"]
    ).dt.total_seconds() / 3600

    # 3.2 Calendar features
    df["hour_of_day"] = df["timestamp"].dt.hour
    df["day_of_week"] = df["timestamp"].dt.dayofweek

    # 3.3 Lag features per bin (own engineered lags, not relying only on
    #     the provided 'recent_fill_trend' column)
    df = df.sort_values(["bin_id", "timestamp"])
    df["fill_lag_1"] = df.groupby("bin_id")["fill_level"].shift(1)
    df["fill_lag_2"] = df.groupby("bin_id")["fill_level"].shift(2)
    df["fill_diff_1"] = df["fill_level"] - df["fill_lag_1"]

    # 3.4 Rolling average (last 3 readings, excluding current)
    df["fill_roll_mean_3"] = (
        df.groupby("bin_id")["fill_level"]
        .shift(1)
        .rolling(3)
        .mean()
        .reset_index(level=0, drop=True)
    )

    # 3.5 Target: next-period fill level (shift -1 within each bin)
    df["target_next_fill"] = df.groupby("bin_id")["fill_level"].shift(-1)
    df["target_next_overflow"] = df.groupby("bin_id")["overflow"].shift(-1)

    # 3.6 Encode categoricals
    df["day_type_enc"] = df["day_type"].map({"weekday": 0, "weekend": 1})
    df = pd.get_dummies(df, columns=["location_zone"], prefix="zone")

    # 3.7 Drop rows with NaNs created by lag/shift (first rows per bin, last row per bin)
    before = len(df)
    df = df.dropna().reset_index(drop=True)
    print(f"[STEP 3] Dropped {before - len(df)} rows with NaNs from lag/shift features")
    print(f"[STEP 3] Final feature set shape: {df.shape}")

    return df


# ============================================================
# STEP 4: TRAIN / TEST SPLIT (time-based, not random shuffle)
# ============================================================
def time_based_split(df, test_frac=0.2):
    print("\n[STEP 4] Splitting data (time-based, per bin)...")
    df = df.sort_values(["bin_id", "timestamp"])
    train_parts, test_parts = [], []
    for _, group in df.groupby("bin_id"):
        cutoff = int(len(group) * (1 - test_frac))
        train_parts.append(group.iloc[:cutoff])
        test_parts.append(group.iloc[cutoff:])
    train_df = pd.concat(train_parts).reset_index(drop=True)
    test_df = pd.concat(test_parts).reset_index(drop=True)
    print(f"[STEP 4] Train: {train_df.shape[0]} rows | Test: {test_df.shape[0]} rows")
    return train_df, test_df


BASE_FEATURE_COLS = [
    "hours_since_collection", "hour_of_day", "day_of_week",
    "fill_lag_1", "fill_lag_2", "fill_diff_1", "fill_roll_mean_3",
    "day_type_enc", "weather_flag", "event_flag", "nearby_activity",
]


def get_feature_cols(df):
    """Base features + whichever zone_* dummy columns exist after one-hot encoding."""
    zone_cols = [c for c in df.columns if c.startswith("zone_")]
    return BASE_FEATURE_COLS + zone_cols


def get_xy(df, target_col, feature_cols):
    cols = [c for c in feature_cols if c in df.columns]
    return df[cols], df[target_col]


# ============================================================
# STEP 5: REGRESSION TRACK - predict next fill level (%)
# ============================================================
def run_regression_track(train_df, test_df):
    print("\n[STEP 5] Regression track: predicting next fill level (%)")

    feature_cols = get_feature_cols(train_df)
    X_train, y_train = get_xy(train_df, "target_next_fill", feature_cols)
    X_test, y_test = get_xy(test_df, "target_next_fill", feature_cols)

    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)

    # 5.1 Baseline: Linear Regression
    lr = LinearRegression()
    lr.fit(X_train_s, y_train)
    lr_pred = lr.predict(X_test_s)
    lr_mae = mean_absolute_error(y_test, lr_pred)
    lr_rmse = np.sqrt(mean_squared_error(y_test, lr_pred))
    print(f"  Baseline Linear Regression -> MAE: {lr_mae:.2f}, RMSE: {lr_rmse:.2f}, R2: {r2_score(y_test, lr_pred):.3f}")

    # 5.2 Main model: Random Forest Regressor
    rf = RandomForestRegressor(n_estimators=300, max_depth=10, random_state=42, n_jobs=-1)
    rf.fit(X_train, y_train)  # tree models don't need scaling
    rf_pred = rf.predict(X_test)
    rf_mae = mean_absolute_error(y_test, rf_pred)
    rf_rmse = np.sqrt(mean_squared_error(y_test, rf_pred))
    print(f"  Random Forest Regressor    -> MAE: {rf_mae:.2f}, RMSE: {rf_rmse:.2f}, R2: {r2_score(y_test, rf_pred):.3f}")

    # 5.3 Predicted vs Actual chart
    plt.figure(figsize=(6, 6))
    plt.scatter(y_test, rf_pred, alpha=0.3, s=10)
    plt.plot([0, 100], [0, 100], "r--")
    plt.xlabel("Actual Next Fill Level (%)")
    plt.ylabel("Predicted Next Fill Level (%)")
    plt.title("Random Forest: Predicted vs Actual Fill Level")
    plt.savefig(f"{PLOTS_DIR}/06_regression_pred_vs_actual.png", bbox_inches="tight")
    plt.close()

    # 5.4 Feature importance
    importances = pd.Series(rf.feature_importances_, index=X_train.columns).sort_values(ascending=False)
    plt.figure(figsize=(8, 5))
    importances.plot(kind="barh")
    plt.title("Random Forest Regressor - Feature Importance")
    plt.gca().invert_yaxis()
    plt.savefig(f"{PLOTS_DIR}/07_regression_feature_importance.png", bbox_inches="tight")
    plt.close()

    return rf, scaler, {"linear_regression": {"mae": lr_mae, "rmse": lr_rmse},
                         "random_forest": {"mae": rf_mae, "rmse": rf_rmse}}


# ============================================================
# STEP 6: CLASSIFICATION TRACK - predict next-period overflow risk
# ============================================================
def run_classification_track(train_df, test_df):
    print("\n[STEP 6] Classification track: predicting next-period overflow risk")

    feature_cols = get_feature_cols(train_df)
    X_train, y_train = get_xy(train_df, "target_next_overflow", feature_cols)
    X_test, y_test = get_xy(test_df, "target_next_overflow", feature_cols)

    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)

    # 6.1 Baseline: Logistic Regression
    logr = LogisticRegression(max_iter=1000)
    logr.fit(X_train_s, y_train)
    logr_pred = logr.predict(X_test_s)
    logr_f1 = f1_score(y_test, logr_pred)
    print(f"  Baseline Logistic Regression -> F1: {logr_f1:.3f}, "
          f"Precision: {precision_score(y_test, logr_pred):.3f}, Recall: {recall_score(y_test, logr_pred):.3f}")

    # 6.2 Main model: Random Forest Classifier
    rfc = RandomForestClassifier(n_estimators=300, max_depth=10, random_state=42, n_jobs=-1)
    rfc.fit(X_train, y_train)
    rfc_pred = rfc.predict(X_test)
    rfc_f1 = f1_score(y_test, rfc_pred)
    print(f"  Random Forest Classifier    -> F1: {rfc_f1:.3f}, "
          f"Precision: {precision_score(y_test, rfc_pred):.3f}, Recall: {recall_score(y_test, rfc_pred):.3f}")
    print("\n  Classification report (Random Forest):")
    print(classification_report(y_test, rfc_pred))

    # 6.3 Confusion matrix
    cm = confusion_matrix(y_test, rfc_pred)
    plt.figure(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=["No Overflow", "Overflow"], yticklabels=["No Overflow", "Overflow"])
    plt.title("Confusion Matrix - Random Forest Classifier")
    plt.ylabel("Actual")
    plt.xlabel("Predicted")
    plt.savefig(f"{PLOTS_DIR}/08_confusion_matrix.png", bbox_inches="tight")
    plt.close()

    return rfc, scaler, {"logistic_regression": {"f1": logr_f1},
                          "random_forest": {"f1": rfc_f1}}


# ============================================================
# STEP 7: BUILD PRIORITY COLLECTION LIST (the product output)
# ============================================================
def build_priority_list(test_df, reg_model, clf_model):
    print("\n[STEP 7] Building priority collection list...")

    feature_cols = get_feature_cols(test_df)

    latest = (
        test_df.sort_values("timestamp")
        .groupby("bin_id")
        .tail(1)
        .reset_index(drop=True)
    )
    X_latest, _ = get_xy(latest, "target_next_fill", feature_cols)

    latest["predicted_next_fill"] = reg_model.predict(X_latest)
    latest["predicted_overflow_risk"] = clf_model.predict_proba(X_latest)[:, 1]

    def priority_label(row):
        if row["predicted_next_fill"] >= 90 or row["predicted_overflow_risk"] >= 0.7:
            return "URGENT"
        elif row["predicted_next_fill"] >= 70 or row["predicted_overflow_risk"] >= 0.4:
            return "SOON"
        else:
            return "LOW"

    latest["priority"] = latest.apply(priority_label, axis=1)

    # location_zone was one-hot encoded away; reconstruct a readable zone label
    # zone_* dummy columns look like "zone_ZONE_A" -> strip the "zone_" prefix only
    zone_cols = [c for c in latest.columns if c.startswith("zone_")]
    if zone_cols:
        latest["location_zone"] = latest[zone_cols].idxmax(axis=1).str.replace("zone_", "", regex=False)
    else:
        latest["location_zone"] = "UNKNOWN"

    priority_order = {"URGENT": 0, "SOON": 1, "LOW": 2}
    priority_list = latest[[
        "bin_id", "location_zone", "predicted_next_fill", "predicted_overflow_risk", "priority"
    ]].copy()

    priority_list = priority_list.sort_values(
        by="priority", key=lambda s: s.map(priority_order)
    ).reset_index(drop=True)

    priority_list.to_csv("priority_collection_list.csv", index=False)
    print("[STEP 7] Saved priority_collection_list.csv")
    print(priority_list.head(15))

    return priority_list


# ============================================================
# STEP 8: SAVE MODELS
# ============================================================
def save_models(reg_model, clf_model):
    joblib.dump(reg_model, "fill_level_regressor.joblib")
    joblib.dump(clf_model, "overflow_classifier.joblib")
    print("\n[STEP 8] Saved fill_level_regressor.joblib and overflow_classifier.joblib")


# ============================================================
# MAIN
# ============================================================
if __name__ == "__main__":
    df = load_data(DATA_PATH)
    run_eda(df)
    df_feat = engineer_features(df)
    train_df, test_df = time_based_split(df_feat)

    reg_model, reg_scaler, reg_metrics = run_regression_track(train_df, test_df)
    clf_model, clf_scaler, clf_metrics = run_classification_track(train_df, test_df)

    priority_list = build_priority_list(test_df, reg_model, clf_model)
    save_models(reg_model, clf_model)

    print("\n=== SUMMARY ===")
    print("Regression (next fill %):", reg_metrics)
    print("Classification (overflow risk):", clf_metrics)
    print("\nDone. Check the 'plots/' folder for charts and priority_collection_list.csv for output.")
