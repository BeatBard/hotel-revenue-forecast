import os
from typing import List, Tuple

import joblib
import json
import numpy as np
import pandas as pd
from sklearn.preprocessing import OneHotEncoder, RobustScaler


###############################################################################
# Configuration
###############################################################################

RC_DATA_DIR = os.path.join(
    os.path.dirname(os.path.dirname(__file__)),
    "notebooks",
    "revenue_center_data",
)
DEFAULT_OUTPUT_DIR = os.path.join(
    os.path.dirname(os.path.dirname(__file__)),
    "processed",
)
os.makedirs(DEFAULT_OUTPUT_DIR, exist_ok=True)


###############################################################################
# Utility helpers
###############################################################################

def _load_revenue_center(revenue_center_id: int) -> pd.DataFrame:
    """Load raw CSV for a revenue center by id.

    Parameters
    ----------
    revenue_center_id : int
        ID number 1-9.

    Returns
    -------
    pd.DataFrame
        Raw dataframe with Date column parsed.
    """
    fname = f"RevenueCenter_{revenue_center_id}_data.csv"
    fpath = os.path.join(RC_DATA_DIR, fname)
    if not os.path.isfile(fpath):
        raise FileNotFoundError(f"Could not find {fpath}.")

    df = pd.read_csv(fpath, parse_dates=["Date"])
    return df


def _basic_cleaning(df: pd.DataFrame) -> pd.DataFrame:
    """Remove obviously bad rows / cast types."""
    df = df.copy()
    # Remove zero-revenue rows (often data entry artefacts)
    df = df[df["CheckTotal"] > 0].reset_index(drop=True)

    # Ensure categorical dtypes for low-card columns
    cat_cols = [
        "MealPeriod",
        "IslamicPeriod",
        "TourismIntensity",
        "RevenueImpact",
    ]
    for col in cat_cols:
        if col in df.columns:
            df[col] = df[col].astype("category")
    return df


def _add_calendar_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add calendar and cyclical time features."""
    df = df.copy()
    df["Year"] = df["Date"].dt.year
    df["Month"] = df["Date"].dt.month
    df["Day"] = df["Date"].dt.day
    df["DayOfWeek"] = df["Date"].dt.dayofweek  # Monday=0
    df["DayOfYear"] = df["Date"].dt.dayofyear
    df["WeekOfYear"] = df["Date"].dt.isocalendar().week.astype(int)
    df["Quarter"] = df["Date"].dt.quarter
    df["IsWeekend"] = df["DayOfWeek"].isin([5, 6]).astype(int)
    df["IsMonthEnd"] = df["Date"].dt.is_month_end.astype(int)

    # Cyclical encodings
    df["Month_sin"] = np.sin(2 * np.pi * df["Month"] / 12)
    df["Month_cos"] = np.cos(2 * np.pi * df["Month"] / 12)
    df["DOW_sin"] = np.sin(2 * np.pi * df["DayOfWeek"] / 7)
    df["DOW_cos"] = np.cos(2 * np.pi * df["DayOfWeek"] / 7)
    df["DOY_sin"] = np.sin(2 * np.pi * df["DayOfYear"] / 365)
    df["DOY_cos"] = np.cos(2 * np.pi * df["DayOfYear"] / 365)
    return df


def _add_lag_features(
    df: pd.DataFrame,
    lags: List[int] = [1, 2, 3, 7, 14],
    windows: List[int] = [7, 14],
) -> pd.DataFrame:
    """Create lag and rolling windows per MealPeriod."""
    df = df.sort_values(["Date", "MealPeriod"]).reset_index(drop=True).copy()

    # Map MealPeriod to integer for convenience if not yet present
    if "MealPeriod_num" not in df.columns and "MealPeriod" in df.columns:
        mapping = {"Breakfast": 0, "Lunch": 1, "Dinner": 2}
        df["MealPeriod_num"] = df["MealPeriod"].map(mapping).astype(int)

    for lag in lags:
        df[f"lag_{lag}"] = (
            df.groupby("MealPeriod")[["CheckTotal"]]
            .shift(lag)
            .fillna(method="bfill")
        )
    for win in windows:
        roll_mean = (
            df.groupby("MealPeriod")["CheckTotal"]
            .rolling(win, min_periods=1)
            .mean()
            .reset_index(level=0, drop=True)
        )
        roll_std = (
            df.groupby("MealPeriod")["CheckTotal"]
            .rolling(win, min_periods=1)
            .std()
            .reset_index(level=0, drop=True)
        )
        df[f"roll_mean_{win}"] = roll_mean
        df[f"roll_std_{win}"] = roll_std
    return df


def _encode_categoricals(df: pd.DataFrame) -> Tuple[pd.DataFrame, OneHotEncoder]:
    """One-Hot encode selected categorical columns and return both df & encoder."""
    cat_cols = [
        "MealPeriod",
        "IslamicPeriod",
        "TourismIntensity",
        "RevenueImpact",
    ]
    existing = [c for c in cat_cols if c in df.columns]
    encoder = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    encoded = encoder.fit_transform(df[existing])
    enc_df = pd.DataFrame(
        encoded,
        columns=encoder.get_feature_names_out(existing),
        index=df.index,
    )
    df = pd.concat([df.drop(columns=existing), enc_df], axis=1)
    return df, encoder


def _train_test_split_chrono(
    df: pd.DataFrame,
    target: str = "CheckTotal",
    test_size: float = 0.2,
):
    """Chronological split preserving temporal order."""
    idx = int(len(df) * (1 - test_size))
    train_df = df.iloc[:idx].copy()
    test_df = df.iloc[idx:].copy()

    X_train = train_df.drop(columns=[target, "Date"]).values
    y_train = train_df[target].values
    X_test = test_df.drop(columns=[target, "Date"]).values
    y_test = test_df[target].values

    feature_names = train_df.drop(columns=[target, "Date"]).columns.tolist()
    return X_train, X_test, y_train, y_test, feature_names


def preprocess_revenue_center(
    revenue_center_id: int = 1,
    output_dir: str = DEFAULT_OUTPUT_DIR,
    scale_features: bool = True,
):
    """End-to-end preprocessing pipeline for a single revenue center.

    The function saves numpy arrays & metadata under
    f"{output_dir}/RevenueCenter_{id}" and returns arrays in memory.
    """
    out_dir = os.path.join(output_dir, f"RevenueCenter_{revenue_center_id}")
    os.makedirs(out_dir, exist_ok=True)

    # 1. Load & basic clean
    df = _load_revenue_center(revenue_center_id)
    df = _basic_cleaning(df)

    # 2. Feature engineering
    df = _add_calendar_features(df)
    df = _add_lag_features(df)
    df, encoder = _encode_categoricals(df)

    # 3. Remove any remaining NA rows (caused by early lags)
    df = df.dropna().reset_index(drop=True)

    # 4. Chronological train/test split
    (
        X_train,
        X_test,
        y_train,
        y_test,
        feature_names,
    ) = _train_test_split_chrono(df)

    # 5. Scaling (optional)
    scaler = None
    if scale_features:
        scaler = RobustScaler().fit(X_train)
        X_train = scaler.transform(X_train)
        X_test = scaler.transform(X_test)

    # 6. Persist artefacts
    np.save(os.path.join(out_dir, "X_train.npy"), X_train)
    np.save(os.path.join(out_dir, "X_test.npy"), X_test)
    np.save(os.path.join(out_dir, "y_train.npy"), y_train)
    np.save(os.path.join(out_dir, "y_test.npy"), y_test)

    joblib.dump(
        encoder,
        os.path.join(out_dir, "ohe_encoder.pkl"),
    )
    if scaler is not None:
        joblib.dump(
            scaler,
            os.path.join(out_dir, "feature_scaler.pkl"),
        )

    with open(os.path.join(out_dir, "feature_list.json"), "w") as fp:
        json.dump(feature_names, fp, indent=2)

    msg = (
        "✅ Preprocessing complete for RevenueCenter_"
        f"{revenue_center_id}. Saved to {out_dir}."
    )
    print(msg)
    return X_train, X_test, y_train, y_test, feature_names


if __name__ == "__main__":
    # Quick CLI usability
    import argparse

    parser = argparse.ArgumentParser(
        description="Preprocess revenue center data."
    )
    parser.add_argument(
        "--rc",
        type=int,
        default=1,
        help="Revenue center ID (1-9)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=DEFAULT_OUTPUT_DIR,
        help="Output directory",
    )
    args = parser.parse_args()

    preprocess_revenue_center(args.rc, args.output)
