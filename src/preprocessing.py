from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, StandardScaler


def add_features(data):
    result = data.copy()
    result["total_nights"] = result["weekend_nights"] + result["weekday_nights"]
    result["total_guests"] = result["num_adults"] + result["num_children"]
    return result


def prepare_data(data_dir="data", seed=42):
    data_dir = Path(data_dir)
    np.random.seed(seed)

    train_df = pd.read_csv(data_dir / "train.csv", dtype={"id": str})
    test_df = pd.read_csv(data_dir / "test.csv", dtype={"id": str})

    train_set, val_set = train_test_split(
        train_df,
        test_size=0.2,
        random_state=seed,
        stratify=train_df["is_canceled"],
    )

    train_set = train_set.copy()
    val_set = val_set.copy()

    target_col = "is_canceled"

    X_train = train_set.drop(columns=["id", target_col]).copy()
    X_valid = val_set.drop(columns=["id", target_col]).copy()
    X_test = test_df.drop(columns=["id"]).copy()

    y_train = train_set[target_col].copy()
    y_valid = val_set[target_col].copy()

    cat_features = X_train.select_dtypes(include=["object", "category"]).columns.tolist()

    for col in ["needs_parking", "is_returning_guest", "arrival_month", "arrival_year"]:
        if col not in cat_features:
            cat_features.append(col)

    num_features = []
    for col in X_train.select_dtypes(include="number").columns:
        if col not in cat_features:
            num_features.append(col)

    final_num_features = num_features.copy()
    final_num_features.append("total_nights")
    final_num_features.append("total_guests")

    num_pipeline = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )

    cat_pipeline = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]
    )

    preprocessor = ColumnTransformer(
        [
            ("numerik", num_pipeline, final_num_features),
            ("kategorikal", cat_pipeline, cat_features),
        ]
    )

    preparation_pipeline = Pipeline(
        [
            ("atribut_baru", FunctionTransformer(add_features, validate=False)),
            ("preprocessing", preprocessor),
        ]
    )

    X_train_prepared = preparation_pipeline.fit_transform(X_train)
    X_valid_prepared = preparation_pipeline.transform(X_valid)
    X_test_prepared = preparation_pipeline.transform(X_test)

    return (
        X_train_prepared,
        X_valid_prepared,
        X_test_prepared,
        y_train,
        y_valid,
        test_df["id"].copy(),
        preparation_pipeline,
    )


if __name__ == "__main__":
    (
        X_train_prepared,
        X_valid_prepared,
        X_test_prepared,
        y_train,
        y_valid,
        test_ids,
        preparation_pipeline,
    ) = prepare_data()

    print("X_train_prepared:", X_train_prepared.shape)
    print("X_valid_prepared:", X_valid_prepared.shape)
    print("X_test_prepared :", X_test_prepared.shape)
    print("Target training:", y_train.shape)
    print("Target validation:", y_valid.shape)
    print("Missing values hasil training:", np.isnan(X_train_prepared).sum())
    print("Missing values hasil validation:", np.isnan(X_valid_prepared).sum())
    print("Missing values hasil test:", np.isnan(X_test_prepared).sum())