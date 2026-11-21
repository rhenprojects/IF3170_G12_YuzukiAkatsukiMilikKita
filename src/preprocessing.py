from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, RobustScaler, OrdinalEncoder, KBinsDiscretizer


NUMERIC_FEATURES = [
    "num_adults",
    "num_children",
    "weekend_nights",
    "weekday_nights",
    "days_before_arrival",
    "prior_cancellations",
    "prior_completed_bookings",
    "nightly_rate_idr",
    "special_requests",
    "arrival_year",
    "arrival_month",
    "arrival_day",
]
CATEGORICAL_FEATURES = ["meal_package", "room_category", "booking_channel"]
BINARY_FEATURES = ["needs_parking", "is_returning_guest"]
NB_BIN_FEATURES = ["days_before_arrival", "nightly_rate_idr"]
# Memperlakukan bulan dan tahun sebagai kategori buat logistic regression, nilainya dikit dan polanya gak berurutan.
LR_DATE_FEATURES = ["arrival_year", "arrival_month"]
LR_BIN_FEATURES = [col for col in NUMERIC_FEATURES if col not in LR_DATE_FEATURES] + ["total_nights", "total_guests"]
LR_ONE_HOT_FEATURES = CATEGORICAL_FEATURES + BINARY_FEATURES + LR_DATE_FEATURES


def add_features(data):
    result = data.copy()
    result["total_nights"] = result["weekend_nights"] + result["weekday_nights"]
    result["total_guests"] = result["num_adults"] + result["num_children"]
    return result


def build_imputer():
    return ColumnTransformer(
        [
            ("num", SimpleImputer(strategy="median"), NUMERIC_FEATURES),
            ("cat", SimpleImputer(strategy="most_frequent"), CATEGORICAL_FEATURES),
            ("binary", SimpleImputer(strategy="most_frequent"), BINARY_FEATURES),
        ],
        verbose_feature_names_out=False,
    ).set_output(transform="pandas")

def build_preparation_pipeline():
    imputer = build_imputer()

    final_num_features = NUMERIC_FEATURES + ["total_nights", "total_guests"]
    preprocessor = ColumnTransformer(
        [
            ("num", RobustScaler(), final_num_features),
            (
                "cat",
                OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                CATEGORICAL_FEATURES,
            ),
            ("binary", "passthrough", BINARY_FEATURES),
        ]
    )

    return Pipeline(
        [
            ("imputer", imputer),
            ("attribute_construction", FunctionTransformer(add_features)),
            ("preprocessing", preprocessor),
            ("pca", "passthrough"),
        ]
    )


def build_naive_bayes_pipeline(n_bins=12):
    """Naive Bayes track: continuous features are quantile-binned, the rest are ordinal-encoded."""
    ordinal_features = [f for f in NUMERIC_FEATURES if f not in NB_BIN_FEATURES]
    preprocessor = ColumnTransformer(
        [
            (
                "num",
                KBinsDiscretizer(
                    n_bins=n_bins, encode="ordinal", strategy="quantile", subsample=None
                ),
                NB_BIN_FEATURES,
            ),
            (
                "cat",
                OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1),
                ordinal_features + CATEGORICAL_FEATURES,
            ),
            ("binary", "passthrough", BINARY_FEATURES),
        ]
    )
    return Pipeline([("imputer", build_imputer()), ("preprocessing", preprocessor)])


# Referensi
# Basis expansion (model tetap linear di fitur turunan) dan binning sebagai basis piecewise constant:
#   - Hastie, Tibshirani, Friedman, The Elements of Statistical Learning ed. 2, subbab 5.1 dan 5.2
# Interaksi dua arah sebagai perkalian basis antarfitur (tensor product basis):
#   - Hastie, Tibshirani, Friedman, The Elements of Statistical Learning ed. 2, subbab 5.7
# Membentuk fitur bin, one-hot, dan interaksi dua arah buat logistic regression, aturannya cuma dari data latih.
class InteractionFeatures(TransformerMixin, BaseEstimator):
    def __init__(self, n_bins=10):
        self.n_bins = n_bins

    def fit(self, data, y=None):
        self.edges = {}
        for col in LR_BIN_FEATURES:
            self.edges[col] = np.unique(np.quantile(data[col], np.linspace(0, 1, self.n_bins + 1)[1:-1]))

        self.categories = {}
        for col in LR_ONE_HOT_FEATURES:
            self.categories[col] = np.unique(data[col].to_numpy(dtype=object))

        # Membuang interaksi yang isinya 0 terus, misalnya dua bin dari kolom yang sama.
        base = self.one_hot(data)
        self.keep = []
        for i in range(base.shape[1] - 1):
            pairs = base[:, [i]] * base[:, i + 1:]
            self.keep.append(pairs.sum(axis=0) > 0)
        return self

    # Ngasih tahu Pipeline sklearn kalau langkah ini udah di-fit, soalnya atributnya gak pakai akhiran _.
    def __sklearn_is_fitted__(self):
        return hasattr(self, "keep")

    def one_hot(self, data):
        blocks = []
        for col in LR_BIN_FEATURES:
            bins = np.digitize(data[col].to_numpy(dtype=float), self.edges[col])
            blocks.append(bins[:, None] == np.arange(len(self.edges[col]) + 1))
        for col in LR_ONE_HOT_FEATURES:
            values = data[col].to_numpy(dtype=object)
            blocks.append(values[:, None] == self.categories[col])
        return np.hstack(blocks).astype(float)

    def transform(self, data):
        base = self.one_hot(data)
        columns = [base]
        for i in range(base.shape[1] - 1):
            pairs = base[:, [i]] * base[:, i + 1:]
            columns.append(pairs[:, self.keep[i]])
        return np.hstack(columns)


# Gak pakai scaler soalnya semua kolom udah 0/1, standardisasi kolom yang jarang aktif malah bikin SGA meledak.
def build_logistic_regression_pipeline(n_bins=10):
    return Pipeline(
        [
            ("imputer", build_imputer()),
            ("attribute_construction", FunctionTransformer(add_features)),
            ("interaction", InteractionFeatures(n_bins=n_bins)),
        ]
    )


def prepare_data(data_dir="data", seed=42, *, use_split=False, n_splits=5, track="scaled"):
    """Prepare the first stratified fold, or an 80:20 hold-out if use_split is true.

    track="scaled" untuk KNN/SVM/DTL,
    track="naive_bayes" untuk Naive Bayes,
    track="logistic_regression" untuk Logistic Regression.
    """
    data_dir = Path(data_dir)
    np.random.seed(seed)

    train_df = pd.read_csv(data_dir / "train.csv", dtype={"id": str})
    test_df = pd.read_csv(data_dir / "test.csv", dtype={"id": str})

    selected_features = NUMERIC_FEATURES + CATEGORICAL_FEATURES + BINARY_FEATURES
    X = train_df[selected_features]
    y = train_df["is_canceled"]
    X_test = test_df[selected_features]

    if use_split:
        X_train, X_valid, y_train, y_valid = train_test_split(
            X, y, test_size=0.2, random_state=seed, stratify=y
        )
    else:
        cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
        train_idx, valid_idx = next(cv.split(X, y))
        X_train, X_valid = X.iloc[train_idx], X.iloc[valid_idx]
        y_train, y_valid = y.iloc[train_idx], y.iloc[valid_idx]

    if track == "naive_bayes":
        preparation_pipeline = build_naive_bayes_pipeline()
    elif track == "logistic_regression":
        preparation_pipeline = build_logistic_regression_pipeline()
    else:
        preparation_pipeline = build_preparation_pipeline()
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