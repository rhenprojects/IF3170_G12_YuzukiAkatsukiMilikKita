from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression as SklearnLogisticRegression
from sklearn.model_selection import StratifiedKFold

from algorithms.logistic_regression import LogisticRegression
from model_io import load_model, save_model
from preprocessing import BINARY_FEATURES, CATEGORICAL_FEATURES, NUMERIC_FEATURES, add_features, build_imputer

# Referensi
# Binning fitur angka (basis piecewise constant) dan interaksi antarfitur (tensor product):
#   - Hastie, Tibshirani, Friedman, The Elements of Statistical Learning ed. 2, subbab 5.2 dan 5.7
# Pemilihan hyperparameter dengan k-fold cross-validation, preprocessing di-fit di dalam tiap fold:
#   - Hastie, Tibshirani, Friedman, The Elements of Statistical Learning ed. 2, subbab 7.10.1 dan 7.10.2

ROOT = Path(__file__).resolve().parent.parent
SEED = 42

N_BINS = 10

CONFIGS = [
    {"penalty": "l2", "reg_lambda": 0.0003},
    {"penalty": "l2", "reg_lambda": 0.001},
    {"penalty": "l2", "reg_lambda": 0.003},
    {"penalty": "l1", "reg_lambda": 0.0003},
    {"penalty": "elasticnet", "reg_lambda": 0.001, "l1_ratio": 0.5},
]

SELECTED_FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES + BINARY_FEATURES
# Memperlakukan bulan dan tahun kedatangan sebagai kategori karena nilainya sedikit dan polanya tidak berurutan
DATE_FEATURES = ["arrival_year", "arrival_month"]
BIN_FEATURES = [col for col in NUMERIC_FEATURES if col not in DATE_FEATURES] + ["total_nights", "total_guests"]
ONE_HOT_FEATURES = CATEGORICAL_FEATURES + BINARY_FEATURES + DATE_FEATURES


# Membentuk fitur bin, one-hot, dan interaksi dua arah, semua aturan dipelajari dari data latih saja
class InteractionFeatures:
    def fit(self, data):
        self.imputer = build_imputer().fit(data[SELECTED_FEATURES])
        data = self.clean(data)

        self.edges = {}
        for col in BIN_FEATURES:
            self.edges[col] = np.unique(np.quantile(data[col], np.linspace(0, 1, N_BINS + 1)[1:-1]))

        self.categories = {}
        for col in ONE_HOT_FEATURES:
            self.categories[col] = np.unique(data[col].to_numpy(dtype=object))

        # Membuang interaksi yang selalu 0 di data latih, misalnya dua bin dari kolom yang sama
        base = self.one_hot(data)
        self.keep = []
        for i in range(base.shape[1] - 1):
            pairs = base[:, [i]] * base[:, i + 1:]
            self.keep.append(pairs.sum(axis=0) > 0)
        return self

    def clean(self, data):
        return add_features(self.imputer.transform(data[SELECTED_FEATURES]))

    def one_hot(self, data):
        blocks = []
        for col in BIN_FEATURES:
            bins = np.digitize(data[col].to_numpy(dtype=float), self.edges[col])
            blocks.append(bins[:, None] == np.arange(len(self.edges[col]) + 1))
        for col in ONE_HOT_FEATURES:
            values = data[col].to_numpy(dtype=object)
            blocks.append(values[:, None] == self.categories[col])
        return np.hstack(blocks).astype(float)

    def transform(self, data):
        base = self.one_hot(self.clean(data))
        columns = [base]
        for i in range(base.shape[1] - 1):
            pairs = base[:, [i]] * base[:, i + 1:]
            columns.append(pairs[:, self.keep[i]])
        return np.hstack(columns)


def macro_f1(y_true, y_pred):
    f1 = []
    for label in [0, 1]:
        tp = np.sum((y_pred == label) & (y_true == label))
        fp = np.sum((y_pred == label) & (y_true != label))
        fn = np.sum((y_pred != label) & (y_true == label))
        f1.append(2 * tp / (2 * tp + fp + fn))
    return np.mean(f1)


def find_threshold(y_true, prob):
    best_score, best_threshold = 0, 0.5
    for threshold in np.arange(0.20, 0.65, 0.05):
        score = macro_f1(y_true, (prob >= threshold).astype(int))
        if score > best_score:
            best_score, best_threshold = score, round(threshold, 2)
    return best_score, best_threshold


def make_model(config):
    return LogisticRegression(learning_rate=0.005, n_epochs=10, lr_decay=0.8, random_state=SEED, **config)


# Menyetarakan C sklearn dengan reg_lambda, C = 1 / (reg_lambda * n) karena sklearn menjumlahkan loss
def make_sklearn_model(config, n_samples):
    l1_ratio = {"l2": 0.0, "l1": 1.0, "elasticnet": config.get("l1_ratio")}[config["penalty"]]
    C = 1 / (config["reg_lambda"] * n_samples)
    solver = "lbfgs" if l1_ratio == 0 else "saga"
    return SklearnLogisticRegression(C=C, l1_ratio=l1_ratio, solver=solver, max_iter=1000)


# Menghitung peluang out-of-fold, tiap baris diprediksi model yang tidak dilatih dengan baris itu
def cross_val_prob(data, y, folds, make):
    prob = np.zeros(len(y))
    for train_idx, valid_idx in folds:
        features = InteractionFeatures().fit(data.iloc[train_idx])
        X_train = features.transform(data.iloc[train_idx])
        X_valid = features.transform(data.iloc[valid_idx])
        model = make(len(train_idx)).fit(X_train, y[train_idx])

        # Menyamakan keluaran model sendiri (1 kolom) dengan sklearn (2 kolom)
        valid_prob = model.predict_proba(X_valid)
        prob[valid_idx] = valid_prob if valid_prob.ndim == 1 else valid_prob[:, 1]
    return prob


def main():
    train_df = pd.read_csv(ROOT / "data" / "train.csv", dtype={"id": str})
    test_df = pd.read_csv(ROOT / "data" / "test.csv", dtype={"id": str})
    y = train_df["is_canceled"].to_numpy()

    folds = list(StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED).split(train_df, y))

    best = None
    for config in CONFIGS:
        prob = cross_val_prob(train_df, y, folds, lambda n, config=config: make_model(config))
        score, threshold = find_threshold(y, prob)
        print(f"{str(config):<62} macro F1 CV = {score:.4f} (threshold {threshold})")

        if best is None or score > best["score"]:
            best = {"score": score, "threshold": threshold, "config": config}

    print(f"\nModel terbaik: {best['config']}, threshold {best['threshold']}, macro F1 CV = {best['score']:.4f}")

    sklearn_prob = cross_val_prob(train_df, y, folds, lambda n: make_sklearn_model(best["config"], n))
    sklearn_score = macro_f1(y, (sklearn_prob >= best["threshold"]).astype(int))
    print(f"sklearn dengan pengaturan yang sama: macro F1 CV = {sklearn_score:.4f}")

    # Melatih ulang model final dengan seluruh train.csv
    features = InteractionFeatures().fit(train_df)
    X_full = features.transform(train_df)
    X_test = features.transform(test_df)
    model = make_model(best["config"]).fit(X_full, y)
    print(f"\nJumlah fitur model final: {X_full.shape[1]}")

    output_dir = ROOT / "output"
    output_dir.mkdir(exist_ok=True)
    model_path = output_dir / "logistic_regression.pkl"
    save_model(model, model_path)

    loaded = load_model(model_path)
    prediction = loaded.predict(X_test, threshold=best["threshold"])
    assert (prediction == model.predict(X_test, threshold=best["threshold"])).all(), "Model hasil load berbeda!"
    submission = pd.DataFrame({"id": test_df["id"], "is_canceled": prediction})
    submission.to_csv(output_dir / "submission_logistic_regression.csv", index=False)
    print("Model dan submission disimpan di folder output/")


if __name__ == "__main__":
    main()
