from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression as SklearnLogisticRegression

if __package__:
    from .algorithms.logistic_regression import LogisticRegression
    from .model_io import load_model, save_model
    from .preprocessing import BINARY_FEATURES, CATEGORICAL_FEATURES, NUMERIC_FEATURES, build_logistic_regression_pipeline, prepare_data
    from .tuning import macro_f1_score, tune
else:
    from algorithms.logistic_regression import LogisticRegression
    from model_io import load_model, save_model
    from preprocessing import BINARY_FEATURES, CATEGORICAL_FEATURES, NUMERIC_FEATURES, build_logistic_regression_pipeline, prepare_data
    from tuning import macro_f1_score, tune

# Referensi
# Pemilihan hyperparameter dengan k-fold cross-validation:
#   - Hastie, Tibshirani, Friedman, The Elements of Statistical Learning ed. 2, subbab 7.10.1

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
OUTPUT_DIR = ROOT / "output"
MODEL_PATH = OUTPUT_DIR / "logistic_regression.pkl"
SUBMISSION_PATH = OUTPUT_DIR / "submission_logistic_regression.csv"
SEED = 42

# Membandingkan L1, L2, dan Elastic Net di SGA, lalu AdaGrad yang cukup 5 epoch soalnya loss-nya udah nyamain SGA 10 epoch.
GRIDS = [
    {"optimizer": ["sga"], "n_epochs": [10], "learning_rate": [0.01], "lr_decay": [0.7], "penalty": ["l2"], "reg_lambda": [0.0003]},
    {"optimizer": ["sga"], "n_epochs": [10], "learning_rate": [0.01], "lr_decay": [0.7], "penalty": ["l1"], "reg_lambda": [0.0001]},
    {"optimizer": ["sga"], "n_epochs": [10], "learning_rate": [0.01], "lr_decay": [0.7], "penalty": ["elasticnet"], "reg_lambda": [0.0003], "l1_ratio": [0.2]},
    {"optimizer": ["adagrad"], "n_epochs": [5], "learning_rate": [0.05], "penalty": ["l2"], "reg_lambda": [0.0003]},
    {"optimizer": ["adagrad"], "n_epochs": [5], "learning_rate": [0.05], "penalty": ["l1"], "reg_lambda": [0.0001]},
]


# Mencari threshold yang bikin macro F1 paling tinggi, soalnya kelas batal cuma sekitar sepertiga data.
def find_threshold(y_true, prob):
    best_score, best_threshold = 0, 0.5
    for threshold in np.arange(0.20, 0.65, 0.05):
        score = macro_f1_score(y_true, (prob >= threshold).astype(int))
        if score > best_score:
            best_score, best_threshold = score, round(threshold, 2)
    return best_score, best_threshold


# Menyamakan C sklearn sama reg_lambda, C = 1 / (reg_lambda * n) soalnya sklearn ngejumlahin loss.
def make_sklearn_model(params, n_samples):
    l1_ratio = {"l2": 0.0, "l1": 1.0, "elasticnet": params.get("l1_ratio")}[params["penalty"]]
    C = 1 / (params["reg_lambda"] * n_samples)
    # Memakai liblinear buat L1 soalnya saga lambat banget di ribuan fitur, saga cuma buat elasticnet.
    solver = {0.0: "lbfgs", 1.0: "liblinear"}.get(l1_ratio, "saga")
    return SklearnLogisticRegression(C=C, l1_ratio=l1_ratio, solver=solver, max_iter=1000)


def main():
    OUTPUT_DIR.mkdir(exist_ok=True)

    # 1. Data Preparation
    X_train, X_valid, _, y_train, y_valid, test_ids, _ = prepare_data(
        data_dir=DATA_DIR, seed=SEED, track="logistic_regression"
    )
    print("Jumlah fitur:", X_train.shape[1])

    # 2. Tuning
    print(f"Mencoba {len(GRIDS)} konfigurasi dengan 5-fold CV di data latih, bisa makan beberapa menit...", flush=True)
    base_model = LogisticRegression(random_state=SEED)
    results = pd.concat([tune(base_model, grid, X_train, y_train, random_state=SEED) for grid in GRIDS])
    results = results.sort_values("kfold_mean", ascending=False, ignore_index=True)
    for _, row in results.iterrows():
        print(f"{str(row['params']):<136} macro F1 CV = {row['kfold_mean']:.4f} ± {row['kfold_std']:.4f}")
    best_params = results.loc[0, "params"]
    print("\nModel terbaik:", best_params)

    # 3. Train dan cari threshold di data validasi
    model = LogisticRegression(random_state=SEED, **best_params).fit(X_train, y_train)
    valid_score, threshold = find_threshold(y_valid, model.predict_proba(X_valid))
    print(f"Macro F1 (validasi): {valid_score:.4f} di threshold {threshold}")

    sklearn_model = make_sklearn_model(best_params, len(y_train)).fit(X_train, y_train)
    sklearn_pred = (sklearn_model.predict_proba(X_valid)[:, 1] >= threshold).astype(int)
    print(f"Macro F1 sklearn (validasi): {macro_f1_score(y_valid, sklearn_pred):.4f}")

    # 4. Melatih ulang model final pakai seluruh train.csv biar datanya gak kebuang 20%.
    print("\nMelatih model final pakai seluruh train.csv...", flush=True)
    train_df = pd.read_csv(DATA_DIR / "train.csv", dtype={"id": str})
    test_df = pd.read_csv(DATA_DIR / "test.csv", dtype={"id": str})
    columns = NUMERIC_FEATURES + CATEGORICAL_FEATURES + BINARY_FEATURES
    pipeline = build_logistic_regression_pipeline()
    X_full = pipeline.fit_transform(train_df[columns])
    X_test = pipeline.transform(test_df[columns])
    model = LogisticRegression(random_state=SEED, **best_params).fit(X_full, train_df["is_canceled"])
    print("Jumlah fitur model final:", X_full.shape[1])
    print("Bobot yang bernilai nol:", np.sum(model.weights[1:] == 0))

    # 5. Save & Load
    save_model(model, MODEL_PATH)
    loaded = load_model(MODEL_PATH)
    prediction = loaded.predict(X_test, threshold=threshold)
    assert (prediction == model.predict(X_test, threshold=threshold)).all(), "Model hasil load berbeda!"
    print("Model disimpan di", MODEL_PATH)

    # 6. Submission
    submission = pd.DataFrame({"id": test_ids, "is_canceled": prediction})
    submission.to_csv(SUBMISSION_PATH, index=False)
    print("Submission disimpan di", SUBMISSION_PATH)


if __name__ == "__main__":
    main()
