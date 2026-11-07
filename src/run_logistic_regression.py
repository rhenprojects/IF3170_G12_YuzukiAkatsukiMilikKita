from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression as SklearnLogisticRegression

from logistic_regression import LogisticRegression
from preprocessing import prepare_data

ROOT = Path(__file__).resolve().parent.parent

# kombinasi penalti yang dicoba
CONFIGS = [
    {"penalty": None},
    {"penalty": "l2", "reg_lambda": 0.0001},
    {"penalty": "l2", "reg_lambda": 0.001},
    {"penalty": "l2", "reg_lambda": 0.01},
    {"penalty": "l1", "reg_lambda": 0.0001},
    {"penalty": "l1", "reg_lambda": 0.001},
    {"penalty": "l1", "reg_lambda": 0.01},
    {"penalty": "elasticnet", "reg_lambda": 0.0001, "l1_ratio": 0.5},
    {"penalty": "elasticnet", "reg_lambda": 0.001, "l1_ratio": 0.5},
    {"penalty": "elasticnet", "reg_lambda": 0.01, "l1_ratio": 0.5},
]


# metrik Kaggle: rata-rata F1 kelas 0 dan kelas 1
def macro_f1(y_true, y_pred):
    f1 = []
    for label in [0, 1]:
        tp = np.sum((y_pred == label) & (y_true == label))
        fp = np.sum((y_pred == label) & (y_true != label))
        fn = np.sum((y_pred != label) & (y_true == label))
        f1.append(2 * tp / (2 * tp + fp + fn))
    return np.mean(f1)


# coba threshold 0.20 sampai 0.60, ambil yang macro F1-nya paling tinggi
def find_threshold(y_true, prob):
    best_score, best_threshold = 0, 0.5
    for threshold in np.arange(0.20, 0.65, 0.05):
        score = macro_f1(y_true, (prob >= threshold).astype(int))
        if score > best_score:
            best_score, best_threshold = score, round(threshold, 2)
    return best_score, best_threshold


# sklearn pakai C = 1 / (reg_lambda * n) karena loss-nya dijumlah, bukan dirata-rata
def make_sklearn_model(config, n_samples):
    if config["penalty"] is None:
        return SklearnLogisticRegression(C=np.inf, max_iter=5000)

    l1_ratio = {"l2": 0.0, "l1": 1.0, "elasticnet": config.get("l1_ratio")}[config["penalty"]]
    C = 1 / (config["reg_lambda"] * n_samples)
    return SklearnLogisticRegression(C=C, l1_ratio=l1_ratio, solver="saga", max_iter=5000)


def main():
    X_train, X_valid, X_test, y_train, y_valid, test_ids, pipeline = prepare_data(ROOT / "data")
    y_train = y_train.to_numpy()
    y_valid = y_valid.to_numpy()

    best = None
    for config in CONFIGS:
        model = LogisticRegression(learning_rate=0.01, n_epochs=10, **config)
        model.fit(X_train, y_train)

        prob = model.predict_proba(X_valid)
        score, threshold = find_threshold(y_valid, prob)
        n_zero = np.sum(model.weights[1:] == 0)
        print(f"{str(config):<62} macro F1 = {score:.4f} (threshold {threshold}), bobot nol = {n_zero}")

        if best is None or score > best["score"]:
            best = {"score": score, "threshold": threshold, "config": config, "model": model}

    print(f"\nModel terbaik: {best['config']}, threshold {best['threshold']}, macro F1 = {best['score']:.4f}")

    sklearn_model = make_sklearn_model(best["config"], len(y_train))
    sklearn_model.fit(X_train, y_train)
    sklearn_prob = sklearn_model.predict_proba(X_valid)[:, 1]
    sklearn_score = macro_f1(y_valid, (sklearn_prob >= best["threshold"]).astype(int))
    print(f"sklearn dengan pengaturan yang sama: macro F1 = {sklearn_score:.4f}")

    output_dir = ROOT / "output"
    output_dir.mkdir(exist_ok=True)
    best["model"].save(output_dir / "logistic_regression.pkl")

    prediction = best["model"].predict(X_test, threshold=best["threshold"])
    submission = pd.DataFrame({"id": test_ids, "is_canceled": prediction})
    submission.to_csv(output_dir / "submission_logistic_regression.csv", index=False)
    print("\nModel dan submission disimpan di folder output/")


if __name__ == "__main__":
    main()
