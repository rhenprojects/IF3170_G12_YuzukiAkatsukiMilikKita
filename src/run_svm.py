from pathlib import Path

import pandas as pd

from algorithms.svm import SVM
from model_io import load_model, save_model
from preprocessing import prepare_data
from tuning import macro_f1_score

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
OUTPUT_DIR = ROOT / "output"
MODEL_PATH = OUTPUT_DIR / "svm.pkl"
SUBMISSION_PATH = OUTPUT_DIR / "submission_svm.csv"

# Parameter terbaik yang didapat dari hasil tuning (5-fold CV, macro F1 = 0,8278)
# gamma_factor dikalikan ke gamma "scale" = 1 / (jumlah fitur * variansi X)
MODEL_PARAMS = {
    "kernel": "rbf",
    "C": 3.0,
    "gamma_factor": 4.5,
    "class_weight": None,
}


def main():
    OUTPUT_DIR.mkdir(exist_ok=True)

    # 1. Data Preparation
    X_train, X_valid, X_test, y_train, y_valid, test_ids, _ = prepare_data(
        data_dir=DATA_DIR, track="scaled"
    )

    # 2. Train
    gamma = MODEL_PARAMS["gamma_factor"] / (X_train.shape[1] * X_train.var())
    print("Training SVM:", MODEL_PARAMS, flush=True)
    model = SVM(
        kernel=MODEL_PARAMS["kernel"],
        C=MODEL_PARAMS["C"],
        gamma=gamma,
        class_weight=MODEL_PARAMS["class_weight"],
    ).fit(X_train, y_train)
    print(f"Iterasi SMO: {model.n_iter_}, jumlah support vector: {len(model.support_)}")

    # 3. Evaluation
    valid_pred = model.predict(X_valid)
    print("Macro F1 (validasi):", round(macro_f1_score(y_valid, valid_pred), 4))

    # 4. Save & Load
    save_model(model, MODEL_PATH)
    loaded = load_model(MODEL_PATH)
    assert (loaded.predict(X_valid) == valid_pred).all(), "Model hasil load berbeda!"
    print("Model disimpan di", MODEL_PATH)

    # 5. Submission
    submission = pd.DataFrame({"id": test_ids, "is_canceled": loaded.predict(X_test)})
    submission.to_csv(SUBMISSION_PATH, index=False)
    print("Submission disimpan di", SUBMISSION_PATH)


if __name__ == "__main__":
    main()
