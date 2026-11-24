from pathlib import Path
import sys

# Tambahkan src agar runner bisa dijalankan langsung maupun sebagai modul.
SRC_DIR = Path(__file__).resolve().parents[1]
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

import pandas as pd
from sklearn.metrics import f1_score

from model_io import load_model, save_model
from preprocessing import prepare_data
from algorithms.naive_bayes import NaiveBayes

ROOT = SRC_DIR.parent
DATA_DIR = ROOT / "data"
OUTPUT_DIR = ROOT / "output"
MODEL_PATH = OUTPUT_DIR / "naive_bayes.pkl"
SUBMISSION_PATH = OUTPUT_DIR / "submission_naive_bayes.csv"


def main():
    OUTPUT_DIR.mkdir(exist_ok=True)

    # 1. Data Preparation
    X_train, X_valid, X_test, y_train, y_valid, test_ids, _ = prepare_data(
        data_dir=DATA_DIR, track="naive_bayes"
    )

    # 2. Train
    model = NaiveBayes(alpha=1.0).fit(X_train, y_train)

    # 3. Evaluation
    valid_pred = model.predict(X_valid)
    print("Macro F1 (validasi):", round(f1_score(y_valid, valid_pred, average="macro"), 4))

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