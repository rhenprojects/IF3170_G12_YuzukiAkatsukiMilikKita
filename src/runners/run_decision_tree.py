from pathlib import Path
import sys

# Tambahkan src agar runner bisa dijalankan langsung maupun sebagai modul.
SRC_DIR = Path(__file__).resolve().parents[1]
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

import pandas as pd

from algorithms.decision_tree import DecisionTree
from model_io import load_model, save_model
from preprocessing import prepare_data
from tuning import macro_f1_score

ROOT = SRC_DIR.parent
DATA_DIR = ROOT / "data"
OUTPUT_DIR = ROOT / "output"
MODEL_PATH = OUTPUT_DIR / "decision_tree.pkl"
SUBMISSION_PATH = OUTPUT_DIR / "submission_decision_tree.csv"

# Parameter terbaik yang didapat dari hasil tuning
MODEL_PARAMS = {
    "criterion": "gini",
    "max_depth": 12,
    "min_sample_leaf": 5,
    "min_sample_split": 15,
    "min_gain": 0.0,
}


def main():
    OUTPUT_DIR.mkdir(exist_ok=True)

    # 1. Data Preparation
    X_train, X_valid, X_test, y_train, y_valid, test_ids, _ = prepare_data(
        data_dir=DATA_DIR, track="scaled"
    )
    
    X_train = pd.DataFrame(X_train)
    X_valid = pd.DataFrame(X_valid)
    X_test = pd.DataFrame(X_test)

    # 2. Train
    print("Training Decision Tree:", MODEL_PARAMS, flush=True)
    model = DecisionTree(**MODEL_PARAMS).fit(X_train, y_train)

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
