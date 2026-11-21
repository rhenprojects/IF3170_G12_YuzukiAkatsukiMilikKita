from __future__ import annotations

from collections.abc import Mapping
from copy import deepcopy
from inspect import Parameter, signature
from numbers import Integral

import numpy as np
import pandas as pd


def tune(
    model, grid, X, y, method=None, *, n_splits=5, shuffle=True,
    random_state=42, stratified=True, scoring="f1_macro", predict_mode="batch",
):
    """Mengembalikan tabel kombinasi parameter dan scorenya, diurutkan dari skor tertinggi."""
    _validate_data(X, y)

    # Pakai seed yang sama agar semua kombinasi mendapat pembagian fold sama
    if random_state is None:
        random_gen = np.random.default_rng()
        random_state = int(random_gen.integers(0, 2**32))

    rows = []
    result_columns = {"params", "kfold_mean", "kfold_std"}

    for params in parameter_grid(grid):
        if result_columns.intersection(params):
            raise ValueError("Nama parameter bertabrakan dengan nama kolom hasil")

        candidate_model = clone_model(model, params)
        scores = kfold_scores(
            candidate_model, X, y, method,
            n_splits=n_splits,
            shuffle=shuffle,
            random_state=random_state,
            stratified=stratified,
            scoring=scoring,
            predict_mode=predict_mode,
        )

        row = {"params": params}
        row.update(params)
        row["kfold_mean"] = float(scores.mean())
        row["kfold_std"] = float(scores.std())
        rows.append(row)

    results = pd.DataFrame(rows)
    results = results.sort_values(
        "kfold_mean",
        ascending=False,
        ignore_index=True,
        kind="stable",
    )

    return results

def parameter_grid(grid):
    if not isinstance(grid, Mapping):
        raise TypeError("grid harus berupa dictionary parameter -> value")

    combinations = [{}]

    for name, candidates in grid.items():
        if not isinstance(name, str):
            raise TypeError("Nama parameter harus string")
        
        try:
            candidates = list(candidates)
        except TypeError as exc:
            raise TypeError(f'Nilai grid {name!r} harus iterable') from exc
        
        if not candidates:
            raise ValueError(f'Daftar pilihan {name!r} tidak boleh kosong')

        # Buat kombinasi dengan nilai parameter berikutnya
        next_combinations = []
        for params in combinations:
            for value in candidates:
                new_params = params.copy()
                new_params[name] = value
                next_combinations.append(new_params)

        combinations = next_combinations

    for params in combinations:
        yield params

def clone_model(model, params=None):
    model_params = _get_model_params(model)
    if params is None:
        params = {}
    else:
        params = dict(params)

    unknown_params = set(params) - set(model_params)
    if unknown_params:
        raise ValueError(f'Parameter model tidak dikenal: {sorted(unknown_params)}')

    model_params.update(params)
    model_params = deepcopy(model_params)
    model_class = type(model)

    return model_class(**model_params)

def _get_model_params(model):
    if hasattr(model, "get_params"):
        return dict(model.get_params(deep=False))

    model_class = type(model)
    constructor_params = signature(model_class.__init__).parameters
    model_params = {}

    for name, parameter in constructor_params.items():
        if name == "self":
            continue
        if parameter.kind in (Parameter.VAR_POSITIONAL, Parameter.VAR_KEYWORD):
            raise TypeError("Model dengan *args/**kwargs perlu get_params(deep=False)")
        if parameter.kind == Parameter.POSITIONAL_ONLY:
            raise TypeError("Parameter konstruktor model harus dapat diberikan dengan keyword")
        if not hasattr(model, name):
            raise TypeError(f"Model harus menyimpan parameter {name!r} sebagai atribut")

        model_params[name] = getattr(model, name)

    return model_params

def kfold_scores(
    model, X, y, method=None, *, n_splits=5, shuffle=True,
    random_state=42, stratified=True, scoring="f1_macro", predict_mode="batch",
):
    _validate_data(X, y)

    if method is not None and not callable(method):
        raise TypeError("method harus None atau fungsi balancing(X_train, y_train)")
    
    if predict_mode not in ("batch", "single"):
        raise ValueError("predict_mode harus 'batch' atau 'single'")
    
    if callable(scoring):
        score_function = scoring
    elif scoring == "f1_macro":
        score_function = macro_f1_score
    elif scoring == "accuracy":
        score_function = accuracy_score
    else:
        raise ValueError("scoring harus 'f1_macro', 'accuracy', atau callable")
    
    scores = []
    folds = kfold(
        y,
        n_splits=n_splits,
        shuffle=shuffle,
        random_state=random_state,
        stratified=stratified,
    )

    for train_indices, valid_indices in folds:
        # Pisahkan data berdasarkan posisi baris, bukan label indeks pandas.
        X_train = _take(X, train_indices)
        y_train = _take(y, train_indices)
        X_valid = _take(X, valid_indices)
        y_valid = _take(y, valid_indices)

        # Balancing hanya dilakukan pada data training.
        if method is not None:
            X_train, y_train = method(X_train, y_train)
            _validate_data(X_train, y_train)

        # Gunakan model baru agar hasil training fold sebelumnya tidak terbawa.
        fold_model = clone_model(model)
        fold_model.fit(X_train, y_train)

        if predict_mode == "single":
            predictions = []
            for row in np.asarray(X_valid):
                prediction = fold_model.predict(row)
                predictions.append(prediction)
        else:
            predictions = fold_model.predict(X_valid)

        _validate_predictions(y_valid, predictions)
        score = float(score_function(y_valid, predictions))
        if not np.isfinite(score):
            raise ValueError("Fungsi scoring harus menghasilkan skor finite")

        scores.append(score)

    return np.asarray(scores, dtype=float)


def kfold(y, n_splits=5, shuffle=True, random_state=42, stratified=True):
    labels = np.asarray(y)
    if labels.ndim != 1 or len(labels) == 0 or pd.isna(labels).any():
        raise ValueError("y harus vektor label yang tidak kosong dan tanpa NaN")
    if (
        isinstance(n_splits, bool)
        or not isinstance(n_splits, Integral)
        or not 2 <= n_splits <= len(labels)
    ):
        raise ValueError("n_splits harus integer diantara 2 dan jumlah sampel")

    random_gen = np.random.default_rng(random_state)
    folds = []
    for fold_idx in range(n_splits):
        folds.append([])

    if stratified:
        class_codes, classes = pd.factorize(labels, sort=False)
        next_fold = 0

        # Sebarkan sampel setiap kelas secara bergiliran ke seluruh fold
        for class_idx in range(len(classes)):
            class_indices = np.flatnonzero(class_codes == class_idx)
            if len(class_indices) < n_splits:
                raise ValueError("Setiap kelas perlu minimal n_splits sampel untuk stratified k-fold")
            if shuffle:
                random_gen.shuffle(class_indices)

            for sample_idx in class_indices:
                folds[next_fold].append(sample_idx)
                next_fold = (next_fold + 1) % n_splits
                
    else:
        indices = np.arange(len(labels))
        if shuffle:
            random_gen.shuffle(indices)
        folds = np.array_split(indices, n_splits)
    
    # Satu fold menjadi data validasi, sisanya menjadi data training
    all_indices = np.arange(len(labels))
    for fold in folds:
        valid_indices = np.sort(np.asarray(fold, dtype=int))
        train_mask = np.ones(len(labels), dtype=bool)
        train_mask[valid_indices] = False
        train_indices = all_indices[train_mask]

        yield train_indices, valid_indices

def accuracy_score(y_true, y_pred):
    y_true, y_pred = _validate_predictions(y_true, y_pred)
    correct_predictions = np.sum(y_true == y_pred)
    accuracy = correct_predictions / len(y_true)

    return float(accuracy)

def macro_f1_score(y_true, y_pred):
    y_true, y_pred = _validate_predictions(y_true, y_pred)
    labels = pd.unique(np.concatenate([y_true, y_pred]))
    scores = []

    for label in labels:
        actual_class = y_true == label
        predicted_class = y_pred == label

        true_positive = np.sum(actual_class & predicted_class)
        false_positive = np.sum(~actual_class & predicted_class)
        false_negative = np.sum(actual_class & ~predicted_class)

        denominator = 2 * true_positive + false_positive + false_negative
        if denominator == 0:
            score = 0.0
        else:
            score = 2 * true_positive / denominator

        scores.append(score)

    return float(np.mean(scores))

def _validate_predictions(y_true, y_pred):
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    if (
        y_true.ndim != 1
        or y_pred.ndim != 1
        or len(y_true) == 0
        or len(y_true) != len(y_pred)
    ):
        raise ValueError("Label asli dan prediksi harus vektor dengan panjang sama, tidak kosong")
    
    if pd.isna(y_true).any() or pd.isna(y_pred).any():
        raise ValueError("Label asli dan prediksi tidak boleh mengandung NaN")

    return y_true, y_pred

def _take(data, indices):
    if isinstance(data, (pd.DataFrame, pd.Series)):
        return data.iloc[indices].copy()
    
    return np.asarray(data)[indices].copy()

def _validate_data(X, y):
    if np.ndim(X) != 2:
        raise ValueError("X harus berbentuk dua dimensi (sampel, fitur)")
    labels = np.asarray(y)
    if (
        labels.ndim != 1
        or len(X) != len(labels)
        or len(labels) == 0
    ):
        raise ValueError("y harus satu dimensi, tidak kosong, dan sepanjang X")
    if pd.isna(labels).any():
        raise ValueError("Label y tidak boleh kosong/NaN")
    return labels