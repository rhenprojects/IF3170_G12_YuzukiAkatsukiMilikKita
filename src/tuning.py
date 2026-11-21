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
    pass

def _validate_data(X, y):
    pass