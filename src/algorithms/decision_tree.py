from __future__ import annotations

from dataclasses import dataclass, field
from numbers import Integral, Real
from typing import Any

import numpy as np
import pandas as pd

# Referensi tambahan di luar materi kuliah
# C4.5: gain ratio dan pemisahan atribut numerik dengan threshold:
#   - Quinlan (1993), C4.5: Programs for Machine Learning, Morgan Kaufmann
#     https://www.sciencedirect.com/book/9780080500584/c4-5
# Gini impurity dan penurunan impurity berbobot:
#   - scikit-learn, Decision Trees, bagian Mathematical formulation
#     https://scikit-learn.org/stable/modules/tree.html#mathematical-formulation
# Batas pertumbuhan pohon (max_depth, min_sample_split, min_sample_leaf):
#   - scikit-learn, DecisionTreeClassifier, bagian Parameters
#     https://scikit-learn.org/stable/modules/generated/sklearn.tree.DecisionTreeClassifier.html
# Histogram-based split finding (bin, histogram kelas, dan jumlah kumulatif):
#   - LightGBM, Features, bagian Optimization in Speed and Memory Usage
#     https://lightgbm.readthedocs.io/en/stable/Features.html
# Di sini teknik histogram diadaptasi untuk satu pohon dengan Gini/entropy/gain ratio.


@dataclass
class Node:
    prediction: Any
    attr_idx: int | None = None
    children: dict[Any, Node] = field(default_factory=dict)
    threshold: float | None = None
    missing_branch: Any = None
    

class DecisionTree:
    """criterion = entropy | gain_ratio | gini"""
    def __init__(self, criterion="gain_ratio", max_depth=None, min_sample_split=5, min_sample_leaf=3, min_gain=0.0, split_method="exact", max_bins=32):
        self.criterion = criterion
        self.max_depth = max_depth
        self.min_sample_split = min_sample_split
        self.min_sample_leaf = min_sample_leaf
        self.min_gain = min_gain
        self.split_method = split_method
        self.max_bins = max_bins
        self._validate_parameters()
        self.root = None
        self.column_names = None
        self.numeric_columns = None

    def fit(self, X: pd.DataFrame, y: pd.Series):
        self._validate_parameters()
        X_np = X.to_numpy()
        y_np = y.to_numpy()
        if (X_np.ndim != 2 or y_np.ndim != 1 or len(X_np) != len(y_np) or len(y) == 0):
            raise ValueError("Data yang diberikan tidak valid")
        if pd.isna(y_np).any():
            raise ValueError("Label target tidak boleh kosong")

        self.numeric_columns = [
            pd.api.types.is_numeric_dtype(dtype)
            and not pd.api.types.is_bool_dtype(dtype)
            for dtype in X.dtypes
        ]
        
        self.root = self._build_tree(X_np, y_np, depth=0)
        self.column_names = X.columns
        
        return self

    def predict(self, X: pd.DataFrame):
        if (self.root is None):
            raise ValueError("Panggil fit() agar model bisa memprediksi")
        if (not X.columns.equals(self.column_names)):
            raise ValueError("Kolom data training dan uji berbeda")

        X_np = X.to_numpy()
        
        return np.asarray([self._predict_one(row) for row in X_np])
    
    def _calculate_entropy(self, y: np.ndarray) -> float:
        entropy = 0
        total_row = len(y)
        _, counts = np.unique(y, return_counts=True)
        for count in counts:
            p = count/total_row
            entropy += -(p * np.log2(p))
        
        return entropy
    
    def _gain(self, y_parent: np.ndarray, y_children: list[np.ndarray]) -> float:
        entropy_s = self._calculate_entropy(y_parent)
        gain = entropy_s
        
        for unique_values_y in y_children:
            gain -= (len(unique_values_y) / len(y_parent)) * self._calculate_entropy(unique_values_y)
            
        return gain
    
    def _split_information(self, y_children: list[np.ndarray]) -> float:
        value = 0
        total = 0
        for child in y_children:
            total += len(child)
        
        if (total == 0):
            return 0
            
        for child in y_children:
            p = len(child) / total
            if (p == 0):
                continue
            
            value -= p * np.log2(p)
            
        return value
    
    def _gain_ratio(
        self, y_parent: np.ndarray,
        y_children: list[np.ndarray],
        y_missing: np.ndarray | None = None,
    ) -> float | None:
        gain = self._gain(y_parent, y_children)
        split_groups = y_children
        if y_missing is not None and len(y_missing) > 0:
            gain *= len(y_parent) / (len(y_parent) + len(y_missing))
            split_groups = [*y_children, y_missing]
        # Filter information gain sebelum membandingkan gain ratio
        if gain < self.min_gain and not np.isclose(gain, self.min_gain, rtol=0, atol=1e-12):
            return None
        gain = max(0.0, gain)
        split_information = self._split_information(split_groups)
        if (split_information == 0):
            return None
        
        return gain / split_information
    
    def _calculate_gini(self, y: np.ndarray) -> float:
        gini = 1
        total_row = len(y)
        _, counts = np.unique(y, return_counts=True)
        for count in counts:
            p = count/total_row
            gini -= (pow(p, 2))
        
        return gini
    
    def _calculate_gini_diff(self, y_parent: np.ndarray, y_children: list[np.ndarray]) -> float:
        gini_parent = self._calculate_gini(y_parent)
        diff = gini_parent
        
        for y_child in y_children:
            diff -= (len(y_child) / len(y_parent)) * self._calculate_gini(y_child)
            
        return diff
    
    def _calculate_score(self, y_known, children, y_missing) -> float | None:
        if (self.criterion == "gain_ratio"):
            score = self._gain_ratio(y_known, children, y_missing)
            return score
        
        elif(self.criterion == "entropy"):
            score = self._gain(y_known, children)
        else:
            score = self._calculate_gini_diff(y_known, children)
        
        score *= len(y_known) / (len(y_known) + len(y_missing))
        if (
            score < self.min_gain
            and not np.isclose(score, self.min_gain, rtol=0, atol=1e-12)
        ):
            return None
            
        return score
        
    def _find_best_attribute(self, X: np.ndarray, y: np.ndarray):
        best_attr = None
        best_threshold = None
        best_score = -np.inf
        for attr_idx in range(X.shape[1]):
            column = X[:, attr_idx]
            known = ~pd.isna(column)
            values = column[known]
            y_known, y_missing = y[known], y[~known]
            if len(values) < 2:
                continue

            if self.numeric_columns[attr_idx]:
                values = np.asarray(values, dtype=float)
                if self.split_method == "histogram":
                    threshold, score = self._find_best_histogram_split(values, y_known, y_missing)
                    if score > best_score:
                        best_score = score
                        best_attr = attr_idx
                        best_threshold = threshold
                else:
                    unique_values = np.unique(values)
                    # Coba batas di antara setiap pasangan nilai berurutan
                    for low, high in zip(unique_values[:-1], unique_values[1:]):
                        threshold = low / 2 + high / 2
                        if threshold >= high:  # Antisipasi pembulatan float
                            threshold = low
                            
                        mask = values <= threshold
                        if not mask.any() or mask.all():
                            continue
                        
                        children = [y_known[mask], y_known[~mask]]
                        # Tolak kandidat jika salah satu anak akan terlalu kecil
                        if not self._validate_min_sample_leaf(children, y_missing):
                            continue

                        score = self._calculate_score(y_known, children, y_missing)
                            
                        if score is not None and score > best_score:
                            best_score = score
                            best_attr = attr_idx
                            best_threshold = float(threshold)
                        
            else:
                unique_values = pd.unique(values)
                if len(unique_values) < 2:
                    continue
                children = []
                for value in unique_values:
                    mask = values == value
                    children.append(y_known[mask])
                
                # Tolak kandidat jika salah satu anak akan terlalu kecil
                if not self._validate_min_sample_leaf(children, y_missing):
                    continue

                score = self._calculate_score(y_known, children, y_missing)
                                        
                if score is not None and score > best_score:
                    best_score = score
                    best_attr = attr_idx
                    best_threshold = None
        
        return best_attr, best_threshold
        
    def _build_tree(self, X: np.ndarray, y: np.ndarray, depth: int) -> Node:
        labels, counts = np.unique(y, return_counts=True)
        node = Node(prediction=labels[np.argmax(counts)])
        
        # Jadikan leaf node jika hasil seragam
        if (
            len(labels) == 1
            or (self.max_depth is not None and depth >= self.max_depth)
            or (len(y) < self.min_sample_split)
        ):
            return node
        
        # Lakukan split pada pohon
        # Cari atribut terbaik
        node.attr_idx, node.threshold = self._find_best_attribute(X, y)
        if node.attr_idx == None:
            return node

        column = X[:, node.attr_idx]
        missing = pd.isna(column)
        known = ~missing
        values = column[known]
        if node.threshold is not None:
            left_mask = np.asarray(values, dtype=float) <= node.threshold
            branches = {"<=": left_mask, ">": ~left_mask}
        else:
            branches = {value: values == value for value in pd.unique(values)}

        # Ubah nilai kosong menjadi nilai sample terbanyak
        node.missing_branch = max(branches, key=lambda key: branches[key].sum())
        for branch, known_mask in branches.items():
            if branch == node.missing_branch:
                mask = missing.copy()
            else:
                mask = np.zeros_like(missing)
            mask[known] = known_mask
            node.children[branch] = self._build_tree(X[mask], y[mask], depth+1)

        return node
        
    def _predict_one(self, row):
        node = self.root
        while node.attr_idx is not None:
            value = row[node.attr_idx]
            if pd.isna(value):
                next_idx = node.missing_branch
            elif node.threshold is not None:
                if float(value) <= node.threshold:
                    next_idx = "<="
                else:
                    next_idx = ">"
            else:
                next_idx = value
            next = node.children.get(next_idx)
            if next is None:
                return node.prediction
            node = next
            
        return node.prediction
    
    def _validate_min_sample_leaf(self, children: list[np.ndarray], y_missing: np.ndarray) -> bool:
        child_counts = [len(child) for child in children]
        # Missing masuk ke cabang terbesar.
        largest_branch = np.argmax(child_counts)
        child_counts[largest_branch] += len(y_missing)
        
        return all(size >= self.min_sample_leaf for size in child_counts)

    def _validate_parameters(self):
        if self.split_method not in ("exact", "histogram"):
            raise ValueError('split_method harus "exact" atau "histogram"')
        
        if self.criterion not in ("gain_ratio", "entropy", "gini"):
            raise ValueError('criterion harus "gain_ratio", "entropy", "gini"')
        
        for name, value, minimum in [
            ("max_depth", self.max_depth, 0),
            ("min_sample_split", self.min_sample_split, 2),
            ("min_sample_leaf", self.min_sample_leaf, 1),
            ("max_bins", self.max_bins, 2),
        ]:
            if name == "max_depth" and value is None:
                continue
            if isinstance(value, (bool, np.bool_)) or not isinstance(value, Integral) or value < minimum:
                raise ValueError(f"{name} harus integer >= {minimum}" + (" atau None" if name == "max_depth" else ""))
            
        if (
            not isinstance(self.min_gain, Real)
            or self.min_gain < 0
        ):
            raise ValueError("min_gain harus angka >= 0")
               
    def _find_best_histogram_split(self, values, y_known, y_missing):
        unique_values = np.unique(values)
        n_bins = min(self.max_bins, len(unique_values))
        if n_bins < 2:
            return None, -np.inf

        # Bagi urutan nilai unik menjadi maksimal max_bins kelompok
        positions = np.arange(1, n_bins) * len(unique_values) // n_bins
        lows, highs = unique_values[positions - 1], unique_values[positions]
        
        with np.errstate(invalid="ignore"):
            thresholds = lows / 2 + highs / 2
        thresholds = np.where(thresholds >= highs, lows, thresholds)
        thresholds = np.unique(thresholds[~np.isnan(thresholds)])
        if len(thresholds) == 0:
            return None, -np.inf

        bin_ids = np.searchsorted(thresholds, values, side="left")
        _, class_codes = np.unique(y_known, return_inverse=True)
        histogram = np.zeros((len(thresholds) + 1, class_codes.max() + 1), dtype=int)
        np.add.at(histogram, (bin_ids, class_codes), 1)
        cumulative_counts = histogram.cumsum(axis=0)
        total_counts = cumulative_counts[-1]
        parent_impurity = self._impurity_from_counts(total_counts)
        missing_count = len(y_missing)
        best_threshold, best_score = None, -np.inf

        for threshold, left_counts in zip(thresholds, cumulative_counts[:-1]):
            right_counts = total_counts - left_counts
            sizes = np.array([left_counts.sum(), right_counts.sum()])
            if min(sizes) == 0:
                continue
            
            # Missing mengikuti cabang known terbesar
            sizes[np.argmax(sizes)] += missing_count
            if min(sizes) < self.min_sample_leaf:
                continue
            
            score = self._score_histogram_split(parent_impurity, left_counts, right_counts, missing_count)
            if score is not None and score > best_score:
                best_threshold, best_score = float(threshold), score

        return best_threshold, best_score

    def _impurity_from_counts(self, counts):
        probabilities = counts[counts > 0] / counts.sum()
        if self.criterion == "gini":
            return 1 - np.sum(probabilities ** 2)
        return -np.sum(probabilities * np.log2(probabilities))

    def _score_histogram_split(self, parent_impurity, left_counts, right_counts, missing_count):
        sizes = np.array([left_counts.sum(), right_counts.sum()])
        known_size = sizes.sum()
        gain = parent_impurity
        for size, counts in zip(sizes, [left_counts, right_counts]):
            gain -= size / known_size * self._impurity_from_counts(counts)
        gain *= known_size / (known_size + missing_count)
        if gain < self.min_gain and not np.isclose(gain, self.min_gain, rtol=0, atol=1e-12):
            return None
        if self.criterion != "gain_ratio":
            return gain

        # Split information memasukkan missing sebagai kelompok tersendiri
        if missing_count:
            sizes = np.append(sizes, missing_count)
        proportions = sizes / sizes.sum()
        split_information = -np.sum(proportions * np.log2(proportions))
        if split_information == 0:
            return None
        return max(0.0, gain) / split_information