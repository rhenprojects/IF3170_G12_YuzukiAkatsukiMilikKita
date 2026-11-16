from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd


@dataclass
class Node:
    prediction: any
    attr_idx: int | None = None
    children: dict[any, Node] = field(default_factory=dict)
    threshold: float | None = None
    

class DecisionTree:
    def __init__(self, criterion="entropy", max_depth=None):
        self.criterion = criterion
        self.max_depth = max_depth
        self.root = None
        self.column_names = None
        self.numeric_columns = None

    def fit(self, X: pd.DataFrame, y: pd.Series):
        X_np = X.to_numpy()
        y_np = y.to_numpy()
        if (X_np.ndim != 2 or y_np.ndim != 1 or len(X_np) != len(y_np) or len(y) == 0):
            raise ValueError("Data yang diberikan tidak valid")

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
    
    def _gain(self, y_parent: np.ndarray, y_children: np.ndarray) -> float:
        entropy_s = self._calculate_entropy(y_parent)
        gain = entropy_s
        
        for unique_values_y in y_children:
            gain -= (len(unique_values_y) / len(y_parent)) * self._calculate_entropy(unique_values_y)
            
        return gain
    
    def _split_information(self, y_children: np.ndarray) -> float:
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
    
    def _gain_ratio(self, y_parent: np.ndarray, y_children: np.ndarray) -> float | None:
        gain = self._gain(y_parent, y_children)
        split_information = self._split_information(y_children)
        if (split_information == 0):
            return None
        
        return gain / split_information
        
    def _find_best_attribute(self, X: np.ndarray, y: np.ndarray):
        best_attr = None
        best_threshold = None
        best_score = -np.inf
        for attr_idx in range(X.shape[1]):
            if self.numeric_columns[attr_idx]:
                values = np.asarray(X[:, attr_idx], dtype=float)
                unique_values = np.unique(values)
                # Coba batas di antara setiap pasangan nilai berurutan
                for low, high in zip(unique_values[:-1], unique_values[1:]):
                    threshold = low / 2 + high / 2
                    if threshold >= high:  # Antisipasi pembulatan float.
                        threshold = low
                        
                    mask = values <= threshold
                    if not mask.any() or mask.all():
                        continue
                    
                    score = self._gain_ratio(y, [y[mask], y[~mask]])
                    if score is not None and score > best_score:
                        best_score = score
                        best_attr = attr_idx
                        best_threshold = float(threshold)
                        
            else:
                unique_values = np.unique(X[:, attr_idx])
                children = []
                for value in unique_values:
                    mask = X[:, attr_idx] == value
                    children.append(y[mask])
                
                score = self._gain_ratio(y, children)
                if score == None:
                    continue
                
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
        ):
            return node
                
        # Cari atribut terbaik
        node.attr_idx, node.threshold = self._find_best_attribute(X, y)
        if node.attr_idx == None:
            return node

        if node.threshold is not None:
            values = np.asarray(X[:, node.attr_idx], dtype=float)
            mask = values <= node.threshold
            node.children["<="] = self._build_tree(X[mask], y[mask], depth+1)
            node.children[">"] = self._build_tree(X[~mask], y[~mask], depth+1)
            
        else:
            for value in np.unique(X[:, node.attr_idx]):
                mask = X[:, node.attr_idx] == value
                node.children[value] = self._build_tree(X[mask], y[mask], depth+1)
                
        return node
        
    def _predict_one(self, row):
        node = self.root
        while node.attr_idx is not None:
            if node.threshold is not None:
                if float(row[node.attr_idx]) <= node.threshold:
                    next_idx = "<="
                else:
                    next_idx = ">"
            else:
                next_idx = row[node.attr_idx]
            next = node.children.get(next_idx)
            if next is None:
                return node.prediction
            node = next
            
        return node.prediction