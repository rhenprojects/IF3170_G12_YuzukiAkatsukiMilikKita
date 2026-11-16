from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass
class Node:
    children: list
    

class DecisionTree:
    def __init__(self, criterion="entropy", max_depth=None):
        self.criterion = criterion
        self.max_depth = max_depth

    def fit(self, X, y):
        self._build_tree(X, y)
        # TODO
        pass

    def predict(self, X):
        # TODO
        pass
    
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
        best_score = -np.inf
        for attr_idx in range(X.shape[1]):
            #TODO numeric case
            unique_values = np.unique(X[:, attr_idx])
            children = []
            for value in unique_values:
                mask = X[:, attr_idx] == value
                children.append(y[mask])
            
            score = self._gain_ratio(y, children)
            if score == None:
                continue
            
            if score > best_score:
                best_score = score
                best_attr = attr_idx
        
        return best_attr
        
    def _build_tree(self, X: np.ndarray, y: np.ndarray):
        #TODO
        pass
    
    def _predict_one(self, data):
        #TODO
        pass