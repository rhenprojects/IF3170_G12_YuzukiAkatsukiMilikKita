from __future__ import annotations

import numpy as np

class DecisionTree:
    def __init__(self, criterion="entropy", max_depth=None):
        self.criterion = criterion
        self.max_depth = max_depth

    def fit(self, X, y):
        # TODO
        pass

    def predict(self, X):
        # TODO
        pass
    
    def _calculate_entropy(self, y: list[any]) -> float:
        entropy = 0
        total_row = len(y)
        unique = np.unique(y, return_counts=True)
        for count in unique[1]:
            p = count/total_row
            entropy += -(p * np.log2(p))
        
        return entropy
    
    def _gain(self, y_parent: list[any], y_child: list[list[any]]) -> float:
        entropy_s = self._calculate_entropy(y_parent)
        gain = entropy_s
        
        for unique in y_child:
            gain -= (len(unique) / len(y_parent)) * self._calculate_entropy(unique)
            
        return gain
        
    def _find_best_attribute(self, x, y):
        # TODO
        pass
        
