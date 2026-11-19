import numpy as np


class NaiveBayes:
    def __init__(self, alpha=1.0):
        self.alpha = alpha

    def fit(self, X, y):
        X = np.asarray(X, dtype=int)
        y = np.asarray(y)

        # 1. Kelas apa saja, dan berapa kategori tiap fitur
        self.classes = np.unique(y)
        self.n_categories = X.max(axis=0) + 1

        # 2. Prior: probabilitas tiap kelas
        self.prior = self.compute_prior(y)

        # 3. Likelihood: probabilitas tiap kategori muncul di tiap kelas
        self.likelihood = self.compute_likelihood(X, y)
        return self

    def predict(self, X):
        scores = self.compute_scores(X)
        best = np.argmax(scores, axis=1)
        return self.classes[best]

    def predict_proba(self, X):
        scores = self.compute_scores(X)

        proba = []
        for row_scores in scores:
            proba.append(row_scores / np.sum(row_scores))
        return np.array(proba)

    def compute_prior(self, y):
        priors = []
        for c in self.classes:
            n_class = np.sum(y == c)
            priors.append(n_class / len(y))
        return np.array(priors)

    def compute_likelihood(self, X, y):
        # Hasil: likelihood[fitur][kelas][kategori]
        table = []

        for j in range(X.shape[1]):
            n_cat = self.n_categories[j]
            per_class = []

            for c in self.classes:
                rows = X[y == c]
                n_class = len(rows)

                probs = []
                for v in range(n_cat):
                    count = np.sum(rows[:, j] == v)
                    # Laplace smoothing
                    probs.append((count + self.alpha) / (n_class + self.alpha * n_cat))

                per_class.append(probs)

            table.append(per_class)
        return table

    def compute_scores(self, X):
        # Hasil: scores[baris][kelas]
        X = np.asarray(X, dtype=int)
        scores = []

        for row in X:
            row_scores = []
            for k in range(len(self.classes)):
                score = self.prior[k]
                for j, value in enumerate(row):
                    # Kategori tak dikenal (-1) dilewati
                    if 0 <= value < self.n_categories[j]:
                        score *= self.likelihood[j][k][value]
                row_scores.append(score)
            scores.append(row_scores)
        return np.array(scores)