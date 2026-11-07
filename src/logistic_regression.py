import numpy as np

# Referensi: slide IF3170 Materi 11 (Linear & Logistic Regression),
# update stochastic gradient ascent: bj = bj + eta * (yi - pi) * xij


class LogisticRegression:
    # weights[0] adalah bias, sisanya bobot tiap fitur
    def __init__(self, learning_rate=0.01, n_epochs=100, random_state=42):
        self.learning_rate = learning_rate
        self.n_epochs = n_epochs
        self.random_state = random_state
        self.weights = None

    # clip supaya np.exp tidak overflow
    def sigmoid(self, score):
        score = np.clip(score, -500, 500)
        return 1 / (1 + np.exp(-score))

    # tambah kolom 1 supaya bias dihitung sebagai x0
    def add_bias_column(self, X):
        ones = np.ones((X.shape[0], 1))
        return np.hstack([ones, X])

    # urutan data diacak tiap epoch
    def fit(self, X, y):
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float)
        n_samples, n_features = X.shape

        X_bias = self.add_bias_column(X)
        self.weights = np.zeros(n_features + 1)

        random_gen = np.random.default_rng(self.random_state)
        for epoch in range(self.n_epochs):
            for row in random_gen.permutation(n_samples):
                score = X_bias[row] @ self.weights
                prob = self.sigmoid(score)
                error = y[row] - prob
                self.weights += self.learning_rate * error * X_bias[row]
        return self

    # peluang kelas 1
    def predict_proba(self, X):
        X_bias = self.add_bias_column(np.asarray(X, dtype=float))
        return self.sigmoid(X_bias @ self.weights)

    # p >= 0.5 jadi kelas 1
    def predict(self, X):
        return (self.predict_proba(X) >= 0.5).astype(int)
