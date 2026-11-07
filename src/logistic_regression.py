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
        self.loss_history = []

    # clip supaya np.exp tidak overflow
    def sigmoid(self, score):
        score = np.clip(score, -500, 500)
        return 1 / (1 + np.exp(-score))

    # rata-rata log-loss, makin kecil makin bagus
    def log_loss(self, X_bias, y):
        prob = self.sigmoid(X_bias @ self.weights)
        prob = np.clip(prob, 1e-12, 1 - 1e-12)
        return -np.mean(y * np.log(prob) + (1 - y) * np.log(1 - prob))

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
        self.loss_history = []

        random_gen = np.random.default_rng(self.random_state)
        for epoch in range(self.n_epochs):
            for row in random_gen.permutation(n_samples):
                score = X_bias[row] @ self.weights
                prob = self.sigmoid(score)
                error = y[row] - prob
                self.weights += self.learning_rate * error * X_bias[row]
            self.loss_history.append(self.log_loss(X_bias, y))
        return self

    # peluang kelas 1
    def predict_proba(self, X):
        X_bias = self.add_bias_column(np.asarray(X, dtype=float))
        return self.sigmoid(X_bias @ self.weights)

    # p >= threshold jadi kelas 1
    def predict(self, X, threshold=0.5):
        return (self.predict_proba(X) >= threshold).astype(int)
