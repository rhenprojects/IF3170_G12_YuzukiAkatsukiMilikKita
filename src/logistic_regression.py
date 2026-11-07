import numpy as np

# Referensi: slide IF3170 Materi 11 (Linear & Logistic Regression),
# update stochastic gradient ascent: bj = bj + eta * (yi - pi) * xij
# L2: C. Guestrin, slide CSE 546 "Logistic Regression, Gradient" (University of Washington, 2013),
#     bagian regularized conditional log likelihood dan regularized update
#     Hastie, Tibshirani, Friedman, The Elements of Statistical Learning ed. 2, subbab 3.4.1 (ridge),
#     alasan bias tidak ikut dipenalti


class LogisticRegression:
    # weights[0] adalah bias, sisanya bobot tiap fitur
    def __init__(self, learning_rate=0.01, n_epochs=100, penalty=None, reg_lambda=0.0,
                 random_state=42):
        self.learning_rate = learning_rate
        self.n_epochs = n_epochs
        self.random_state = random_state

        self.penalty = penalty
        self.reg_lambda = reg_lambda

        self.weights = None
        self.loss_history = []

    # clip supaya np.exp tidak overflow
    def sigmoid(self, score):
        score = np.clip(score, -500, 500)
        return 1 / (1 + np.exp(-score))

    # rata-rata log-loss ditambah penalti, makin kecil makin bagus
    def log_loss(self, X_bias, y):
        prob = self.sigmoid(X_bias @ self.weights)
        prob = np.clip(prob, 1e-12, 1 - 1e-12)

        loss = -np.mean(y * np.log(prob) + (1 - y) * np.log(1 - prob))

        if self.penalty == "l2":
            loss += self.reg_lambda / 2 * np.sum(self.weights[1:] ** 2)

        return loss

    # turunan penalti, bias (weights[0]) tidak ikut dihukum
    def penalty_gradient(self):
        grad = np.zeros_like(self.weights)

        if self.penalty == "l2":
            grad[1:] = self.reg_lambda * self.weights[1:]

        return grad

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

                gradient = error * X_bias[row] - self.penalty_gradient()
                self.weights += self.learning_rate * gradient

            self.loss_history.append(self.log_loss(X_bias, y))

        return self

    # peluang kelas 1
    def predict_proba(self, X):
        X_bias = self.add_bias_column(np.asarray(X, dtype=float))
        return self.sigmoid(X_bias @ self.weights)

    # p >= threshold jadi kelas 1
    def predict(self, X, threshold=0.5):
        return (self.predict_proba(X) >= threshold).astype(int)
