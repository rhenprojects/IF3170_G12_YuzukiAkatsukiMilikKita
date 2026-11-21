import numpy as np

# Referensi
# Logistic regression dan stochastic gradient ascent:
#   - Slide IF3170 Materi 11, Linear Regression & Logistic Regression, ITB
# Bentuk penalti L1, L2, dan Elastic Net (reg_lambda, l1_ratio, bias tidak dipenalti):
#   - Friedman, Hastie, Tibshirani (2010), Regularization Paths for Generalized Linear Models
#     via Coordinate Descent, Journal of Statistical Software 33(1), bagian 2 dan 3
# L1 pada SGD (metode clipping: bobot dipotong di nol setelah update, bukan cumulative penalty):
#   - Tsuruoka, Tsujii, Ananiadou (2009), Stochastic Gradient Descent Training for
#     L1-regularized Log-linear Models with Cumulative Penalty, ACL-IJCNLP, bagian 3.1
# Learning rate mengecil tiap epoch (exponential decay, eta = eta0 * decay^epoch):
#   - Tsuruoka, Tsujii, Ananiadou (2009), bagian 3.3


class LogisticRegression:
    # Menaruh bias di weights[0], sisanya bobot tiap fitur.
    def __init__(self, learning_rate=0.01, n_epochs=100, penalty=None, reg_lambda=0.0, l1_ratio=0.5, lr_decay=1.0, history_every=0, random_state=42):
        if penalty not in (None, "l1", "l2", "elasticnet"):
            raise ValueError("penalty harus None, 'l1', 'l2', atau 'elasticnet'")

        self.learning_rate = learning_rate
        self.n_epochs = n_epochs
        self.lr_decay = lr_decay
        self.history_every = history_every
        self.random_state = random_state

        self.penalty = penalty
        self.reg_lambda = reg_lambda
        self.l1_ratio = l1_ratio

        self.weights = None
        self.loss_history = []
        self.weights_history = []

    # Membatasi skor biar np.exp gak overflow.
    def sigmoid(self, score):
        score = np.clip(score, -500, 500)
        return 1 / (1 + np.exp(-score))

    def add_bias_column(self, X):
        ones = np.ones((X.shape[0], 1))
        return np.hstack([ones, X])

    # Memecah reg_lambda jadi porsi L1 dan L2, elasticnet dibagi pakai l1_ratio.
    def penalty_strength(self):
        if self.penalty == "l1":
            return self.reg_lambda, 0.0
        if self.penalty == "l2":
            return 0.0, self.reg_lambda
        if self.penalty == "elasticnet":
            return self.reg_lambda * self.l1_ratio, self.reg_lambda * (1 - self.l1_ratio)
        return 0.0, 0.0

    def log_loss(self, X_bias, y):
        prob = self.sigmoid(X_bias @ self.weights)
        prob = np.clip(prob, 1e-12, 1 - 1e-12)

        loss = -np.mean(y * np.log(prob) + (1 - y) * np.log(1 - prob))

        l1, l2 = self.penalty_strength()
        loss += l1 * np.sum(np.abs(self.weights[1:]))
        loss += l2 / 2 * np.sum(self.weights[1:] ** 2)

        return loss

    # Menghitung turunan L2 tanpa bias, soalnya bias gak ikut dipenalti.
    def l2_gradient(self):
        l2 = self.penalty_strength()[1]

        grad = np.zeros_like(self.weights)
        grad[1:] = l2 * self.weights[1:]

        return grad

    # Mengecilkan bobot buat L1, kalau udah lewat nol langsung dipatok di nol.
    def shrink_l1(self, step):
        w = self.weights[1:]
        self.weights[1:] = np.sign(w) * np.maximum(np.abs(w) - step, 0)

    # Melatih pakai SGA kayak di slide, learning rate dikali lr_decay tiap epoch.
    def fit(self, X, y):
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float)
        n_samples, n_features = X.shape

        X_bias = self.add_bias_column(X)
        self.weights = np.zeros(n_features + 1)
        self.loss_history = []
        self.weights_history = [self.weights.copy()]
        n_updates = 0

        l1 = self.penalty_strength()[0]
        random_gen = np.random.default_rng(self.random_state)

        for epoch in range(self.n_epochs):
            learning_rate = self.learning_rate * self.lr_decay ** epoch

            for row in random_gen.permutation(n_samples):
                score = X_bias[row] @ self.weights
                prob = self.sigmoid(score)
                error = y[row] - prob

                gradient = error * X_bias[row] - self.l2_gradient()
                self.weights += learning_rate * gradient

                if l1 > 0:
                    self.shrink_l1(learning_rate * l1)

                # Mencatat posisi bobot tiap history_every update, kepake buat video lintasan.
                n_updates += 1
                if self.history_every > 0 and n_updates % self.history_every == 0:
                    self.weights_history.append(self.weights.copy())

            self.loss_history.append(self.log_loss(X_bias, y))

        return self

    def predict_proba(self, X):
        X_bias = self.add_bias_column(np.asarray(X, dtype=float))
        return self.sigmoid(X_bias @ self.weights)

    def predict(self, X, threshold=0.5):
        return (self.predict_proba(X) >= threshold).astype(int)
