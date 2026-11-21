import pickle  # untuk menyimpan / memuat model ke file
import warnings  # untuk memberi peringatan kalau SMO belum konvergen
from collections import OrderedDict  # struktur data untuk cache LRU

import numpy as np

_TAU = 1e-12

class SVM:

    def __init__(
            self,
            C=1.0,
            kernel = "rbf",
            degree=3,
            gamma="scale",
            coef0=0.0,
            tol=1e-3,
            max_iter=-1,
            class_weight=None,
            cache_size=500,
            verbose=False,
    ):
        self.C = C #Penalti soft margin. Makin besar makin dikit pelanggaran margin
        self.kernel = kernel # ada 3 yaitu linear, poly, rbf
        self.degree = degree # derajat untuk kernel plynomial
        self.gamma = gamma # untuk rbf dan poly, mengatur seberapa lokal pengaruh setiap titik data
        self.coef0 = coef0 # digunakan polynomial
        self.tol = tol # batas toleransi untuk menghentikan SMO
        self.max_iter = max_iter # batas maksimal iterasi SMO
        self.class_weight = class_weight  #mengatasi kelas imbalance
        self.cache_size = cache_size # kapasitas untuk matrix kernel
        self.verbose = verbose # menampilkan perkembangan training atau tidak

    def _resolve_kernel(self):
        aliases = {"polynomial": "poly", "gaussian":"rbf"}
        kernel = aliases.get(self.kernel, self.kernel)
        if kernel not in ("linear", "poly", "rbf"):
            raise ValueError(f"kernel tidak dikenal: {self.kernel!r}")
        return kernel

    def _resolve_gamma(self, X):
        if self.gamma == "scale":
            var = X.var()
            return 1.0/ (X.shape[1] * var) if var >0 else 1.0
        if self.gamma == "auto":
            return 1.0 / X.shape[1]
        return float(self.gamma)

    def _kernel(self, A, B, sq_A=None, sq_B=None):
        if self.kernel_ == "linear":
            return A @ B.T
        if self.kernel_ == "poly":
            return (self.gamma_ * (A @ B.T) + self.coef0) ** self.degree

        if sq_A is None:
            sq_A = np.einsum("ij,ij->i", A, A)
        if sq_B is None:
            sq_B = np.einsum("ij,ij->i", B, B)
        d2 = sq_A[:, None] + sq_B[None, :] - 2.0 * (A @ B.T)
        # Karena pembulatan, d2 bisa sedikit negatif (mis. -1e-15); dipotong ke 0.
        np.maximum(d2, 0.0, out=d2)
        return np.exp(-self.gamma_ * d2)

    def _kernel_diag(self, X, sq):
        if self.kernel_ == "linear":
            return sq.copy()
        if self.kernel_ == "poly":
            return (self.gamma_ * sq + self.coef0) ** self.degree
        return np.ones(len(X))

    def _class_C(self, y_signed):
        if self.class_weight is None:
            w_pos = w_neg = 1.0
        elif self.class_weight == "balanced":
            n = len(y_signed)
            w_pos = n/ (2.0 * max(np.sum(y_signed == 1), 1))
            w_neg = n/ (2.0 * max(np.sum(y_signed == -1), 1))
        else:
            w_neg = float(self.class_weight.get(self.classes_[0], 1.0))
            w_pos = float(self.class_weight.get(self.classes_[1], 1.0))
        return np.where(y_signed == 1, self.C * w_pos,  self.C * w_neg)

    def fit(self, X, y):
        X = np.ascontiguousarray(X, dtype=np.float64)
        y = np.asarray(y).ravel()
        if X.ndim != 2:
            raise ValueError("X harus 2D (n_samples, n_features)")
        if len(X) != len(y):
            raise ValueError("Jumlah baris X dan y berbeda")

        self.classes_ = np.unique(y)
        if len(self.classes_) != 2:
            raise ValueError("SVM ini hanya untuk klasifikasi biner")

        self.kernel_ = self._resolve_kernel()
        self.gamma_ = self._resolve_gamma(X)

        n = len(X)

        ys = np.where(y == self.classes_[1], 1.0, -1.0)

        Cs = self._class_C(ys)

        sq = np.einsum("ij,ij->i", X, X)
        K_diag = self._kernel_diag(X, sq)

        max_rows = max(2, int(self.cache_size * 1024 * 1024 / (8 * n)))
        cache = OrderedDict()

        def kernel_row(i):
            row = cache.get(i)
            if row is not None:
                cache.move_to_end(i)
                return row
            row = self._kernel(X, X[i:i + 1], sq, sq[i:i + 1]).ravel()
            cache[i] = row 
            if len(cache) > max_rows:
                cache.popitem(last=False)
            return row

        alpha = np.zeros(n)
        G = -np.ones(n)
        pos = ys > 0

        up = pos.copy()
        low = ~pos

        def  refresh_mask(t):
            free_up = alpha[t] < Cs[t]
            free_low = alpha[t] > 0
            up[t] = free_up if pos[t] else free_low
            low[t] = free_low if pos[t] else free_up

        neg_inf = np.full(n, -np.inf)
        pos_inf = np.full(n, np.inf)

        it = 0
        while True:
            if 0 <= self.max_iter <= it:
                warnings.warn(
                    f"SMO berhenti di max_iter={self.max_iter} sebelum konvergen",
                    RuntimeWarning,
                )
                break

            yG = -ys * G
            yG_up = np.where(up, yG, neg_inf)
            i = int(np.argmax(yG_up))
            g_max = yG_up[i]
            yG_low = np.where(low, yG, pos_inf)
            g_min = yG_low.min()

            if g_max - g_min < self.tol:
                break

            K_i = kernel_row(i)
            b_t = g_max - yG_low
            a_t = K_diag + (K_diag[i] - 2.0 * K_i)
            np.maximum(a_t, _TAU, out= a_t)
            obj = np.where(b_t > 0, -(b_t * b_t) / a_t, pos_inf)
            j = int(np.argmin(obj))
            K_j = kernel_row(j)

            yi, yj = ys[i], ys[j]
            Ci, Cj = Cs[i], Cs[j]
            ai_old, aj_old = alpha[i], alpha[j]
            quad = K_diag[i] + K_diag[j] - 2.0 * K_i[j]
            if quad <= 0:
                quad = _TAU

            if yi != yj:
                delta = (-G[i] - G[j]) /quad
                diff = ai_old - aj_old
                ai, aj = ai_old + delta, aj_old + delta

                if diff > 0:
                    if aj < 0:
                        aj, ai = 0.0, diff
                elif ai< 0:
                    ai, aj= 0.0, -diff

                if diff > Ci - Cj:
                    if ai > Ci:
                        ai, aj = Ci, Ci - diff
                elif aj > Cj:
                    aj, ai = Cj, Cj + diff
            else: 
                delta = (G[i] - G[j]) /quad
                total = ai_old + aj_old
                ai, aj = ai_old - delta, aj_old + delta

                if total > Ci:
                    if ai > Ci:
                        ai, aj = Ci, total - Ci
                elif aj < 0:
                    aj, ai = 0.0, total
                if total > Cj:
                    if aj > Cj:
                        aj, ai = Cj, total - Cj
                elif ai < 0:
                    ai, aj = 0.0, total

            alpha[i], alpha[j] = ai, aj
            refresh_mask(i)
            refresh_mask(j)

            G += ys * (yi * (ai- ai_old) * K_i + yj * (aj- aj_old) * K_j)

            it += 1
            if self.verbose and it % 1000 == 0:
                print(f"iter {it}: gap={g_max - g_min:.5f}")

        self.n_iter_ = it

        yG = ys * G
        free = (alpha > 0) & (alpha < Cs)
        if np.any(free):
            rho = np.mean(yG[free])
        else:
            at_up = alpha >= Cs
            at_low = alpha <= 0
            ub_mask = ((ys > 0) & at_low) | ((ys < 0) & at_up)
            lb_mask = ((ys > 0) & at_up) | ((ys < 0) & at_low)
            ub = np.min(yG[ub_mask]) if np.any(ub_mask) else np.inf
            lb = np.max(yG[lb_mask]) if np.any(lb_mask) else -np.inf
            rho = (ub + lb) / 2.0
        self.intercept_ = -rho

        sv = alpha > 0
        self.support_ = np.flatnonzero(sv)
        self.support_vectors_ = X[sv]
        self.dual_coef_ = alpha[sv] * ys[sv]
        self.n_support_ = np.array([np.sum(ys[sv] < 0), np.sum(ys[sv] > 0)])
        if self.kernel_ == "linear":
            self.coef_ = self.dual_coef_ @ self.support_vectors_
        return self

    def _check_fitted(self):
        if not hasattr(self, "dual_coef_"):
            raise RuntimeError("Model belum di-fit. Panggil fit() terlebih dahulu.")

    def decision_function(self, X, batch_size=2048):
        self._check_fitted()
        X = np.asarray(X, dtype=np.float64)
        if self.kernel_ == "linear":
            return X @ self.coef_ + self.intercept_
        
        sv = self.support_vectors_
        sq_sv = np.einsum("ij,ij->i", sv, sv)
        out = np.empty(len(X))
        for start in range(0, len(X), batch_size):
            Xb = X[start:start + batch_size]
            out[start:start + batch_size] = (
                self._kernel(Xb, sv, sq_B=sq_sv) @ self.dual_coef_ + self.intercept_
            )
        return out

    def predict(self, X):
        scores = self.decision_function(X)
        return np.where(scores > 0, self.classes_[1], self.classes_[0])

    def score(self, X, y):
        return float(np.mean(self.predict(X) == np.asarray(y).ravel()))

    def save(self, path):
        with open(path, "wb") as f:
            pickle.dump(self, f)

    @staticmethod
    def load(path):
        with open(path, "rb") as f:
            model = pickle.load(f)
        if not isinstance(model, SVM):
            raise TypeError(f"{path} bukan model SVM")
        return model
