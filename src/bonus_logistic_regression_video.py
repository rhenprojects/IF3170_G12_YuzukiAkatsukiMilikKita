from pathlib import Path

import imageio_ffmpeg
import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.animation import FFMpegWriter

from algorithms.logistic_regression import LogisticRegression
from preprocessing import BINARY_FEATURES, CATEGORICAL_FEATURES, NUMERIC_FEATURES, build_imputer

# Referensi
# Log-loss (negatif log conditional likelihood) dan stochastic gradient ascent:
#   - Slide IF3170 Materi 11, Linear Regression & Logistic Regression, ITB

ROOT = Path(__file__).resolve().parent.parent
FEATURE = "days_before_arrival"
VIDEO_PATH = ROOT / "output" / "logistic_regression_contour.mp4"

LEARNING_RATE = 0.0005
N_EPOCHS = 2
HISTORY_EVERY = 100
FPS = 30


# Mengambil satu fitur aja biar parameternya cuma theta0 dan theta1, distandardisasi biar konturnya gak gepeng.
def load_feature():
    data = pd.read_csv(ROOT / "data" / "train.csv", dtype={"id": str})
    columns = NUMERIC_FEATURES + CATEGORICAL_FEATURES + BINARY_FEATURES
    clean = build_imputer().fit_transform(data[columns])

    x = clean[FEATURE].to_numpy(dtype=float)
    x = (x - x.mean()) / x.std()
    return x.reshape(-1, 1), data["is_canceled"].to_numpy()


def log_loss_grid(x, y, theta0, theta1):
    score = theta0[..., None] + theta1[..., None] * x[:, 0]
    prob = np.clip(1 / (1 + np.exp(-np.clip(score, -500, 500))), 1e-12, 1 - 1e-12)
    return -np.mean(y * np.log(prob) + (1 - y) * np.log(1 - prob), axis=-1)


def main():
    x, y = load_feature()

    model = LogisticRegression(learning_rate=LEARNING_RATE, n_epochs=N_EPOCHS, history_every=HISTORY_EVERY)
    model.fit(x, y)
    path = np.array(model.weights_history)
    path_loss = log_loss_grid(x, y, path[:, 0], path[:, 1])

    margin = 0.5
    theta0_range = np.linspace(min(path[:, 0].min(), -2) - margin, max(path[:, 0].max(), 0) + margin, 120)
    theta1_range = np.linspace(min(path[:, 1].min(), 0) - margin, max(path[:, 1].max(), 2) + margin, 120)
    grid0, grid1 = np.meshgrid(theta0_range, theta1_range)
    surface = np.vstack([log_loss_grid(x, y, row0, row1) for row0, row1 in zip(grid0, grid1)])
    best = np.unravel_index(surface.argmin(), surface.shape)

    fig, (ax_contour, ax_loss) = plt.subplots(1, 2, figsize=(12, 5))
    levels = np.quantile(surface, np.linspace(0, 1, 25))
    contour = ax_contour.contour(grid0, grid1, surface, levels=np.unique(levels), cmap="viridis", linewidths=0.8)
    ax_contour.clabel(contour, contour.levels[::4], fontsize=7, fmt="%.3f")
    ax_contour.plot(grid0[best], grid1[best], marker="*", color="tab:red", markersize=12, label="minimum log-loss")
    ax_contour.set_xlabel("theta0 (bias)")
    ax_contour.set_ylabel(f"theta1 ({FEATURE}, distandardisasi)")
    trail, = ax_contour.plot([], [], color="tab:orange", linewidth=1.2, label="lintasan SGA")
    point, = ax_contour.plot([], [], marker="o", color="tab:orange", markeredgecolor="black")
    ax_contour.legend(loc="upper right", fontsize=8)

    ax_loss.set_xlim(0, len(path) * HISTORY_EVERY)
    ax_loss.set_ylim(path_loss.min() - 0.01, path_loss.max() + 0.01)
    ax_loss.set_xlabel("jumlah update")
    ax_loss.set_ylabel("log-loss rata-rata")
    ax_loss.grid(alpha=0.3)
    loss_line, = ax_loss.plot([], [], color="tab:orange")
    title = fig.suptitle("")

    matplotlib.rcParams["animation.ffmpeg_path"] = imageio_ffmpeg.get_ffmpeg_exe()
    VIDEO_PATH.parent.mkdir(exist_ok=True)
    writer = FFMpegWriter(fps=FPS)
    updates_per_epoch = len(y)

    with writer.saving(fig, str(VIDEO_PATH), dpi=100):
        for i in range(len(path)):
            trail.set_data(path[: i + 1, 0], path[: i + 1, 1])
            point.set_data([path[i, 0]], [path[i, 1]])
            loss_line.set_data(np.arange(i + 1) * HISTORY_EVERY, path_loss[: i + 1])
            n_updates = i * HISTORY_EVERY
            title.set_text(
                f"Epoch {n_updates // updates_per_epoch + 1}, update ke-{n_updates}: "
                f"theta0 = {path[i, 0]:.3f}, theta1 = {path[i, 1]:.3f}, log-loss = {path_loss[i]:.4f}"
            )
            writer.grab_frame()

    print(f"Jumlah frame: {len(path)}, durasi sekitar {len(path) / FPS:.0f} detik")
    print(f"Theta akhir: theta0 = {path[-1, 0]:.3f}, theta1 = {path[-1, 1]:.3f}, log-loss = {path_loss[-1]:.4f}")
    print(f"Minimum di grid: theta0 = {grid0[best]:.3f}, theta1 = {grid1[best]:.3f}, log-loss = {surface[best]:.4f}")
    print("Video disimpan di", VIDEO_PATH)


if __name__ == "__main__":
    main()
