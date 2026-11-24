from pathlib import Path

import matplotlib.pyplot as plt
import networkx as nx
from preprocessing import NUMERIC_FEATURES, CATEGORICAL_FEATURES, BINARY_FEATURES

ROOT = Path(__file__).resolve().parent.parent
OUTPUT_DIR = ROOT / "output"


def main():
    OUTPUT_DIR.mkdir(exist_ok=True)

    features = NUMERIC_FEATURES + CATEGORICAL_FEATURES + BINARY_FEATURES
    G = nx.DiGraph([("is_canceled", f) for f in features])

    pos = nx.circular_layout(features)
    pos["is_canceled"] = (0, 0)

    plt.figure(figsize=(12, 12))
    nx.draw(G, pos, with_labels=True, labels={n: n.replace("_", "\n") for n in G},
            node_color=["salmon" if n == "is_canceled" else "lightblue" for n in G],
            node_size=3500, font_size=7, arrowsize=15)
    plt.title("Struktur Naive Bayes: is_canceled -> fitur")
    path = OUTPUT_DIR / "naive_bayes_graph.png"
    plt.savefig(path, dpi=200, bbox_inches="tight")
    print("Graf disimpan di", path)


if __name__ == "__main__":
    main()