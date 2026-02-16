# ============================================
# 1. IMPORTY
# ============================================

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.datasets import fetch_openml
from sklearn.metrics import accuracy_score

from strlearn2.classifiers import (
    SlidingWindowPerceptron,
    SlidingWindowClassifier,
    FisherUnlearningClassifier,
    UnlearningClassifier
)
from strlearn.evaluators import TestThenTrain


# ============================================
# 2. STREAM + POMOCNICZE
# ============================================

class MNISTDriftStream:
    def __init__(self, X, y, chunk_size=500, drift_chunk=40):
        self.X = X
        self.y = y
        self.chunk_size = chunk_size
        self.drift_chunk = drift_chunk

        self.n_chunks = len(X) // chunk_size
        self.chunk_id = 0
        self.previous_chunk = None
        self.classes_ = np.arange(10)

    def get_chunk(self):
        start = self.chunk_id * self.chunk_size
        end = start + self.chunk_size

        Xc = self.X[start:end]
        yc = self.y[start:end]

        # --- CONCEPT DRIFT ---
        if self.chunk_id >= self.drift_chunk:
            mask = yc >= 5  # po drifcie tylko cyfry 5–9
            Xc = Xc[mask]
            yc = yc[mask]

        self.previous_chunk = (Xc, yc)
        self.chunk_id += 1

        return Xc, yc

    def is_dry(self):
        return self.chunk_id >= self.n_chunks - 1


def make_balanced_chunks(X, y, chunk_size=500, seed=42):
    rng = np.random.default_rng(seed)
    classes = np.unique(y)
    samples_per_class = chunk_size // len(classes)

    class_indices = {
        c: rng.permutation(np.where(y == c)[0])
        for c in classes
    }

    max_chunks = min(
        len(class_indices[c]) // samples_per_class
        for c in classes
    )

    ordered_indices = []
    for i in range(max_chunks):
        for c in classes:
            start = i * samples_per_class
            end = start + samples_per_class
            ordered_indices.extend(class_indices[c][start:end])

    ordered_indices = np.array(ordered_indices)

    for i in range(0, len(ordered_indices), chunk_size):
        ordered_indices[i:i+chunk_size] = rng.permutation(
            ordered_indices[i:i+chunk_size]
        )

    return X[ordered_indices], y[ordered_indices]


def recovery_analysis(accuracy, rolling_acc, drift_chunk, max_chunk):
    drift_eval = drift_chunk - 1
    min_val = np.min(accuracy[drift_eval:])
    max_val = np.max(accuracy[drift_eval:])
    theta = 0.9 * max_val

    T_drop, T_rec = None, None

    for i in range(drift_eval, len(accuracy)):
        if accuracy[i] <= 1.1 * min_val:
            T_drop = i
            break

    if T_drop is not None:
        for i in range(T_drop + 1, len(accuracy)):
            if accuracy[i] >= theta:
                T_rec = i
                break

    return {
        "theta": theta,
        "T_drop": T_drop,
        "T_recovery": T_rec,
        "recovery_time": None if T_drop is None or T_rec is None else T_rec - T_drop
    }


# ============================================
# 3. WCZYTANIE MNIST + DRIFT
# ============================================

X, y = fetch_openml("mnist_784", version=1, return_X_y=True)
X = X.to_numpy() / 255.0
y = y.astype(int).to_numpy()

chunk_size = 500
drift_chunk = 40   # 40 * 500 = 20 000 próbek

X, y = make_balanced_chunks(X, y, chunk_size)
classes = np.arange(10)


# ============================================
# 4. FUNKCJA EKSPERYMENTU
# ============================================

def run_experiment(clf_class, window_size):
    stream = MNISTDriftStream(X, y, chunk_size)
    evaluator = TestThenTrain(metrics=(accuracy_score,))
    clf = clf_class(window_size=window_size)

    # warm-up
    X0, y0 = stream.get_chunk()
    clf.partial_fit(X0, y0, classes=classes)

    evaluator.process(stream, clf)

    accuracy = evaluator.scores[0, :, 0]
    rolling_acc = (
        pd.Series(accuracy)
        .rolling(window=5, min_periods=1)
        .mean()
        .values
    )

    train_times = np.array(clf.train_times_)
    memory = np.array(clf.memory_usage_)

    recovery = recovery_analysis(
        accuracy,
        rolling_acc,
        drift_chunk,
        len(X) // chunk_size
    )

    return {
        "window": window_size,
        "accuracy": accuracy,
        "rolling": rolling_acc,
        "mean_acc": np.nanmean(accuracy),
        "mean_time": train_times.mean(),
        "mean_memory": memory.mean(),
        **recovery
    }


# ============================================
# 5. URUCHOMIENIE EKSPERYMENTÓW
# ============================================

window_sizes = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]

results_unlearning = []
results_fisher = []

for w in window_sizes:
    print(f"Running SlidingWindow | L={w}")
    results_unlearning.append(run_experiment(SlidingWindowPerceptron, w))
    #results_unlearning.append(run_experiment(SlidingWindowClassifier, w))

    print(f"Running Unlearning | L={w}")
    results_fisher.append(run_experiment(UnlearningClassifier, w))
    #results_fisher.append(run_experiment(FisherUnlearningClassifier, w))


# ============================================
# 6. TABELA WYNIKÓW
# ============================================

def to_df(results, name):
    return pd.DataFrame([
        {
            "classifier": name,
            "window": r["window"],
            "mean_accuracy": r["mean_acc"],
            "mean_train_time": r["mean_time"],
            "mean_memory": r["mean_memory"],
            "T_drop": r["T_drop"],
            "T_recovery": r["T_recovery"],
            "recovery_time": r["recovery_time"],
        }
        for r in results
    ])

df = pd.concat([
    to_df(results_unlearning, "SlidingWindow"),
    #to_df(results_fisher, "Unlearning")
])

print("\n===== PODSUMOWANIE =====")
print(df)


# ============================================
# 7. WYKRESY
# ============================================

plt.figure(figsize=(12, 6))
for r in results_unlearning:
    plt.plot(r["rolling"], label=f"L={r['window']}")
plt.axvline(drift_chunk - 1, linestyle="--", color="black", label="Drift")
plt.title("SlidingWindow for MLPClassifier in MNIST dataset– change in window length")
plt.xlabel("Chunk")
plt.ylabel("Rolling accuracy")
plt.legend()
plt.tight_layout()
plt.show()

"""plt.figure(figsize=(12, 6))
for r in results_fisher:
    plt.plot(r["rolling"], label=f"L={r['window']}")
plt.axvline(drift_chunk - 1, linestyle="--", color="black", label="Drift")
plt.title("UnlearningClassifier for SGDClassifer in MNIST dataset– change in window length")
plt.xlabel("Chunk")
plt.ylabel("Rolling accuracy")
plt.legend()
plt.tight_layout()
plt.show()"""
