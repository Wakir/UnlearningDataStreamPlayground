# ============================================
# 1. IMPORTY
# ============================================

import numpy as np
import pandas as pd
from sklearn.datasets import fetch_openml
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt

from strlearn.ensembles import SEA
from strlearn2.classifiers import SlidingWindowClassifier
from strlearn2.classifiers import SlidingWindowPerceptron
from strlearn2.classifiers import FisherUnlearningClassifier
from strlearn2.classifiers import UnlearningClassifier
from strlearn.evaluators import TestThenTrain


# ============================================
# 2. STREAM MNIST Z DRIFTEM (CHUNK-BASED)
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
    n_classes = len(classes)

    assert chunk_size % n_classes == 0, "chunk_size musi być podzielny przez liczbę klas"

    samples_per_class = chunk_size // n_classes

    # indeksy próbek dla każdej klasy (potasowane)
    class_indices = {
        c: rng.permutation(np.where(y == c)[0])
        for c in classes
    }

    # liczba pełnych chunków
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

    # dodatkowe mieszanie wewnątrz każdego chunka
    ordered_indices = np.array(ordered_indices)

    for i in range(0, len(ordered_indices), chunk_size):
        block = ordered_indices[i:i+chunk_size]
        ordered_indices[i:i+chunk_size] = rng.permutation(block)

    return X[ordered_indices], y[ordered_indices]

def recovery_analysis(accuracy, rolling_acc, drift_chunk, max_chunk):
    drift_chunk_eval = drift_chunk - 1

    baseline = rolling_acc[max_chunk - 6 : max_chunk - 1].mean()

    min_val = min(accuracy[drift_chunk_eval:])
    max_val = max(accuracy[drift_chunk_eval:])
    theta = 0.90 * max_val

    # DROP
    T_drop = None
    for i in range(drift_chunk_eval, len(accuracy)):
        if accuracy[i] < 1.1 * min_val:
            T_drop = i
            break

    # RECOVERY
    T_recovery = None
    if T_drop is not None:
        for i in range(T_drop + 1, len(accuracy)):
            if accuracy[i] >= theta:
                T_recovery = i
                break

    results = {
        "theta": theta,
        "T_drop": T_drop,
        "T_recovery": T_recovery,
        "recovery_time": None
    }

    if T_drop is not None and T_recovery is not None:
        results["recovery_time"] = T_recovery - T_drop

    return results


# ============================================
# 3. WCZYTANIE I PRZYGOTOWANIE MNIST
# ============================================

X, y = fetch_openml("mnist_784", version=1, return_X_y=True)
X = X.to_numpy() / 255.0
y = y.astype(int).to_numpy()

chunk_size = 500
drift_chunk = 40   # 40 * 500 = 20 000 próbek

X, y = make_balanced_chunks(X, y, chunk_size)

stream = MNISTDriftStream(X, y, chunk_size, drift_chunk)
stream2 = MNISTDriftStream(X, y, chunk_size, drift_chunk)


# ============================================
# 4. KLASYFIKATOR SEA + EWALUATOR
# ============================================

#clf = SlidingWindowClassifier(window_size=5)  # L z pseudokodu
clf = SlidingWindowPerceptron(window_size=5) 
#clf2 = UnlearningClassifier(window_size=5)  # L z pseudokodu
clf2 = FisherUnlearningClassifier(window_size=5)

evaluator = TestThenTrain(metrics=(accuracy_score,))
evaluator2 = TestThenTrain(metrics=(accuracy_score,))
# ===== WARM-UP =====
#X0, y0 = stream.get_chunk()
#clf.partial_fit(X0, y0, classes=stream.classes_)
X0, y0 = stream2.get_chunk()
clf2.partial_fit(X0, y0)

# ===== EVALUATION =====
evaluator2.process(stream2, clf2)
evaluator.process(stream, clf)

#batch_times = np.array(clf.batch_times_)
train_times = np.array(clf.train_times_)
train_times2 = np.array(clf2.train_times_)

print("===== TIMING SLIDING WINDOW =====")
#print(f"Mean batch time      : {batch_times.mean():.6f} s")
#print(f"Std batch time       : {batch_times.std():.6f} s")
print(f"Mean training time   : {train_times.mean():.6f} s")
#print(f"Max batch time       : {batch_times.max():.6f} s")

print("===== TIMING UNLEARNING =====")
#print(f"Mean batch time      : {batch_times.mean():.6f} s")
#print(f"Std batch time       : {batch_times.std():.6f} s")
print(f"Mean training time   : {train_times2.mean():.6f} s")
#print(f"Max batch time       : {batch_times.max():.6f} s")

time_efficiency = (train_times.mean() - train_times2.mean()) / train_times.mean()*100

print("===== TIMING EFFICIENCY =====")
print(f"Time gain   : {time_efficiency:.4f} %")

# ============================================
# 5. MEMORY
# ============================================

memory = np.array(clf.memory_usage_)
memory2 = np.array(clf2.memory_usage_)

print("===== MEMORY USAGE SLIDING WINDOW =====")
#print(f"Mean batch time      : {batch_times.mean():.6f} s")
#print(f"Std batch time       : {batch_times.std():.6f} s")
print(f"Mean memory usage  : {memory.mean():.6f} s")
#print(f"Max batch time       : {batch_times.max():.6f} s")

print("===== TIMING UNLEARNING =====")
#print(f"Mean batch time      : {batch_times.mean():.6f} s")
#print(f"Std batch time       : {batch_times.std():.6f} s")
print(f"Mean memory usage  : {memory2.mean():.6f} s")
#print(f"Max batch time       : {batch_times.max():.6f} s")

memory_efficiency = (memory.mean() - memory2.mean()) / memory.mean()*100

print("===== TIMING EFFICIENCY =====")
print(f"Memory gain   : {memory_efficiency:.4f} %")



# ============================================
# 6. ACCURACY PER CHUNK
# ============================================

accuracy = evaluator.scores[0, :, 0]
accuracy2 = evaluator2.scores[0, :, 0]

rolling_acc = (
    pd.Series(accuracy)
    .rolling(window=5, min_periods=1)
    .mean()
    .values
)

rolling_acc2 = (
    pd.Series(accuracy2)
    .rolling(window=5, min_periods=1)
    .mean()
    .values
)

mean_accuracy = np.nanmean(accuracy)
mean_accuracy2 = np.nanmean(accuracy2)

print("===== BASIC RESULTS SLIDING WINDOW=====")
print(f"Mean accuracy: {mean_accuracy:.4f}")

print("===== BASIC RESULTS UNLEARNING=====")
print(f"Mean accuracy: {mean_accuracy2:.4f}")

# ============================================
# 7. RECOVERY ANALYSIS
# ============================================

# indeks driftu w osi ewaluacji
drift_chunk_eval = drift_chunk - 1

max_chunk = len(X) // chunk_size

res1 = recovery_analysis(accuracy, rolling_acc, drift_chunk, max_chunk)
res2 = recovery_analysis(accuracy2, rolling_acc2, drift_chunk, max_chunk)

import matplotlib.pyplot as plt
import numpy as np

plt.figure(figsize=(12, 6))

# --- Sliding Window ---
plt.plot(
    accuracy,
    label="Sliding Window",
    linewidth=2
)

# --- Unlearning ---
plt.plot(
    accuracy2,
    label="Unlearning Classifier",
    linewidth=2
)

# --- Drift ---
plt.axvline(
    drift_chunk - 1,
    linestyle="--",
    label="Drift"
)

# --- Progi recovery ---
plt.axhline(
    res1["theta"],
    linestyle="--",
    alpha=0.6,
    label="Recovery threshold (SW)"
)

plt.axhline(
    res2["theta"],
    linestyle="--",
    alpha=0.6,
    label="Recovery threshold (Unlearning)"
)

# --- Drop & recovery ---
for res, color, name in [
    (res1, "black", "SW"),
    (res2, "green", "Unlearning")
]:
    if res["T_drop"] is not None:
        plt.axvline(
            res["T_drop"],
            linestyle=":",
            color=color,
            alpha=0.8,
            label=f"{name} drop"
        )
    if res["T_recovery"] is not None:
        plt.axvline(
            res["T_recovery"],
            linestyle="-.",
            color=color,
            alpha=0.8,
            label=f"{name} recovery"
        )

plt.xlabel("Chunk")
plt.ylabel("Rolling accuracy")
plt.title("Accuracy over time for MNIST 0-9 + 5-9")
plt.legend()
plt.tight_layout()
plt.show()

recovery_efficiency = (
    (res1["recovery_time"] - res2["recovery_time"])
    / res1["recovery_time"]
    * 100
)

print("===== RECOVERY EFFICIENCY =====")
print(res1)
print(res2)
print(f"Recovery gain: {recovery_efficiency:.2f} %")




