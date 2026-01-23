# ============================================
# 1. IMPORTY
# ============================================

import numpy as np
import pandas as pd
from sklearn.datasets import fetch_openml
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt

from strlearn2.classifiers import SlidingWindowClassifier
from strlearn2.classifiers import UnlearningClassifier
from strlearn.evaluators import TestThenTrain


# ============================================
# 2. STREAM MNIST Z DRIFTEM (CHUNK-BASED)
# ============================================

class MNISTDriftStream:
    def __init__(self, X, y, chunk_size=500):
        self.X = X
        self.y = y
        self.chunk_size = chunk_size

        self.n_chunks = len(X) // chunk_size
        self.chunk_id = 0
        self.previous_chunk = None
        self.classes_ = np.arange(10)

    def get_chunk(self):
        start = self.chunk_id * self.chunk_size
        end = start + self.chunk_size

        Xc = self.X[start:end]
        yc = self.y[start:end]

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

def recovery_analysis(rolling_acc, drift_chunk, max_chunk, title):
    drift_chunk_eval = drift_chunk - 1

    # --- baseline na końcu dryftu ---
    baseline = rolling_acc[
        max_chunk - 6 : max_chunk - 1
    ].mean()
    theta = baseline  # próg recovery (90% baseline)

    # === DROP ===
    T_drop = None
    for i in range(drift_chunk_eval, len(rolling_acc)):
        if rolling_acc[i] < theta:
            T_drop = i
            break

    # === RECOVERY ===
    T_recovery = None
    if T_drop is not None:
        for i in range(T_drop + 1, len(rolling_acc)):
            if rolling_acc[i] >= theta:
                T_recovery = i
                break

    if T_recovery is not None:
        # --- metryki recovery ---
        recovery_time = T_recovery - T_drop
        recovery_depth = baseline - rolling_acc[T_drop]
        recovery_quality = (
            rolling_acc[T_recovery:T_recovery + 5].mean() / baseline
        )

        aurc = np.trapz(
            rolling_acc[T_drop:T_recovery],
            dx=1
        )


        # ============================================
        # 8. RAPORT
        # ============================================

        print("===== RECOVERY ANALYSIS =====")
        print(f"Baseline accuracy      : {baseline:.3f}")
        print(f"Recovery threshold     : {theta:.3f}")
        print(f"T_drop (chunk)         : {T_drop}")
        print(f"T_recovery (chunk)     : {T_recovery}")
        print(f"Recovery time (chunks) : {recovery_time}")
        print(f"Recovery depth         : {recovery_depth:.3f}")
        print(f"Recovery quality       : {recovery_quality:.3f}")
        print(f"AURC                   : {aurc:.3f}")
    else:
        print("===== RECOVERY ANALYSIS =====")
        print("No significant dropout detected.")
        recovery_time = 0


    # ============================================
    # 9. WYKRES RECOVERY
    # ============================================

    import matplotlib.pyplot as plt

    plt.figure(figsize=(10, 5))

    # --- accuracy ---
    plt.plot(
        accuracy,
        alpha=0.35,
        label="Accuracy per chunk"
    )

    plt.plot(
        rolling_acc,
        linewidth=2,
        label="Rolling accuracy"
    )

    # --- drift ---
    plt.axvline(
        drift_chunk_eval,
        color="orange",
        linestyle="--",
        label="Drift"
    )

    # --- próg recovery ---
    plt.axhline(
        theta,
        color="red",
        linestyle="--",
        alpha=0.7,
        label="Recovery threshold"
    )

    # --- drop ---
    if T_drop is not None:
        plt.axvline(
            T_drop,
            color="black",
            linestyle=":",
            label="Performance drop"
        )

    # --- recovery ---
    if T_recovery is not None:
        plt.axvline(
            T_recovery,
            color="green",
            linestyle=":",
            label=f"Recovery (Δ={T_recovery - T_drop} chunks)"
        )

    plt.xlabel("Chunk (evaluation axis)")
    plt.ylabel("Accuracy")
    plt.title(f"{title} – Accuracy over time for MNIST + Foged C_MNIST (mean = {mean_accuracy:.3f})")
    plt.legend()
    plt.tight_layout()
    plt.show()
    return recovery_time


# ============================================
# 3. WCZYTANIE I PRZYGOTOWANIE MNIST
# ============================================

X_adv = np.load("mnist_adv/train_images.npy")
y_adv = np.load("mnist_adv/train_labels.npy")

if X_adv.ndim == 4 and X_adv.shape[-1] == 1:
    X_adv = X_adv.squeeze(-1)

# jeśli obrazy są 28x28 → spłaszcz
if X_adv.ndim == 3:
    X_adv = X_adv.reshape(X_adv.shape[0], -1)


# normalizacja
X = X_adv.astype(np.float32) / 255.0
y = y_adv.astype(int)

print(X_adv.shape, y_adv.shape)


X_clean, y_clean = fetch_openml("mnist_784", version=1, return_X_y=True)
X_clean = X_clean.to_numpy() / 255.0
y_clean = y_clean.astype(int).to_numpy()

print(X_clean.shape, y_clean.shape)

chunk_size = 500

X_clean, y_clean = make_balanced_chunks(X_clean, y_clean, chunk_size)
X_adv, y_adv = make_balanced_chunks(X_adv, y_adv, chunk_size)

X = np.vstack([X_clean, X_adv])
y = np.hstack([y_clean, y_adv])

drift_chunk = len(X_clean) // chunk_size

stream = MNISTDriftStream(X, y, chunk_size)
stream2 = MNISTDriftStream(X, y, chunk_size)


# ============================================
# 4. KLASYFIKATOR SEA + EWALUATOR
# ============================================

clf = SlidingWindowClassifier(window_size=20)  # L z pseudokodu
clf2 = UnlearningClassifier(window_size=20)  # L z pseudokodu

evaluator = TestThenTrain(metrics=(accuracy_score,))
evaluator2 = TestThenTrain(metrics=(accuracy_score,))
# ===== WARM-UP =====
X0, y0 = stream.get_chunk()
clf.partial_fit(X0, y0, classes=stream.classes_)
clf2.partial_fit(X0, y0, classes=stream.classes_)

# ===== EVALUATION =====
evaluator.process(stream, clf)
evaluator2.process(stream2, clf2)

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

recovery_time1 = recovery_analysis(rolling_acc, drift_chunk, max_chunk, "Sliding_Window")
recovery_time2 = recovery_analysis(rolling_acc2, drift_chunk, max_chunk, "Unlearning_Classifier")

recovery_efficiency = (recovery_time1- recovery_time2) / recovery_time1*100

print("===== RECOVERY EFFICIENCY =====")
print(f"Recovery gain   : {recovery_efficiency:.4f} %")





