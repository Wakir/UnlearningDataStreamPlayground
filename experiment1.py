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
from strlearn2.ensembles import SlidingWindowSEA
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


# ============================================
# 3. WCZYTANIE I PRZYGOTOWANIE MNIST
# ============================================

X, y = fetch_openml("mnist_784", version=1, return_X_y=True)
X = X.to_numpy() / 255.0
y = y.astype(int).to_numpy()

chunk_size = 500
drift_chunk = 40   # 40 * 500 = 20 000 próbek

stream = MNISTDriftStream(X, y, chunk_size, drift_chunk)


# ============================================
# 4. KLASYFIKATOR SEA + EWALUATOR
# ============================================

clf = SlidingWindowSEA(
    base_estimator=SGDClassifier(loss="log_loss"),
    n_estimators=10,
)

"""clf = SEA(
    base_estimator=SGDClassifier(loss="log_loss"),
    n_estimators=10,
)"""

evaluator = TestThenTrain(metrics=(accuracy_score,))
# ===== WARM-UP =====
X0, y0 = stream.get_chunk()
clf.partial_fit(X0, y0, classes=stream.classes_)

# ===== EVALUATION =====
evaluator.process(stream, clf)



# ============================================
# 5. ACCURACY PER CHUNK
# ============================================

accuracy = evaluator.scores[0, :, 0]

rolling_acc = (
    pd.Series(accuracy)
    .rolling(window=5, min_periods=1)
    .mean()
    .values
)

mean_accuracy = np.nanmean(accuracy)

print("===== BASIC RESULTS =====")
print(f"Mean accuracy: {mean_accuracy:.4f}")

# ============================================
# 6. RECOVERY ANALYSIS
# ============================================

# indeks driftu w osi ewaluacji
drift_chunk_eval = drift_chunk - 1

# --- baseline przed driftem ---
baseline = rolling_acc[
    drift_chunk_eval - 5 : drift_chunk_eval
].mean()
theta = 0.9 * baseline  # próg recovery (90% baseline)

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
    # 7. RAPORT
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


# ============================================
# 8. WYKRES RECOVERY
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
plt.title(f"SEA_Window – Accuracy over time for MNIST 0-9 + 5-9 (mean = {mean_accuracy:.3f})")
plt.legend()
plt.tight_layout()
plt.show()





