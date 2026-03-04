import numpy as np
import pandas as pd
from tensorflow.keras.datasets import mnist, cifar10
from strlearn.evaluators import TestThenTrain
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score,  balanced_accuracy_score as bac, precision_score, recall_score
from specificity import specificity, specificity_macro
from strlearn2.classifiers import SlidingWindowPerceptron
from collections import defaultdict


def recovery_analysis(metric, rolling_metric, drift_chunk, max_chunk):

    # ❗ brak driftu lub drift poza zakresem
    if drift_chunk is None:
        return {
            "theta": None,
            "T_drop": None,
            "T_recovery": None,
            "recovery_time": None,
            "status": "no_drift"
        }

    drift_chunk_eval = drift_chunk - 1

    if drift_chunk_eval < 0 or drift_chunk_eval >= len(metric):
        return {
            "theta": None,
            "T_drop": None,
            "T_recovery": None,
            "recovery_time": None,
            "status": "drift_out_of_range"
        }

    post_drift = metric[drift_chunk_eval:]

    if len(post_drift) == 0:
        return {
            "theta": None,
            "T_drop": None,
            "T_recovery": None,
            "recovery_time": None,
            "status": "empty_post_drift"
        }

    # ================= NORMALNA ANALIZA =================

    min_val = min(post_drift)
    max_val = max(post_drift)
    theta = 0.90 * max_val

    # DROP
    T_drop = None
    for i in range(drift_chunk_eval, len(metric)):
        if metric[i] < 1.1 * min_val:
            T_drop = i
            break

    # RECOVERY
    T_recovery = None
    if T_drop is not None:
        for i in range(T_drop + 1, len(metric)):
            if metric[i] >= theta:
                T_recovery = i
                break

    recovery_time = None
    if T_drop is not None and T_recovery is not None:
        recovery_time = T_recovery - T_drop

    return {
        "theta": theta,
        "T_drop": T_drop,
        "T_recovery": T_recovery,
        "recovery_time": recovery_time,
        "status": "ok"
    }

from tensorflow.keras.datasets import fashion_mnist, cifar10
import numpy as np
from collections import defaultdict


class DataStream:

    def __init__(
        self,
        chunk_size: int,
        dataset_name: str,
        semantic_case_1: str,
        semantic_case_2: str,
        random_seed: int = 42,
    ):

        self.chunk_size = chunk_size
        self.dataset_name = dataset_name.upper()
        self.semantic_case_1 = semantic_case_1
        self.semantic_case_2 = semantic_case_2
        self.classes_ = np.array([0, 1])
        self.random_seed = random_seed
        self.rng = np.random.default_rng(random_seed)

        self.X, self.y_original = self._load_raw_dataset()

        self._shuffle()

        self.chunks = self._create_chunks()
        self.n_chunks = len(self.chunks)
        self.drift_chunk = self.n_chunks // 2

        self.reset()

    # --------------------------------------------------
    # RAW DATA
    # --------------------------------------------------

    def _load_raw_dataset(self):

        if self.dataset_name == "FASHION-MNIST":
            (X1, y1), (X2, y2) = fashion_mnist.load_data()
            X = np.concatenate([X1, X2])
            y = np.concatenate([y1, y2])
            X = X[..., np.newaxis]

        elif self.dataset_name == "CIFAR-10":
            (X1, y1), (X2, y2) = cifar10.load_data()
            X = np.concatenate([X1, X2])
            y = np.concatenate([y1.flatten(), y2.flatten()])

        else:
            raise ValueError("Supported datasets: FASHION-MNIST, CIFAR-10")

        return X.astype(np.float32) / 255.0, y

    # --------------------------------------------------
    # SEMANTIC MAPS
    # --------------------------------------------------

    def _get_positive_set(self, case):

        if self.dataset_name == "CIFAR-10":

            if case == "animals":
                return {2, 3, 4, 5, 6, 7}

            elif case == "swim":
                return {6, 8}

            elif case == "fly":
                return {0, 2}

            else:
                raise ValueError("CIFAR cases: animals, swim, fly")

        elif self.dataset_name == "FASHION-MNIST":

            if case == "fashion_v1":
                return {0, 2, 4, 7, 9}

            elif case == "fashion_v2":
                return {1, 3, 5}

            else:
                raise ValueError("Fashion cases: fashion_v1, fashion_v2")

    # --------------------------------------------------
    # STREAM API
    # --------------------------------------------------

    def reset(self):
        self.chunk_id = 0
        self.previous_chunk = None

    def __len__(self):
        return self.n_chunks

    def get_chunk(self, i=None):

        if i is None:
            i = self.chunk_id

        if i >= self.n_chunks:
            raise IndexError("Chunk index out of range")

        idx = self.chunks[i]
        X_chunk = self.X[idx]
        y_raw = self.y_original[idx]

        # WYBÓR SEMANTYKI
        if i < self.drift_chunk:
            positive_set = self._get_positive_set(self.semantic_case_1)
        else:
            positive_set = self._get_positive_set(self.semantic_case_2)

        y_chunk = np.array(
            [1 if label in positive_set else 0 for label in y_raw]
        )

        self.previous_chunk = (X_chunk, y_chunk)
        self.chunk_id = i + 1

        return X_chunk, y_chunk

    def __iter__(self):
        self.reset()
        return self

    def __next__(self):
        if self.chunk_id >= self.n_chunks:
            raise StopIteration
        return self.get_chunk()

    # --------------------------------------------------
    # CHUNKING
    # --------------------------------------------------

    def _shuffle(self):
        idx = self.rng.permutation(len(self.X))
        self.X = self.X[idx]
        self.y_original = self.y_original[idx]

    def _create_chunks(self):
        chunks = []
        for start in range(0, len(self.X), self.chunk_size):
            end = start + self.chunk_size
            if end > len(self.X):
                break
            chunks.append(np.arange(start, end))
        return chunks
    def is_dry(self):
        return self.chunk_id >= self.n_chunks - 1
    

def run_experiment(chunk_size, dataset_name, semantic_case_1, semantic_case_2, window_size, random_seed, metrics):
    stream = DataStream(
        chunk_size=chunk_size,
        dataset_name=dataset_name,
        semantic_case_1=semantic_case_1,
        semantic_case_2=semantic_case_2,
        random_seed=random_seed,
    )

    clf = SlidingWindowPerceptron(window_size=window_size)
    evaluator = TestThenTrain(metrics=list(metrics.values()))

    X0, y0 = next(iter(stream))
    clf.partial_fit(X0, y0, classes=stream.classes_)

    evaluator.process(stream, clf)

    scores = evaluator.scores[0]  # (metrics, time)
    train_times = np.array(clf.train_times_)
    memory = np.array(clf.memory_usage_)

    return {
        "metric_curves": {
            name: scores[:, i]   
            for i, name in enumerate(metrics.keys())
        },
        "drift_chunk": stream.drift_chunk,
        "max_chunk": stream.n_chunks,
        "mean_time": train_times.mean(),
        "mean_memory": memory.mean(),
    }

import mlflow

def mlflow_run(chunk_size, dataset_name, semantic_case_1, semantic_case_2, window_size, random_seed, metrics):
    with mlflow.start_run(nested=True):

        mlflow.log_params({
            "chunk_size": chunk_size,
            "dataset_name": dataset_name,
            "semantic_case_1": semantic_case_1,
            "semantic_case_2": semantic_case_2,
            "window_size": window_size,
            "random_seed": random_seed
        })

        output = run_experiment(
            chunk_size,
            dataset_name,
            semantic_case_1,
            semantic_case_2,
            window_size,
            random_seed,
            metrics
        )

        curves = output["metric_curves"]
        drift_chunk = output["drift_chunk"]
        max_chunk = output["max_chunk"]
        mean_time = output["mean_time"]
        mean_memory = output["mean_memory"]

        recovery_results_all = {}

        for metric_name, values in curves.items():

            # 📈 1. PRZEBIEG METRYKI
            for step, value in enumerate(values):
                mlflow.log_metric(metric_name, float(value), step=step)

            # 📉 2. ROLLING
            rolling = (
                pd.Series(values)
                .rolling(window=5, min_periods=1)
                .mean()
                .values
            )

            # ♻️ 3. RECOVERY ANALYSIS
            recovery = recovery_analysis(
                values,
                rolling,
                drift_chunk,
                max_chunk
            )

            recovery_results_all[metric_name] = recovery

            # 🔹 4. ZAPIS DO MLFLOW (metryki scalar)
            if recovery["status"] == "ok":
                for k, v in recovery.items():
                    if k != "status" and v is not None:
                        mlflow.log_metric(f"{metric_name}_{k}", float(v))
            else:
                mlflow.log_param(f"{metric_name}_recovery_status", recovery["status"])
        mlflow.log_metric("drift_chunk", drift_chunk)
        mlflow.log_metric("mean_time", mean_time)
        mlflow.log_metric("mean_memory", mean_memory)

# HIPERPARAMETRY

chunk_sizes = [100]
dataset_name = "FASHION-MNIST"
semantic_cases_1 = ["fashion_v1", "fashion_v2"]
semantic_cases_2 = ["fashion_v1", "fashion_v2"]
window_sizes = [20, 40, 60, 80, 100]
random_seeds = [42, 65, 88]

from functools import partial

#NIE HIPERPARAMETRY (METRYKi DO ZAPISU)
metrics = {
    "accuracy": accuracy_score,
    "balanced_accuracy": bac,
    "precision_macro": partial(precision_score, average="macro"),
    "recall_macro": partial(recall_score, average="macro"),
    "f1_macro": partial(f1_score, average="macro"),
    "specificity_macro": specificity_macro
}

mlflow.set_experiment("FashionMNIST_ChunkSize")

from joblib import Parallel, delayed
import itertools

# 🔽 GENEROWANIE TYLKO POPRAWNYCH KOMBINACJI
param_grid = [
    (chunk_size, semantic_case_1, semantic_case_2, window_size, random_seed)
    for chunk_size, semantic_case_1, semantic_case_2, window_size, random_seed
    in itertools.product(
        chunk_sizes,
        semantic_cases_1,
        semantic_cases_2,
        window_sizes,
        random_seeds
    )
    if semantic_case_1 != semantic_case_2
]

print(f"Liczba uruchamianych eksperymentów: {len(param_grid)}")

from tensorflow.keras.datasets import fashion_mnist

(X1, y1), (X2, y2) = fashion_mnist.load_data()
print("Download OK")

results = Parallel(n_jobs=-1, verbose=10)(
    delayed(mlflow_run)(
        chunk_size,
        dataset_name,
        semantic_case_1,
        semantic_case_2,
        window_size,
        random_seed,
        metrics
    )
    for chunk_size, semantic_case_1, semantic_case_2, window_size, random_seed in param_grid
)

df = pd.DataFrame(results)
print(df.sort_values("accuracy", ascending=False).head())

"""drift_chunk = stream.noise_change_chunk

drift_chunk_eval = drift_chunk - 1

max_chunk =  stream.n_chunks

res1 = recovery_analysis(accuracy, rolling_acc, drift_chunk, max_chunk)

import matplotlib.pyplot as plt
import numpy as np

plt.figure(figsize=(12, 6))

# --- Sliding Window ---
plt.plot(
    accuracy,
    label="Sliding Window",
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

# --- Drop & recovery ---
for res, color, name in [
    (res1, "black", "SW"),
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
plt.show()"""