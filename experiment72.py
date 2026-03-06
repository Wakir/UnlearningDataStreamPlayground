import numpy as np
import pandas as pd
from tensorflow.keras.datasets import mnist, cifar10
from strlearn.evaluators import TestThenTrain
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score,  balanced_accuracy_score as bac, precision_score, recall_score
from specificity import specificity, specificity_macro
from strlearn2.classifiers import SlidingWindowPerceptron, FisherUnlearningAdam
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

class DataStream:

    def __init__(
        self,
        chunk_size: int,
        dataset_name: str,
        start_noise: float,
        target_noise: float,
        noise_step: float = 0.02,
        drift_type: str = "incremental",   # sudden / gradual / incremental
        overlap_chunks: int = 5,
        random_seed: int = 42,
    ):

        self.chunk_size = chunk_size
        self.dataset_name = dataset_name.upper()

        self.start_noise = start_noise
        self.target_noise = target_noise
        self.noise_step = noise_step

        self.drift_type = drift_type
        self.overlap_chunks = overlap_chunks

        self.rng = np.random.default_rng(random_seed)

        # -----------------------
        # Load dataset
        # -----------------------

        self.X, self.y = self._load_dataset()
        self.classes_ = np.unique(self.y)

        self._shuffle()

        self.chunks = self._create_balanced_chunks()

        self.n_chunks = len(self.chunks)

        # -----------------------
        # Drift definition
        # -----------------------

        self.drift_start = self.n_chunks // 2

        if drift_type == "incremental":

            self.drift_chunks = int(
                np.ceil(abs(target_noise - start_noise) / abs(noise_step))
            )

            self.drift_end = self.drift_start + self.drift_chunks

        elif drift_type == "gradual":

            self.drift_end = self.drift_start + overlap_chunks

        else:  # sudden
            self.drift_end = self.drift_start

        self.reset()

    # --------------------------------------------------
    # API for stream evaluators
    # --------------------------------------------------

    def reset(self):

        self.chunk_id = 0
        self.previous_chunk = None

    def __len__(self):

        return self.n_chunks

    def __iter__(self):

        self.reset()
        return self

    def __next__(self):

        if self.chunk_id >= self.n_chunks:
            raise StopIteration

        return self.get_chunk()

    def is_dry(self):

        return self.chunk_id >= self.n_chunks - 1

    # --------------------------------------------------
    # Stream logic
    # --------------------------------------------------

    def get_chunk(self, i=None):

        if i is None:
            i = self.chunk_id

        if i >= self.n_chunks:
            raise IndexError("Chunk index out of range")

        idx = self.chunks[i]

        X_chunk = self.X[idx]
        y_chunk = self.y[idx]

        noise_level = self._current_noise(i)

        X_chunk = self._add_noise(X_chunk, noise_level)

        self.previous_chunk = (X_chunk, y_chunk)

        self.chunk_id = i + 1

        return X_chunk, y_chunk

    # --------------------------------------------------
    # Noise schedule
    # --------------------------------------------------

    def _current_noise(self, chunk_id):

        # BEFORE DRIFT
        if chunk_id < self.drift_start:
            return self.start_noise

        # -------------------------
        # SUDDEN
        # -------------------------

        if self.drift_type == "sudden":

            return self.target_noise

        # -------------------------
        # GRADUAL
        # -------------------------

        if self.drift_type == "gradual":

            if chunk_id <= self.drift_end:

                progress = (chunk_id - self.drift_start) / (
                    self.drift_end - self.drift_start
                )

                return (
                    self.start_noise * (1 - progress)
                    + self.target_noise * progress
                )

            return self.target_noise

        # -------------------------
        # INCREMENTAL
        # -------------------------

        if self.drift_type == "incremental":

            if chunk_id <= self.drift_end:

                step = chunk_id - self.drift_start

                noise = self.start_noise + step * self.noise_step

                if self.start_noise < self.target_noise:
                    noise = min(noise, self.target_noise)
                else:
                    noise = max(noise, self.target_noise)

                return noise

            return self.target_noise

    # --------------------------------------------------
    # Dataset loading
    # --------------------------------------------------

    def _load_dataset(self):

        if self.dataset_name == "MNIST":

            from tensorflow.keras.datasets import mnist

            (X1, y1), (X2, y2) = mnist.load_data()

            X = np.concatenate([X1, X2])
            y = np.concatenate([y1, y2])

            X = X[..., np.newaxis]

        elif self.dataset_name == "CIFAR-10":

            from tensorflow.keras.datasets import cifar10

            (X1, y1), (X2, y2) = cifar10.load_data()

            X = np.concatenate([X1, X2])
            y = np.concatenate([y1.flatten(), y2.flatten()])

        else:
            raise ValueError("Supported datasets: MNIST, CIFAR-10")

        return X.astype(np.float32) / 255.0, y

    # --------------------------------------------------
    # Utilities
    # --------------------------------------------------

    def _shuffle(self):

        idx = self.rng.permutation(len(self.X))

        self.X = self.X[idx]
        self.y = self.y[idx]

    def _create_balanced_chunks(self):

        per_class_indices = defaultdict(list)

        for i, label in enumerate(self.y):
            per_class_indices[label].append(i)

        n_classes = len(self.classes_)
        samples_per_class = self.chunk_size // n_classes

        pointers = {c: 0 for c in self.classes_}

        chunks = []

        while True:

            chunk_idx = []

            for c in self.classes_:

                start = pointers[c]
                end = start + samples_per_class

                if end > len(per_class_indices[c]):
                    return chunks

                chunk_idx.extend(per_class_indices[c][start:end])

                pointers[c] = end

            self.rng.shuffle(chunk_idx)

            chunks.append(chunk_idx)

    # --------------------------------------------------
    # Noise injection
    # --------------------------------------------------

    def _add_noise(self, X, noise_percent, sigma=0.5):

        if noise_percent <= 0:
            return X

        X_noisy = X.copy()

        N, H, W, C = X.shape

        total_pixels = H * W * C
        n_noisy = int(noise_percent * total_pixels)

        for i in range(N):

            idx = self.rng.choice(total_pixels, n_noisy, replace=False)

            noise = self.rng.normal(0, sigma, n_noisy)

            flat = X_noisy[i].reshape(-1)

            flat[idx] += noise

            X_noisy[i] = flat.reshape(H, W, C)

        return np.clip(X_noisy, 0.0, 1.0)

def run_experiment(chunk_size, start_noise, target_noise, noise_step, window_size, random_seed, metrics):
    stream = DataStream(
        chunk_size=chunk_size,
        dataset_name="MNIST",
        start_noise=start_noise,
        target_noise=target_noise,
        noise_step = noise_step,
        drift_type="incremental",
        random_seed=random_seed
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
        "drift_start": stream.drift_start,
        "drift_end": stream.drift_end,
        "max_chunk": stream.n_chunks,
        "mean_time": train_times.mean(),
        "mean_memory": memory.mean(),
    }

import mlflow

def mlflow_run(chunk_size, start_noise, target_noise, noise_step, window_size, random_seed, metrics):
    with mlflow.start_run(nested=True):

        mlflow.log_params({
            "chunk_size": chunk_size,
            "start_noise": start_noise,
            "target_noise": target_noise,
            "noise_step": noise_step,
            "window_size": window_size,
            "random_seed": random_seed
        })

        output = run_experiment(
            chunk_size,
            start_noise,
            target_noise,
            noise_step,
            window_size,
            random_seed,
            metrics
        )

        curves = output["metric_curves"]
        drift_start = output["drift_start"]
        drift_end = output["drift_end"]
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
                drift_start,
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
        mlflow.log_metric("drift_start", drift_start)
        mlflow.log_metric("drift_end", drift_end)
        mlflow.log_metric("mean_time", mean_time)
        mlflow.log_metric("mean_memory", mean_memory)


# HIPERPARAMETRY
from joblib import Parallel, delayed
import itertools
import mlflow

chunk_sizes = [50, 100, 150, 200, 250, 300, 350, 400, 450, 500]
start_noises = [0.0, 0.25, 0.5, 0.75, 1.0]
target_noises = [0.0, 0.25, 0.5, 0.75, 1.0]
noise_steps = [0.01,0.02, 0.03, 0.04, 0.05]
window_sizes = [20]
random_seeds = [42, 65]

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

mlflow.set_experiment("MNIST_IncrementalDrift_WindowSize_Sliding")

# 🔽 GENEROWANIE TYLKO POPRAWNYCH KOMBINACJI
param_grid = [
    (chunk_sizes, start_noises, target_noises, noise_steps, window_size, random_seed)
    for chunk_sizes, start_noises, target_noises, noise_steps, window_size, random_seed
    in itertools.product(
        chunk_sizes,
        start_noises,
        target_noises,
        noise_steps,
        window_sizes,
        random_seeds
    )
    if target_noises != start_noises
]

print(f"Liczba uruchamianych eksperymentów: {len(param_grid)}")

results = Parallel(n_jobs=-1, verbose=10)(
    delayed(mlflow_run)(
        chunk_size,
        start_noise,
        target_noise,
        noise_step,
        window_size,
        random_seed
    )
    for chunk_size, start_noise, target_noise, noise_step, window_size, random_seed in param_grid
)

df = pd.DataFrame(results)
print(df.sort_values("accuracy", ascending=False).head())