import os
import json
import itertools

# ==============================
# 1. PARAM GRID
# ==============================

"""chunk_sizes = [200]
noise_percents = [0.0]
new_noises = [0.0]
window_sizes = [4, 8, 12, 16, 20, 24, 28, 32, 36, 40]
random_seeds = [42, 65, 88]
ulrealing_rates = [0.0001, 0.0002, 0.0003, 0.0004, 0.0005,
                   0.0006, 0.0007, 0.0008, 0.0009, 0.001]
learning_rates = [0.0001, 0.0002, 0.0003, 0.0004, 0.0005,
                  0.0006, 0.0007, 0.0008, 0.0009, 0.001]

param_grid = [
    (chunk_size, noise_percent, delta_noise, window_size, ulr, lr, random_seed)
    for chunk_size, noise_percent, delta_noise, window_size, ulr, lr, random_seed
    in itertools.product(
        chunk_sizes,
        noise_percents,
        new_noises,
        window_sizes,
        ulrealing_rates,
        learning_rates,
        random_seeds
    )
]"""

import os
import json
import itertools

# ==============================
# 1. PARAM GRID
# ==============================

chunk_sizes = [200]
noise_percents = [0.0]
new_noises = [0.0]
window_sizes = [4, 8, 12, 16, 20, 24, 28, 32, 36, 40]
random_seeds = [42, 65, 88]
learning_rates = [0.0001, 0.0002, 0.0003, 0.0004, 0.0005, 0.0006, 0.0007, 0.0008, 0.0009, 0.001,
                  0.0011, 0.0012, 0.0013, 0.0014, 0.0015, 0.0016, 0.0017, 0.0018, 0.0019, 0.0020,
                  0.0021, 0.0022, 0.0023, 0.0024, 0.0025]
unlearning_rates = [0.0001, 0.0002, 0.0003, 0.0004, 0.0005, 0.0006, 0.0007, 0.0008, 0.0009, 0.001,
                  0.0011, 0.0012, 0.0013, 0.0014, 0.0015, 0.0016, 0.0017, 0.0018, 0.0019, 0.0020,
                  0.0021, 0.0022, 0.0023, 0.0024, 0.0025]

param_grid = [
    (chunk_size, noise_percent, delta_noise, window_size, unlearning_rate, lr, random_seed)
    for chunk_size, noise_percent, delta_noise, window_size, unlearning_rate, lr, random_seed
    in itertools.product(
        chunk_sizes,
        noise_percents,
        new_noises,
        window_sizes,
        unlearning_rates,
        learning_rates,
        random_seeds,
    )
]

print(f"Total combinations: {len(param_grid)}")


# ==============================
# 2. CZYTANIE PARAMS
# ==============================

MLRUNS_DIR = "C:/Users/maciek/Documents/GitHub/UnlearningDataStreamPlayground/mlruns2/0"

def read_params(run_path):
    params_path = os.path.join(run_path, "params")

    if not os.path.exists(params_path):
        return None

    params = {}

    for p in os.listdir(params_path):
        try:
            with open(os.path.join(params_path, p)) as f:
                params[p] = f.read().strip()
        except:
            continue

    return params


# ==============================
# 3. FILTR NA ACCURACY
# ==============================

def has_accuracy(run_path):
    metrics_path = os.path.join(run_path, "metrics", "accuracy")

    if not os.path.exists(metrics_path):
        return False

    # opcjonalnie: upewnij się, że plik nie jest pusty
    return os.path.getsize(metrics_path) > 0


# ==============================
# 4. RUNY -> TUPLE
# ==============================

existing_tuples = set()

for run_id in os.listdir(MLRUNS_DIR):
    run_path = os.path.join(MLRUNS_DIR, run_id)

    if not os.path.isdir(run_path):
        continue

    params = read_params(run_path)

    if not params:
        continue

    # 🔥 KLUCZOWY FILTR
    if not has_accuracy(run_path):
        continue

    try:
        t = (
            int(params["chunk_size"]),
            float(params.get("noise_percent", 0.0)),
            float(params.get("delta_noise", 0.0)),
            int(params["window_size"]),
            float(params["unlearning_rate"]),
            float(params["learning_rate"]),
            int(params["random_seed"]),
        )

        existing_tuples.add(t)

    except Exception:
        continue

print(f"Parsed valid runs (with accuracy): {len(existing_tuples)}")


# ==============================
# 5. BRAKUJĄCE EKSPERYMENTY
# ==============================

missing = [p for p in param_grid if p not in existing_tuples]

print(f"Missing experiments: {len(missing)}")


# ==============================
# 6. ZAPIS
# ==============================

with open("missing_runsUN.txt", "w") as f:
    json.dump(missing, f)

print("Saved missing_runsUN.txt")

"""# ==============================
# 6. WCZYTANIE
# ==============================

with open("missing_runs.txt") as f:
    loaded_param_grid = json.load(f)

loaded_param_grid = [tuple(p) for p in loaded_param_grid]

print(f"Reloaded: {len(loaded_param_grid)}")"""
# ==============================
# 7. (OPCJONALNIE) AUTO-RESUME
# ==============================

# param_grid = [p for p in param_grid if p not in existing_tuples]

# from joblib import Parallel, delayed
# results = Parallel(n_jobs=-1)(
#     delayed(mlflow_run)(
#         chunk_size,
#         noise_percent,
#         delta_noise,
#         window_size,
#         random_seed,
#         ulr,
#         lr,
#         metrics,
#     )
#     for chunk_size, noise_percent, delta_noise, window_size, ulr, lr, random_seed in param_grid
# )