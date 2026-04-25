import os
import pandas as pd

MLRUNS_PATH = "mlruns/0"
OUTPUT_FILE = "mean_time_tables_sw.xlsx"

records = []

for run_id in os.listdir(MLRUNS_PATH):

    run_path = os.path.join(MLRUNS_PATH, run_id)

    if not os.path.isdir(run_path):
        continue

    try:
        params_path = os.path.join(run_path, "params")
        metrics_path = os.path.join(run_path, "metrics")

        required_params = [
            "window_size",
            "epochs",
            "learning_rate",
            "random_seed"
        ]

        params = {}

        for p in required_params:
            p_file = os.path.join(params_path, p)

            if not os.path.exists(p_file):
                raise ValueError("missing param")

            with open(p_file) as f:
                params[p] = f.read().strip()

        accuracy_file = os.path.join(metrics_path, "mean_time")

        if not os.path.exists(accuracy_file):
            continue

        with open(accuracy_file) as f:
            lines = f.readlines()
            last = lines[-1].strip().split()
            mean_time = float(last[1])

        record = {
            "window_size": int(params["window_size"]),
            "epochs": float(params["epochs"]),
            "learning_rate": float(params["learning_rate"]),
            "random_seed": int(params["random_seed"]),
            "mean_time": mean_time
        }

        records.append(record)

    except Exception:
        continue


df = pd.DataFrame(records)

print("Runów poprawnych:", len(df))

# ======================
# UŚREDNIENIE PO SEED
# ======================

df_mean = (
    df
    .groupby(["window_size", "epochs", "learning_rate"])
    .mean_time
    .mean()
    .reset_index()
)

# ======================
# TWORZENIE TABEL
# ======================

tables = {}

for lr in sorted(df_mean.learning_rate.unique()):

    pivot = (
        df_mean[df_mean.learning_rate == lr]
        .pivot(
            index="window_size",
            columns="epochs",
            values="mean_time"
        )
        .sort_index()
        .sort_index(axis=1)
    )

    tables[lr] = pivot

# ======================
# ZAPIS DO EXCEL
# ======================

with pd.ExcelWriter(OUTPUT_FILE, engine="xlsxwriter") as writer:

    workbook = writer.book
    bold_format = workbook.add_format({'bold': True})

    for lr, table in tables.items():

        sheet_name = f"lr_{lr}"
        table.to_excel(writer, sheet_name=sheet_name)

        worksheet = writer.sheets[sheet_name]

        # maksimum w tabeli
        max_val = table.max().max()

        for r in range(table.shape[0]):
            for c in range(table.shape[1]):

                val = table.iloc[r, c]

                if pd.isna(val):
                    continue

                row = r + 1
                col = c + 1

                if val == max_val:
                    worksheet.write(row, col, val, bold_format)

print("Zapisano:", OUTPUT_FILE)