import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import linear_sum_assignment
import csv

EXPERIMENTS = {
    #"Original": "./checkpoints/CITRISVAE/CITRISVAE_12l_5b_32hid_pong/version_36",
    "history_1": "./hl_1",
    "history_2": "./hl_2",
    "history_3": "./hl_3",
    "history_4": "./hl_4",
}

RESULTS_DIR = "./results/results-citris"

os.makedirs(RESULTS_DIR, exist_ok=True)

CAUSAL_NAMES = [
    "ball-vel-dir",
    "ball-x",
    "ball-y",
    "paddle-left-y",
    "paddle-right-y",
    "score-left",
    "score-right"
]

def compute_diag_sep(matrix):
    rows, cols = linear_sum_assignment(-matrix)

    diag = matrix[rows, cols].mean()

    sep_values = []

    for r, c in zip(rows, cols):
        remaining = np.delete(matrix[r], c)
        sep_values.append(remaining.max())

    sep = np.mean(sep_values)

    return diag, sep, rows, cols

def load_results(folder):

    r2 = np.load(
        os.path.join(folder, "r2_matrix.npz")
    )

    spearman = np.load(
        os.path.join(folder, "spearman_matrix.npz")
    )

    triplet = np.load(
        os.path.join(folder, "triplet_dists.npz")
    )

    r2_final = r2["values"][-1]
    sp_final = spearman["values"][-1]

    r2_diag, r2_sep, r2_rows, r2_cols = compute_diag_sep(
        r2_final
    )

    sp_diag, sp_sep, sp_rows, sp_cols = compute_diag_sep(
        sp_final
    )

    triplet_values = {}

    for key in triplet.files:
        if key != "epochs":
            triplet_values[key] = triplet[key][-1]

    mean_ball = np.mean([
        triplet_values["ball-vel-dir"],
        triplet_values["ball-x"],
        triplet_values["ball-y"]
    ])

    return {

        "r2_diag": r2_diag,
        "r2_sep": r2_sep,

        "spearman_diag": sp_diag,
        "spearman_sep": sp_sep,

        "mean_ball_triplet": mean_ball,

        "triplet": triplet_values,

        "r2_history": r2["values"],
        "r2_epochs": r2["epochs"],

        "r2_final": r2_final,
        "r2_rows": r2_rows,
        "r2_cols": r2_cols
    }

def save_heatmap(result, name):

    matrix = result["r2_final"]

    plt.figure(figsize=(8, 5))

    plt.imshow(
        matrix,
        cmap="viridis",
        aspect="auto"
    )

    plt.colorbar(label="R²")

    plt.xticks(
        range(len(CAUSAL_NAMES)),
        CAUSAL_NAMES,
        rotation=45,
        ha="right"
    )

    plt.yticks(
        range(matrix.shape[0]),
        [
            f"latent {i}"
            for i in range(matrix.shape[0])
        ]
    )

    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            plt.text(
                j,
                i,
                f"{matrix[i, j]:.2f}",
                ha="center",
                va="center",
                fontsize=8
            )

    plt.title(
        f"R² Matrix - {name}"
    )

    plt.tight_layout()

    plt.savefig(
        os.path.join(RESULTS_DIR, f"{name}_r2_heatmap.png"),
        dpi=200
    )

    plt.close()

def save_learning_curve(all_results):

    plt.figure(figsize=(7,5))


    for name,result in all_results.items():

        diag_values=[]

        for matrix in result["r2_history"]:

            diag,_,_,_=compute_diag_sep(matrix)

            diag_values.append(diag)

        plt.plot(
            result["r2_epochs"],
            diag_values,
            label=name
        )

    plt.xlabel("Epoch")
    plt.ylabel("R² diagonal")
    plt.title("Disentanglement during training")

    plt.legend()
    plt.grid(True)

    plt.tight_layout()

    plt.savefig(
        os.path.join(RESULTS_DIR, "r2_training_comparison.png"),
        dpi=200
    )

    plt.close()

all_results={}

for name,folder in EXPERIMENTS.items():

    print(f"Loading {name}")

    all_results[name]=load_results(folder)

    save_heatmap(
        all_results[name],
        name
    )

print("\n")
print("="*90)

print(
    f"{'Experiment':15s}"
    f"{'Triplet':12s}"
    f"{'R2diag':12s}"
    f"{'R2sep':12s}"
    f"{'SpDiag':12s}"
    f"{'SpSep':12s}"
)

print("="*90)

for name,r in all_results.items():
    print(
        f"{name:15s}"
        f"{r['mean_ball_triplet']:<12.4f}"
        f"{r['r2_diag']:<12.4f}"
        f"{r['r2_sep']:<12.4f}"
        f"{r['spearman_diag']:<12.4f}"
        f"{r['spearman_sep']:<12.4f}"
    )

print("="*90)

with open(
    os.path.join(RESULTS_DIR, "comparison_results.csv"),
    "w",
    newline=""
) as f:
    writer=csv.writer(f)

    writer.writerow([
        "experiment",
        "mean_ball_triplet",
        "r2_diag",
        "r2_sep",
        "spearman_diag",
        "spearman_sep"
    ])

    for name,r in all_results.items():
        writer.writerow([
            name,
            r["mean_ball_triplet"],
            r["r2_diag"],
            r["r2_sep"],
            r["spearman_diag"],
            r["spearman_sep"]
        ])

save_learning_curve(all_results)

print("\nGenerated:")
print("- comparison_results.csv")
print("- r2_training_comparison.png")
print("- <experiment>_r2_heatmap.png")