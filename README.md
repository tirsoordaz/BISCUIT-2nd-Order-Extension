# Causal Representation Learning Extensions: CITRIS & BISCUIT

This repository contains the implementation, experiments, and report for our project on **Causal Representation Learning (CRL)**.

The project extends two state-of-the-art causal representation learning frameworks:

- **CITRIS** by introducing a configurable **Time Step Extension**, allowing the transition prior to condition on multiple previous latent states.
- **BISCUIT** by implementing a **2nd-Order Temporal Extension**, enabling the transition model to use two previous latent states instead of one.

The repository also contains the accompanying notebook, evaluation scripts, and experimental results used in the final report.

---

# Repository Contents

```
.
├── CRL.ipynb                      # Final project notebook
├── README.md
├── citris-requirements.txt        # Python environment used for the CITRIS experiments
├── CML.pptx                       # Final project presentation
│
├── extensions/
│   ├── BISCUIT/
│   │   ├── biscuit_nf_2nd_order.py
│   │   ├── train_nf_2nd_order.py
│   │   └── run_nf_2nd_order.sh
│   │
│   └── CITRIS/
│       ├── experiments/
│       └── models/
│
└── results/
    ├── baseline_r2_matrix.npz
    ├── 2nd_order_r2_matrix.npz
    └── results-citris/
```

---

# CITRIS Time Step Extension

The original CITRIS transition prior predicts the next latent state using only the previous latent state:

\[
p(z_{t+1}\mid z_t,I_{t+1})
\]

Our extension generalizes this formulation by introducing a configurable history length:

\[
p(z_{t+1}\mid z_{t-k+1},\ldots,z_t,I_{t+1})
\]

where `history_length = k`.

The implementation modifies the transition prior and training pipeline to support arbitrary temporal context while remaining fully compatible with the original implementation when

```text
history_length = 1
```

Experiments were conducted on the **Interventional Pong** dataset using history lengths from **1 to 4**.

Evaluation follows the original CITRIS protocol using:

- Triplet Prediction
- R² Correlation
- Spearman Rank Correlation

The repository includes:

- implementation files
- evaluation script (`analyze_results.py`)
- generated figures
- evaluation matrices
- summary tables

Running

```bash
python analyze_results.py
```

reproduces the tables and figures presented in the notebook.

---
# BISCUIT

## The Experiment: Moving Beyond the Markov Assumption

The standard BISCUIT Normalizing Flow models transitions by predicting the next state ($z_t$) using only the immediately preceding state ($z_{t-1}$) and an intervention/action ($a_t$):
$$P(z_t \vert{} z_{t-1}, a_t)$$

However, many real-world physical systems involve higher-order dynamics (such as velocity or acceleration) where a single prior frame is insufficient. Our goal was to test if BISCUIT's identifiability proofs hold up when we force the Transition Prior to learn from a concatenated history of two previous timesteps:
$$P(z_t \vert{} z_{t-1}, z_{t-2}, a_t)$$

By modifying the PyTorch dimensions, concatenating historical latents, and adjusting the `MultivarLinear` and `MultivarSequential` layers of the transition prior, the model successfully learned this 2nd-order dynamic.

---

## Results & Causal Disentanglement

We evaluated the model's ability to disentangle true causal variables on the **CausalWorld** dataset by computing the average diagonal $R^2$ score between the learned latents and the true causal factors.

| Architecture | Temporal Context | Average Diagonal $R^2$ |
| :--- | :--- | :--- |
| **Baseline (Original)** | 1st-Order ($t-1 \to t$) | **0.9678** |
| **Custom Extension** | 2nd-Order ($t-2, t-1 \to t$) | **0.9652** |

**Conclusion:** A score of 0.9652 demonstrates that the 2nd-order model successfully learned the causal representations without breaking the theoretical identifiability proofs, performing almost identically to the baseline.

Note: You can view the full process and the plotted $R^2$ matrices in the included `biscuit_tutorial_2nd_order.ipynb` notebook.

---

# Based on

- Lippe et al., **CITRIS** (https://github.com/phlippe/CITRIS)
- Lippe et al., **BISCUIT** (https://github.com/phlippe/BISCUIT)

This repository contains only the code developed for the project together with the evaluation scripts and experimental results. The original implementations remain available from their respective official repositories.
