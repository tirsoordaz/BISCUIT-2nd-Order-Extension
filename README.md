# BISCUIT: 2nd-Order Temporal Extension

This repository is an architectural extension of the **[BISCUIT (phlippe/BISCUIT)](https://github.com/phlippe/BISCUIT)** framework for causal representation learning. 

While the original BISCUIT model successfully disentangles causal variables under a strict 1st-order Markov assumption, this project modifies the architecture to incorporate **2nd-order temporal dependencies**, proving that causal identifiability is maintained even when the model must condition on deeper historical contexts.

---

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

*(Note: You can view the full process and the plotted $R^2$ matrices in the included `biscuit_tutorial_2nd_order.ipynb` notebook!)*

---

## Repository Structure

Instead of duplicating the entire original BISCUIT codebase, this repository only contains our specific modifications and evaluation results:

```text
├── README.md
├── biscuit_tutorial_2nd_order.ipynb   # Full interactive tutorial and heatmap plotting
├── results/
│   ├── baseline_r2_matrix.npz         # Extracted 1st-order matrix
│   └── 2nd_order_r2_matrix.npz        # Extracted 2nd-order matrix
└── custom_code/
    ├── biscuit_nf_2nd_order.py        # The custom 2nd-order PyTorch Lightning module
    ├── train_nf_2nd_order.py          # Modified training script to load the custom class
    └── run_nf_2nd_order.sh            # Bash script to execute the model with seq_len=3
