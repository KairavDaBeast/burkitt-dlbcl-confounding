# %% [markdown]
# # Figure 7: Stage 05 batch assessment
#
# Reads the saved results from 05_batch_confound_check_local.py and draws two panels:
#   A. External validation: model P(DLBCL) for every TCGA-DLBC and CTSP-DLBCL1 sample.
#      Both cohorts are DLBCL and were never seen by the model. Dashed line = 0.5 cut-off.
#   B. Lab-signature probe: balanced accuracy in each of 5 draws, NCICCR-DLBCL vs TCGA-DLBC.
#      Chance = 50%. Mean and SD use the same definition as the printed Stage 05 numbers.
#
# Run: python 05_fig7_batch_check.py   (writes fig7_batch_check.png and .pdf)

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

DATA = ""
per_sample = pd.read_csv(DATA + "stage05_per_sample.csv")
draws = pd.read_csv(DATA + "stage05_probe_draws.csv")
ext = pd.read_csv(DATA + "stage05_external_validation.csv")

# Validated two-color palette (dataviz skill, references/palette.md; checked with validate_palette.js)
INK, BLUE, ORANGE = "#52514e", "#2a78d6", "#eb6834"
plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False})

fig, (axA, axB) = plt.subplots(1, 2, figsize=(7.2, 3.6), gridspec_kw={"width_ratios": [1.1, 1]})
rng = np.random.default_rng(1)

# ---- Panel A: external validation
axA.axhline(0.5, color=INK, lw=0.8, ls="--", alpha=0.6)
axA.text(1.47, 0.53, "0.5 cut-off", ha="right", va="bottom", color=INK, fontsize=7.5)
for x, (name, col) in enumerate([("TCGA-DLBC", BLUE), ("CTSP-DLBCL1", ORANGE)]):
    d = per_sample[per_sample.cohort == name]
    jitter = rng.uniform(-0.14, 0.14, len(d))
    axA.scatter(x + jitter, d.p_dlbcl, s=14, color=col, alpha=0.6, linewidths=0)
    axA.plot([x - 0.22, x + 0.22], [d.p_dlbcl.median()] * 2, color=INK, lw=1.2)
    row = ext[ext.cohort == name].iloc[0]
    axA.text(x, 1.10, f"{row.called_DLBCL}/{row.n} called DLBCL", ha="center", color=INK, fontsize=8)
axA.set_xlim(-0.5, 1.5)
axA.set_ylim(-0.05, 1.18)
axA.set_yticks([0, 0.25, 0.5, 0.75, 1.0])
axA.set_xticks([0, 1])
axA.set_xticklabels(["TCGA-DLBC\n(n=48)", "CTSP-DLBCL1\n(n=44)"])
axA.set_ylabel("Model P(DLBCL), per sample", color=INK)
axA.set_title("A   External validation", loc="left", fontweight="bold")

# ---- Panel B: lab-signature probe
mean, sd = draws.balanced.mean(), np.std(draws.balanced, ddof=0)
axB.axhline(0.5, color=INK, lw=0.8, ls="--", alpha=0.6)
axB.text(1.05, 0.47, "chance 50%", ha="right", va="top", color=INK, fontsize=7.5)
jitter = rng.uniform(-0.12, 0.12, len(draws))
axB.scatter(jitter, draws.balanced, s=24, color=BLUE, alpha=0.8, linewidths=0)
axB.plot([-0.22, 0.22], [mean, mean], color=INK, lw=1.4)
axB.text(0.26, mean, f"{mean * 100:.1f}% ± {sd * 100:.1f}%", ha="left", va="center", color=INK, fontsize=8)
axB.set_xlim(-0.5, 1.05)
axB.set_ylim(0.4, 1.05)
axB.set_yticks([0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
axB.set_xticks([0])
axB.set_xticklabels(["NCICCR vs TCGA\n(5 draws)"])
axB.set_ylabel("Balanced accuracy", color=INK)
axB.set_title("B   Lab-signature probe", loc="left", fontweight="bold")

fig.tight_layout()
for ext_name in ("png", "pdf"):
    fig.savefig(DATA + f"fig7_batch_check.{ext_name}", dpi=300)
print("saved fig7_batch_check.png / .pdf")
