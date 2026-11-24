"""Figure 9: transfer of RNA-seq-trained models to GSE4475 (AUC for BL vs DLBCL).
Reads transfer_gse4475_results.csv written by biology_gene_set_model.py.
Run: python fig9_transfer.py  (from the project folder)
"""
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

t = pd.read_csv("transfer_gse4475_results.csv")
groups = ["morphology, all", "morphology, adults (age >= 18)", "molecular, all", "molecular, adults (age >= 18)"]
labels = ["Morphology\nall", "Morphology\nadults", "Molecular\nall", "Molecular\nadults"]
models = [("top-500 variance", "Top-500 variance genes", "#2a78d6"),
          ("MYC targets", "MYC targets (223)", "#eb6834"),
          ("matched random sets (mean of 100)", "Matched random sets (mean, 5th–95th pct)", "#9a9893"),
          ("lab-only model, top-500 variance (orientation-free)", "Lab-only model", "#52514e")]
plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False})
fig, ax = plt.subplots(figsize=(7.2, 3.6))
w = 0.2
x = np.arange(len(groups))
for i, (key, name, col) in enumerate(models):
    sub = t[t["model"] == key].set_index("labels").reindex(groups)
    auc = sub["auc"].values
    err = None
    if sub["ci_low"].notna().all():
        err = np.vstack([auc - sub["ci_low"].values, sub["ci_high"].values - auc])
    ax.bar(x + (i - 1.5) * w, auc, w, color=col, label=name, yerr=err, capsize=2, error_kw={"lw": 0.8})
ax.axhline(0.5, color="#52514e", lw=0.8, ls="--")
ax.set_xticks(x)
ax.set_xticklabels(labels)
ax.set_ylim(0.4, 1.02)
ax.set_ylabel("AUC on GSE4475 (BL vs DLBCL)\ndashed line = chance")
ax.legend(frameon=False, fontsize=7, loc="upper center", bbox_to_anchor=(0.5, -0.2), ncol=4)
fig.tight_layout()
for ext in ("png", "pdf"):
    fig.savefig(f"fig9_transfer.{ext}", dpi=300)
print("saved fig9_transfer.png/.pdf")
