# %% [markdown]
# # Figures 1-6, regenerated from the local rebuild
#
#   Fig 1  Cohort flow (files -> patients -> genes -> validation cohorts)
#   Fig 2  Batch effect before and after quantile normalization, by study
#   Fig 3  Confusion matrix, 5-fold cross-validation
#   Fig 4  Permutation null distribution with the observed accuracy
#   Fig 5  MYC and the 9 starter genes, BL vs DLBCL (full normalized matrix)
#   Fig 6  Top-20 model coefficients
#
# Inputs: the rebuild (06_rebuild_results.py outputs and the full TPM table).
# Outputs: fig1_cohort_flow ... fig6_top20_coefficients, each as .png and .pdf.

import warnings
warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

DATA = "data/"
OUT = ""
BLUE, ORANGE, INK, GREY = "#2a78d6", "#eb6834", "#52514e", "#b7b5ac"   # validated 2-colour palette
plt.rcParams.update({"font.size": 8.5, "axes.spines.top": False, "axes.spines.right": False,
                     "savefig.dpi": 300, "pdf.fonttype": 42})


def save(fig, stem):
    for ext in ("png", "pdf"):
        fig.savefig(OUT + f"{stem}.{ext}", dpi=300, bbox_inches="tight")
    plt.close(fig)
    print("saved", stem)


# %% Load the rebuild (same preprocessing as 06_rebuild_results.py)
# read_pickle is safe here: the file is written by 05_download_full_tpm.py on this PC from GDC data.
full = pd.read_pickle(DATA + "full/gdc_tpm_all.pkl")
full = full[full.index.isin(pd.read_csv("data/gdc_tpm.csv.gz", index_col=0, usecols=[0]).index)]  # Stage 09 switch (2026-11-20): protein-coding genes only
proj = pd.read_csv(DATA + "full/samples_all.csv", index_col="sample")["project"].reindex(full.columns)
bl_cols = full.columns[proj == "CGCI-BLGSP"]
nc_cols = full.columns[proj == "NCICCR-DLBCL"]
train_raw = full[list(bl_cols) + list(nc_cols)].dropna(axis=0, how="any")
log_before = np.log2(train_raw + 1)


def quantile_normalize(df):
    reference = np.sort(df.values, axis=0).mean(axis=1)
    ranks = df.rank(method="average", axis=0).values
    out = np.empty_like(df.values, dtype=float)
    pos = np.arange(1, len(reference) + 1)
    for c in range(df.shape[1]):
        out[:, c] = np.interp(ranks[:, c], pos, reference)
    return pd.DataFrame(out, index=df.index, columns=df.columns)


log_after = quantile_normalize(log_before)
is_bl = log_after.columns.isin(bl_cols)

# %% Fig 1: cohort flow
fig, ax = plt.subplots(figsize=(7.2, 5.2))
ax.set_xlim(0, 1)
ax.set_ylim(0, 1)
ax.axis("off")


def box(x, y, w, h, text, fc="white", ec=INK):
    ax.add_patch(FancyBboxPatch((x - w / 2, y - h / 2), w, h, boxstyle="round,pad=0.01",
                                fc=fc, ec=ec, lw=0.8))
    ax.text(x, y, text, ha="center", va="center", fontsize=8, color=INK)


def arrow(x1, y1, x2, y2):
    ax.annotate("", xy=(x2, y2), xytext=(x1, y1), arrowprops=dict(arrowstyle="-|>", color=INK, lw=0.8))


box(0.27, 0.87, 0.36, 0.13, "Burkitt lymphoma (BL)\nCGCI-BLGSP: 351 files, 283 patients\n254 with a tumour sample", fc="#e6f0fc", ec=BLUE)
box(0.73, 0.87, 0.36, 0.13, "DLBCL\nNCICCR-DLBCL: 481 files\n481 patients", fc="#fdeee8", ec=ORANGE)
arrow(0.27, 0.805, 0.40, 0.68)
arrow(0.73, 0.805, 0.60, 0.68)
box(0.5, 0.62, 0.44, 0.12, "Training cohort: 735 patients\n(254 BL, 481 DLBCL)")
arrow(0.5, 0.56, 0.5, 0.50)
box(0.5, 0.44, 0.44, 0.12, "Genes: 19,962 protein-coding\n(non-coding genes excluded)")
arrow(0.5, 0.38, 0.5, 0.32)
box(0.5, 0.26, 0.44, 0.11, "Classifier: top-500 variance genes\n5-fold CV, permutation test (200)")
box(0.16, 0.09, 0.28, 0.16, "External validation\n(never used in training)\nTCGA-DLBC 48\nCTSP-DLBCL1 44", fc="#f4f3f0")
box(0.50, 0.09, 0.28, 0.16, "Lab-signature probe\nNCICCR vs TCGA-DLBC", fc="#f4f3f0")
box(0.84, 0.09, 0.28, 0.16, "Same-lab check\nGSE4475 (microarray)\n37 BL, 166 DLBCL", fc="#f4f3f0")
save(fig, "fig1_cohort_flow")

# %% Fig 2: batch effect before and after normalization
rng = np.random.default_rng(0)


def pooled(df, mask, n=300_000):
    vals = df.values[:, mask].ravel()
    return rng.choice(vals, min(n, vals.size), replace=False)


fig, (axA, axB) = plt.subplots(1, 2, figsize=(7.2, 3.0), sharey=True)
for ax, df, title in [(axA, log_before, "A   Before normalization"), (axB, log_after, "B   After quantile normalization")]:
    bins = np.linspace(-1, 17, 120)
    for mask, col, lab in [(is_bl, BLUE, "BL (CGCI-BLGSP)"), (~is_bl, ORANGE, "DLBCL (NCICCR)")]:
        ax.hist(pooled(df, mask), bins=bins, density=True, histtype="step", color=col, lw=1.4, label=lab)
    ax.set_title(title, loc="left", fontweight="bold")
    ax.set_xlabel("log2(TPM + 1)", color=INK)
axA.set_ylabel("Density", color=INK)
axB.legend(frameon=False, loc="upper right", fontsize=7.5)
fig.tight_layout()
save(fig, "fig2_batch_before_after")

# %% Fig 3: confusion matrix
cm = pd.read_csv(OUT + "rebuild_confusion_nested.csv", index_col=0)
fig, ax = plt.subplots(figsize=(3.6, 3.2))
im = ax.imshow(cm.values, cmap="Blues")
labels = ["BL", "DLBCL"]
ax.set_xticks([0, 1])
ax.set_yticks([0, 1])
ax.set_xticklabels(["Predicted BL", "Predicted DLBCL"])
ax.set_yticklabels(["True BL", "True DLBCL"])
for i in range(2):
    for j in range(2):
        v = int(cm.values[i, j])
        ax.text(j, i, f"{v}", ha="center", va="center", color="white" if v > cm.values.max() / 2 else INK, fontsize=11)
ax.set_title("5-fold CV, n = 764", loc="left", fontweight="bold")
ax.spines[:].set_visible(False)
save(fig, "fig3_confusion_matrix")

# %% Fig 4: permutation null
null = pd.read_csv(OUT + "rebuild_permutation_null_nested.csv")["null_fold_mean_accuracy"].values
summ = pd.read_csv(OUT + "rebuild_summary_nested.csv", index_col=0)["value"]
observed, p = float(summ["observed_fold_mean"]), float(summ["p_value"])
fig, ax = plt.subplots(figsize=(4.8, 3.2))
ax.hist(null * 100, bins=24, color=BLUE, alpha=0.55, edgecolor="white", lw=0.4)
ax.axvline(observed * 100, color=ORANGE, lw=1.6)
ax.text(observed * 100 - 0.6, ax.get_ylim()[1] * 0.9, f"observed\n{observed * 100:.2f}%", ha="right", va="top",
        color=INK, fontsize=8)
ax.set_xlabel("Fold-mean accuracy, shuffled labels (%)", color=INK)
ax.set_ylabel("Shuffles (of 200)", color=INK)
ax.set_title(f"Permutation test: p = {p:.4f} (floor)", loc="left", fontweight="bold")
ax.set_xlim(40, 102)
fig.tight_layout()
save(fig, "fig4_permutation_null")

# %% Fig 5: MYC and starter genes
myc = pd.read_csv(OUT + "rebuild_myc_starter_genes.csv")
myc = myc[myc.get("gene_id").notna()] if "gene_id" in myc else myc
fig, axes = plt.subplots(2, 5, figsize=(7.4, 4.6))
for ax, (_, row) in zip(axes.ravel(), myc.iterrows()):
    vals_bl = log_after.loc[row.gene_id, is_bl].values
    vals_dl = log_after.loc[row.gene_id, ~is_bl].values
    bp = ax.boxplot([vals_bl, vals_dl], widths=0.5, patch_artist=True, showfliers=False)
    for patch, col in zip(bp["boxes"], [BLUE, ORANGE]):
        patch.set_facecolor(col)
        patch.set_alpha(0.35)
        patch.set_edgecolor(INK)
    for med in bp["medians"]:
        med.set_color(INK)
    ax.set_xticks([1, 2])
    ax.set_xticklabels(["BL", "DLBCL"], fontsize=7)
    pval = "p<0.0001" if row.welch_p < 0.0001 else f"p={row.welch_p:.2f}"
    ax.set_title(f"{row.gene}\n{pval}", fontsize=8, color=INK)
axes[0, 0].set_ylabel("log2 expression", color=INK)
axes[1, 0].set_ylabel("log2 expression", color=INK)
fig.tight_layout()
save(fig, "fig5_myc_starter_genes")

# %% Fig 6: top-20 model coefficients
top = pd.read_csv(OUT + "stage05_top20_named.csv", index_col=0)
top = top.sort_values("coef")
names = top["gene_name"].fillna(top.index.to_series()).astype(str)
fig, ax = plt.subplots(figsize=(4.8, 4.4))
colors = [ORANGE if c > 0 else BLUE for c in top["coef"]]
ax.barh(range(len(top)), top["coef"], color=colors, alpha=0.85)
ax.set_yticks(range(len(top)))
ax.set_yticklabels(names, fontsize=7.5)
ax.axvline(0, color=INK, lw=0.8)
ax.set_xlabel("Logistic regression coefficient", color=INK)
ax.text(0.99, 0.02, "toward DLBCL ->", transform=ax.transAxes, ha="right", color=ORANGE, fontsize=7.5)
ax.text(0.01, 0.02, "<- toward BL", transform=ax.transAxes, ha="left", color=BLUE, fontsize=7.5)
ax.set_title("Top-20 genes by |coefficient|", loc="left", fontweight="bold")
fig.tight_layout()
save(fig, "fig6_top20_coefficients")
