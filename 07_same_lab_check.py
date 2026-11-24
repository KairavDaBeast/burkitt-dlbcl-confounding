# %% [markdown]
# # Stage 07: Same-lab check (is the BL vs DLBCL signal biology or lab?)
#
# Problem: in our RNA-seq data every BL sample came from one study (CGCI-BLGSP) and every
# DLBCL sample from another (NCICCR-DLBCL), so disease and lab are perfectly tangled.
#
# Test: GSE4475 (Hummel 2006, NEJM) measured BL and DLBCL in ONE lab network on ONE
# microarray platform, so its BL vs DLBCL differences cannot come from a lab difference.
# If our RNA-seq gene differences line up with GSE4475's, they are mostly biology.
#
# Control: NCICCR-DLBCL vs TCGA-DLBC is a pure lab difference (same disease, two labs).
# It should NOT line up with GSE4475. If it did, the test would be meaningless.
#
# We compare *differences* (BL mean minus DLBCL mean, log2 scale), not raw values, so the
# RNA-seq vs microarray scale mismatch cancels out.
#
# Run: python 07_download_gdc_tpm.py (once), then python 07_same_lab_check.py

# %% Step 1. Load the RNA-seq cohort (log2 TPM+1), one value per gene symbol.
import gzip
import numpy as np
import pandas as pd
from scipy import stats

DATA = "data/"
rng = np.random.default_rng(0)

tpm = pd.read_csv(DATA + "gdc_tpm.csv.gz", index_col=0)
samples = pd.read_csv(DATA + "gdc_samples.csv", index_col="sample")["project"]
rna = np.log2(tpm.drop(columns="gene_name").groupby(tpm["gene_name"]).mean() + 1)
bl_r = rna.columns[samples.reindex(rna.columns) == "CGCI-BLGSP"]
dl_r = rna.columns[samples.reindex(rna.columns) == "NCICCR-DLBCL"]
tc_r = rna.columns[samples.reindex(rna.columns) == "TCGA-DLBC"]
print(f"RNA-seq: {len(bl_r)} BL, {len(dl_r)} NCICCR-DLBCL, {len(tc_r)} TCGA-DLBC")

# %% Step 2. Load GSE4475 (already log2) and turn probe IDs into gene symbols.
with gzip.open(DATA + "geo/GSE4475_series_matrix.txt.gz", "rt") as f:
    lines = f.read().splitlines()
diag = next(l for l in lines if "Morphologic.diagnosis" in l).split("\t")[1:]
diag = [d.strip('"').split(" : ", 1)[1] for d in diag]
start = lines.index("!series_matrix_table_begin") + 1
end = lines.index("!series_matrix_table_end")
arr = pd.DataFrame([l.split("\t") for l in lines[start + 1:end]]).set_index(0)
arr.columns = [c.strip('"') for c in lines[start].split("\t")[1:]]
arr.index = arr.index.str.strip('"')
arr = arr.apply(pd.to_numeric, errors="coerce")

# The .annot file has 27 lines of notes before the table header (checked by hand).
annot = pd.read_csv(DATA + "geo/GPL96.annot.gz", sep="\t", skiprows=27, usecols=["ID", "Gene symbol"]).dropna()
annot = annot[~annot["Gene symbol"].str.contains("///")].set_index("ID")["Gene symbol"]
arr = arr.loc[arr.index.intersection(annot.index)]
# Several probes per gene: keep the probe with the highest average signal.
arr["gene"] = annot.reindex(arr.index)
arr["mean"] = arr.drop(columns="gene").mean(axis=1)
arr = arr.sort_values("mean", ascending=False).drop_duplicates("gene").set_index("gene").drop(columns="mean")

is_bl = np.array([d in ("BL", "atypical BL", "BL-leukemia") for d in diag])
is_dl = np.array([d == "DLBCL" for d in diag])
bl_a, dl_a = arr.columns[is_bl], arr.columns[is_dl]
print(f"GSE4475: {len(bl_a)} BL (morphology), {len(dl_a)} DLBCL")

# %% Step 3. Keep genes measured in both and actually expressed in the RNA-seq data.
genes = rna.index.intersection(arr.index)
genes = genes[rna.loc[genes, list(bl_r) + list(dl_r)].mean(axis=1) >= 1]
print(f"Genes compared: {len(genes)}")


def diff(df, a, b):
    """BL-minus-DLBCL difference per gene, log2 scale."""
    return df.loc[genes, a].mean(axis=1) - df.loc[genes, b].mean(axis=1)


d_rna = diff(rna, bl_r, dl_r)          # disease + lab (tangled)
d_arr = diff(arr, bl_a, dl_a)          # disease only (same lab)
d_lab = diff(rna, dl_r, tc_r)          # lab only (same disease)
p_arr = pd.Series(stats.ttest_ind(arr.loc[genes, bl_a], arr.loc[genes, dl_a], axis=1, equal_var=False).pvalue, index=genes)
fdr_arr = pd.Series(stats.false_discovery_control(p_arr.fillna(1)), index=genes)

# %% Step 4. Do the differences line up?
rho = stats.spearmanr(d_rna, d_arr).statistic
rho_lab = stats.spearmanr(d_lab, d_arr).statistic


def boot_rho(n=200):
    """Resample patients (not genes, which are correlated) to get a 95% interval."""
    out = []
    for _ in range(n):
        pick = lambda cols: rng.choice(cols, len(cols))
        out.append(stats.spearmanr(diff(rna, pick(bl_r), pick(dl_r)), diff(arr, pick(bl_a), pick(dl_a))).statistic)
    return np.percentile(out, [2.5, 97.5])


ci = boot_rho()
strong = (fdr_arr < 0.01) & (d_arr.abs() >= 1)          # clear same-lab disease genes
agree = (np.sign(d_rna[strong]) == np.sign(d_arr[strong])).mean()
fit = stats.linregress(d_arr, d_rna)
print(f"\nSpearman rho, RNA-seq BL-vs-DLBCL vs same-lab GSE4475: {rho:.3f} (95% CI {ci[0]:.3f} to {ci[1]:.3f})")
print(f"Control, lab-only difference vs GSE4475:              {rho_lab:.3f}")
print(f"Clear same-lab disease genes (FDR<0.01, |diff|>=1): {int(strong.sum())}; same direction in RNA-seq: {agree:.1%}")
print(f"Share of RNA-seq gene differences explained by same-lab biology (R^2): {fit.rvalue**2:.2f}")

# %% Step 5. Do the model's top genes hold up in the same-lab data?
# Coefficient > 0 pushes toward DLBCL, so we expect BL-minus-DLBCL < 0 (and the reverse).
# FINAL model top-20 (rebuild, 2026-11-08), from stage05_top20_named.csv. This replaces the earlier
# exploratory list that was hard-coded here before (see Author note A in the draft).
final_top20 = pd.read_csv("stage05_top20_named.csv")
top = {r.gene_name: round(float(r.coef), 4) for r in final_top20.itertuples()}
top["MYC"] = None
rows = []
for g, coef in top.items():
    if g not in arr.index:
        rows.append({"gene": g, "note": "not on GSE4475 array"})
        continue
    t = stats.ttest_ind(arr.loc[g, bl_a], arr.loc[g, dl_a], equal_var=False)
    da = arr.loc[g, bl_a].mean() - arr.loc[g, dl_a].mean()
    rows.append({"gene": g, "model_coef": coef, "diff_same_lab": round(da, 2), "p_same_lab": float(f"{t.pvalue:.2g}"),
                 "diff_rnaseq": round(d_rna.get(g, np.nan), 2),
                 "holds_up": None if coef is None else bool(np.sign(da) == -np.sign(coef) and t.pvalue < 0.05)})
top_df = pd.DataFrame(rows)
print("\n" + top_df.to_string(index=False))

# %% Step 6. Save results for the paper.
pd.DataFrame({"diff_rnaseq": d_rna, "diff_same_lab": d_arr, "fdr_same_lab": fdr_arr, "diff_lab_only": d_lab}) \
    .round(4).to_csv("stage07_gene_differences.csv", index_label="gene")
top_df.to_csv("stage07_top_genes.csv", index=False)
pd.Series({"genes": len(genes), "rho": rho, "ci_low": ci[0], "ci_high": ci[1], "rho_lab_only": rho_lab,
           "strong_genes": int(strong.sum()), "direction_agreement": agree, "r2": fit.rvalue ** 2,
           "n_bl_rnaseq": len(bl_r), "n_dlbcl_rnaseq": len(dl_r), "n_tcga": len(tc_r),
           "n_bl_array": len(bl_a), "n_dlbcl_array": len(dl_a)}).to_csv("stage07_summary.csv", header=["value"])

# %% Step 7. Figure 8: one dot per gene. Left = our data vs same-lab data. Right = lab-only control.
import matplotlib.pyplot as plt

plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False})
INK, GENE, MODEL = "#52514e", "#2a78d6", "#eb6834"   # validated palette (dataviz skill)
model_genes = [g for g in top if top[g] is not None and g in genes]
fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.6), sharey=True)
panels = [(d_rna, rho, "BL − DLBCL, our RNA-seq\n(disease + lab tangled)"),
          (d_lab, rho_lab, "NCICCR − TCGA DLBCL\n(lab only, same disease)")]
for ax, (x, r, xlabel), tag in zip(axes, panels, "AB"):
    ax.axhline(0, color=INK, lw=0.5, alpha=0.4)
    ax.axvline(0, color=INK, lw=0.5, alpha=0.4)
    ax.scatter(x, d_arr, s=3, color=GENE, alpha=0.25, linewidths=0, label="All genes")
    if tag == "A":
        ax.scatter(x[model_genes], d_arr[model_genes], s=18, color=MODEL, edgecolors="white", linewidths=0.6,
                   label="Model top-20 genes")
        for g in sorted(model_genes, key=lambda g: -abs(x[g]))[:5]:   # label the 5 farthest out, so labels do not overlap
            if g in model_genes:
                ax.annotate(g, (x[g], d_arr[g]), xytext=(4, 2), textcoords="offset points", fontsize=7, color=INK)
        ax.legend(frameon=False, loc="upper left", fontsize=7, markerscale=1.5)
    ax.text(0.98, 0.04, f"Spearman ρ = {r:.2f}", transform=ax.transAxes, ha="right", color=INK)
    ax.set_xlabel(xlabel, color=INK)
    ax.set_title(tag, loc="left", fontweight="bold")
axes[0].set_ylabel("BL − DLBCL, GSE4475\n(same lab, log2)", color=INK)
fig.tight_layout()
for ext in ("png", "pdf"):
    fig.savefig(f"fig8_same_lab_check.{ext}", dpi=300)
print("saved fig8_same_lab_check.png/.pdf")
