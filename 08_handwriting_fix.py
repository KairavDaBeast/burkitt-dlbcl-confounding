"""Stage 08: can the lab "handwriting" be removed from the RNA-seq data?

In the GDC data every BL sample comes from one study and every DLBCL sample from others, so lab and
disease are the same split. Nothing computed from that data alone can separate them. Each method below
brings in outside information and is judged on GSE4475 (one lab network, both diseases, never trained on).

  M1a  Lab directions: the mean differences between the three DLBCL labs (NCICCR, TCGA, CTSP) are taken
       as handwriting directions and projected out of every sample.
  M1b  Lab-sensitive genes: genes that differ most between the DLBCL labs (top half by Cohen's d) are
       dropped from the model's gene pool. Values are unchanged.
  M2   RUV-4 style correction with negative-control genes. Controls = genes with no BL vs DLBCL gap in
       half A of GSE4475 (Welch p > 0.5 and |diff| < 0.2). Handwriting patterns = top-k singular vectors
       of the training residuals after removing each class mean (within-group wobble). Each sample's
       amount of each pattern is read from its control genes, then subtracted from all genes.
       M2-naive is plain RUVg (factors from controls, regressed out of every gene) for comparison.
  M3   k-TSP: rules of the form "gene i lower than gene j in this sample". No cross-dataset scaling.

Tests, averaged over 10 random halves of GSE4475 (controls for M2 come from half A, tests use half B):
  transfer AUC (morphology labels and molecular labels), gene-level agreement with same-lab
  BL-minus-DLBCL differences (Spearman rho, control genes excluded), MYC difference.
Lab probe: NCICCR vs TCGA balanced accuracy after correction (chance 0.50). For M1 the directions/genes
are learned from NCICCR vs CTSP only, so TCGA stays unseen; otherwise the probe would be circular.

Run: python 08_handwriting_fix.py   (from the project folder; about 5-10 minutes)
"""
import gzip
import warnings
warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler

BASE = "data/"
OUT = ""
N_TOP, N_SPLITS, K_RUV, K_TSP = 500, 10, 5, 9      # fixed before looking at any result

# %% Step 1. RNA-seq: protein-coding genes by symbol, log2(TPM+1), all four cohorts.
# read_pickle is safe here: the file is written by 05_download_full_tpm.py on this PC from GDC data.
full = pd.read_pickle(BASE + "full/gdc_tpm_all.pkl")
proj = pd.read_csv(BASE + "full/samples_all.csv", index_col="sample")["project"].reindex(full.columns)
names = pd.read_csv(BASE + "gdc_tpm.csv.gz", index_col=0, usecols=[0, 1])["gene_name"]
rna = np.log2(full.loc[names.index].groupby(names.values).mean() + 1)
del full
BL, NC, TC, CT = (list(rna.columns[(proj == p).values])
                  for p in ["CGCI-BLGSP", "NCICCR-DLBCL", "TCGA-DLBC", "CTSP-DLBCL1"])
TR = BL + NC
nTR = len(TR)
rna = rna[TR + TC + CT]
y_tr = np.array(["BL"] * len(BL) + ["DLBCL"] * len(NC), dtype=object)
print(f"RNA-seq: BL {len(BL)} | NCICCR {len(NC)} | TCGA {len(TC)} | CTSP {len(CT)}")

# %% Step 2. GSE4475 (same steps as biology_gene_set_model.py).
with gzip.open(BASE + "geo/GSE4475_series_matrix.txt.gz", "rt") as f:
    lines = f.read().splitlines()


def field(key):
    l = next(l for l in lines if l.startswith("!Sample_characteristics") and key in l)
    return [d.strip('"').split(" : ", 1)[1].strip() for d in l.split("\t")[1:]]


morph = np.array(field("Morphologic.diagnosis"))
molec = np.array(field("Molecular.Diagnosis"))
start = lines.index("!series_matrix_table_begin") + 1
end = lines.index("!series_matrix_table_end")
arr = pd.DataFrame([l.split("\t") for l in lines[start + 1:end]]).set_index(0)
arr.columns = [c.strip('"') for c in lines[start].split("\t")[1:]]
arr.index = arr.index.str.strip('"')
arr = arr.apply(pd.to_numeric, errors="coerce")
annot = pd.read_csv(BASE + "geo/GPL96.annot.gz", sep="\t", skiprows=27, usecols=["ID", "Gene symbol"]).dropna()
annot = annot[~annot["Gene symbol"].str.contains("///")].set_index("ID")["Gene symbol"]
arr = arr.loc[arr.index.intersection(annot.index)]
arr["gene"] = annot.reindex(arr.index)
arr["mean"] = arr.drop(columns="gene").mean(axis=1)
arr = arr.sort_values("mean", ascending=False).drop_duplicates("gene").set_index("gene").drop(columns="mean")

genes = rna.index.intersection(arr.index)
genes = genes[rna.loc[genes, TR].mean(axis=1) >= 1]
R = rna.loc[genes]
A = arr.loc[genes]
del rna
print(f"Genes on both platforms (expressed): {len(genes)}")

m_bl = np.isin(morph, ["BL", "atypical BL", "BL-leukemia"])
m_dl = morph == "DLBCL"
mol_bl, mol_dl = molec == "mBL", molec == "non-mBL"


def zrows(M):
    M = np.asarray(M, dtype=float)
    sd = M.std(axis=1, ddof=1, keepdims=True)
    sd[sd == 0] = 1
    return np.nan_to_num((M - M.mean(axis=1, keepdims=True)) / sd)


Az = pd.DataFrame(zrows(A.values), index=genes)
splits = []
for s in range(N_SPLITS):
    a_idx, _ = next(StratifiedKFold(2, shuffle=True, random_state=s).split(morph, 2 * m_bl + m_dl))
    inA = np.zeros(len(morph), bool)
    inA[a_idx] = True
    splits.append(inA)


# %% Step 3. Shared tests.
def transfer_lr(C, pool):
    """Train top-500-variance logistic regression on corrected BL+NCICCR, score every GSE4475 sample."""
    sub = C.loc[pool, TR]
    top = sub.var(axis=1).sort_values(ascending=False).index[:N_TOP]
    clf = LogisticRegression(max_iter=1000).fit(zrows(sub.loc[top].values).T, y_tr)
    return clf.decision_function(Az.loc[top].values.T)          # > 0 = DLBCL


def auc(dec, pos, neg, mask):
    p, n = pos & mask, neg & mask
    return roc_auc_score(np.r_[np.ones(n.sum()), np.zeros(p.sum())], np.r_[dec[n], dec[p]])


def rho_same_lab(C, inB, exclude):
    g = genes[~exclude]
    d_r = C.loc[g, BL].mean(axis=1) - C.loc[g, NC].mean(axis=1)
    d_a = A.loc[g, A.columns[m_bl & inB]].mean(axis=1) - A.loc[g, A.columns[m_dl & inB]].mean(axis=1)
    return stats.spearmanr(d_r, d_a).statistic


def gap_kept(C):
    """Size of the BL-minus-DLBCL difference vector after correction, relative to before (1 = untouched)."""
    d0 = R[BL].mean(axis=1) - R[NC].mean(axis=1)
    d1 = C[BL].mean(axis=1) - C[NC].mean(axis=1)
    return float(np.linalg.norm(d1) / np.linalg.norm(d0))


def lr_dec(Xtr, ytr, Xte):
    sel = np.argsort(-Xtr.var(axis=0))[:N_TOP]
    sc = StandardScaler().fit(Xtr[:, sel])
    clf = LogisticRegression(max_iter=1000).fit(sc.transform(Xtr[:, sel]), ytr)
    return clf.decision_function(sc.transform(Xte[:, sel]))


def probe(C, pool, fit=lr_dec, cut=0.0):
    """NCICCR vs TCGA, 5 draws of 144 NCICCR vs 48 TCGA, 5-fold CV, balanced accuracy (chance 0.50)."""
    Xn, Xt = C.loc[pool, NC].values.T, C.loc[pool, TC].values.T
    out = []
    for seed in range(5):
        pick = np.random.default_rng(seed).choice(len(NC), 3 * len(TC), replace=False)
        Xp = np.vstack([Xn[pick], Xt])
        yp = np.r_[np.zeros(len(pick)), np.ones(len(TC))].astype(int)
        dp = np.zeros(len(yp))
        for tr, te in StratifiedKFold(5, shuffle=True, random_state=42).split(Xp, yp):
            dp[te] = fit(Xp[tr], yp[tr], Xp[te])
        out.append(balanced_accuracy_score(yp, dp > cut))
    return float(np.mean(out))


def myc_diff(C):
    return float(C.loc["MYC", BL].mean() - C.loc["MYC", NC].mean())


rows = []


def record(name, C, pool, dec_fn, probe_val, ctrl_fn=None, values_changed=True):
    """dec_fn(split) -> decision scores on GSE4475; ctrl_fn(split) -> control-gene mask (excluded from rho)."""
    res = {"morph": [], "molec": [], "rho": []}
    for inA in splits:
        inB = ~inA
        dec = dec_fn(inA)
        res["morph"].append(auc(dec, m_bl, m_dl, inB))
        res["molec"].append(auc(dec, mol_bl, mol_dl, inB))
        if values_changed:
            Cs = C(inA) if callable(C) else C
            excl = ctrl_fn(inA) if ctrl_fn else np.zeros(len(genes), bool)
            res["rho"].append(rho_same_lab(Cs, inB, excl))
    C0 = C(splits[0]) if callable(C) else C
    row = {"method": name,
           "auc_morph": np.mean(res["morph"]), "auc_morph_sd": np.std(res["morph"]),
           "auc_molec": np.mean(res["molec"]), "auc_molec_sd": np.std(res["molec"]),
           "rho_same_lab": np.mean(res["rho"]) if res["rho"] else np.nan,
           "myc_diff": myc_diff(C0) if values_changed else myc_diff(R),
           "gap_kept": gap_kept(C0) if values_changed else 1.0,
           "lab_probe": probe_val}
    rows.append(row)
    print(f"{name:34s} AUC morph {row['auc_morph']:.3f}±{row['auc_morph_sd']:.3f} | molec {row['auc_molec']:.3f} | "
          f"rho {row['rho_same_lab']:.3f} | MYC {row['myc_diff']:.2f} | gap kept {row['gap_kept']:.2f} | "
          f"probe {row['lab_probe']:.3f}")


# %% Step 4. Baseline (no correction). Must reproduce the earlier transfer result on all of GSE4475.
all_genes = genes
dec0 = transfer_lr(R, all_genes)
full_morph = auc(dec0, m_bl, m_dl, np.ones(len(morph), bool))
full_molec = auc(dec0, mol_bl, mol_dl, np.ones(len(morph), bool))
print(f"\nBaseline on all of GSE4475: morphology AUC {full_morph:.3f}, molecular {full_molec:.3f} (earlier: 0.88, 0.98)")
print("matches pre-clean run (expected to drift after the BL cleanup):", abs(full_morph - 0.88) < 0.02 and abs(full_molec - 0.98) < 0.02)
print("\nHalf-B results, mean over", N_SPLITS, "splits:")
record("baseline (no correction)", R, all_genes, lambda inA: dec0, probe(R, all_genes))

# %% Step 5. M1a: project out DLBCL-lab directions.
mu = {k: R[c].mean(axis=1).values for k, c in [("NC", NC), ("TC", TC), ("CT", CT)]}
m_tr = R[TR].mean(axis=1).values[:, None]


def project_out(diffs):
    D, _ = np.linalg.qr(np.column_stack(diffs))
    V = R.values - m_tr
    return pd.DataFrame(R.values - D @ (D.T @ V), index=genes, columns=R.columns)


C1a = project_out([mu["CT"] - mu["NC"], mu["TC"] - mu["NC"]])
C1a_fair = project_out([mu["CT"] - mu["NC"]])
dec1a = transfer_lr(C1a, all_genes)
record("M1a lab directions removed", C1a, all_genes, lambda inA: dec1a, probe(C1a_fair, all_genes))

# %% Step 6. M1b: drop lab-sensitive genes.


def cohen_d(a, b):
    sa, sb = R[a].std(axis=1), R[b].std(axis=1)
    sp = np.sqrt(((len(a) - 1) * sa ** 2 + (len(b) - 1) * sb ** 2) / (len(a) + len(b) - 2)).replace(0, np.nan)
    return ((R[a].mean(axis=1) - R[b].mean(axis=1)) / sp).abs().fillna(0)


lab_full = pd.concat([cohen_d(NC, TC), cohen_d(NC, CT), cohen_d(TC, CT)], axis=1).max(axis=1)
lab_fair = cohen_d(NC, CT)
keep_full = genes[(lab_full <= lab_full.median()).values]
keep_fair = genes[(lab_fair <= lab_fair.median()).values]
top0 = R[TR].var(axis=1).sort_values(ascending=False).index[:N_TOP]
print(f"   (M1b: {len(keep_full)} genes kept; {int(top0.isin(keep_full).sum())} of the baseline top-500 survive)")
dec1b = transfer_lr(R, keep_full)
record("M1b lab-sensitive genes dropped", R, keep_full, lambda inA: dec1b, probe(R, keep_fair), values_changed=False)

# %% Step 7. M2: RUV with negative-control genes from half A of GSE4475.


def controls(inA):
    a_bl, a_dl = A.columns[m_bl & inA], A.columns[m_dl & inA]
    t = stats.ttest_ind(A[a_bl], A[a_dl], axis=1, equal_var=False)
    d = A[a_bl].mean(axis=1) - A[a_dl].mean(axis=1)
    return ((np.nan_to_num(t.pvalue) > 0.5) & (d.abs().values < 0.2))


ctrl_sets = [controls(inA) for inA in splits]
print(f"   (M2: control genes per split {min(c.sum() for c in ctrl_sets)}-{max(c.sum() for c in ctrl_sets)})")
V_all = R.values - m_tr
res_tr = np.hstack([R[BL].values - R[BL].values.mean(axis=1, keepdims=True),
                    R[NC].values - R[NC].values.mean(axis=1, keepdims=True)])
alpha_all = np.linalg.svd(res_tr.T, full_matrices=False)[2][:20]       # within-group wobble patterns


def ruv4(k, ctrl):
    al = alpha_all[:k]
    ac = al[:, ctrl]
    W = np.linalg.solve(ac @ ac.T, ac @ V_all[ctrl])                  # each sample's amount, from its controls
    return pd.DataFrame(R.values - al.T @ W, index=genes, columns=R.columns)


def ruvg(k, ctrl):
    P = np.linalg.svd(V_all[ctrl][:, :nTR].T, full_matrices=False)[2][:k]
    W = P @ V_all[ctrl]
    alpha = np.linalg.lstsq(W[:, :nTR].T, V_all[:, :nTR].T, rcond=None)[0]
    return pd.DataFrame(R.values - alpha.T @ W, index=genes, columns=R.columns)


def run_ruv(name, fn, k):
    cache = {}

    def C(inA):
        key = inA.tobytes()
        if key not in cache:
            cache[key] = fn(k, ctrl_sets[[s.tobytes() for s in splits].index(key)])
        return cache[key]

    pr = np.mean([probe(C(inA), all_genes) for inA in splits[:3]])
    record(name, C, all_genes, lambda inA: transfer_lr(C(inA), all_genes), pr,
           ctrl_fn=lambda inA: ctrl_sets[[s.tobytes() for s in splits].index(inA.tobytes())])


run_ruv(f"M2 RUV-4 control genes (k={K_RUV})", ruv4, K_RUV)
run_ruv(f"M2-naive RUVg (k={K_RUV})", ruvg, K_RUV)

# %% Step 8. M3: k-TSP gene-pair rules.


def ktsp_fit(X, y01, k):
    P0 = (X[y01 == 0][:, :, None] < X[y01 == 0][:, None, :]).mean(axis=0)
    P1 = (X[y01 == 1][:, :, None] < X[y01 == 1][:, None, :]).mean(axis=0)
    D = P1 - P0
    S = np.abs(np.triu(D, 1))
    pairs, used = [], set()
    for flat in np.argsort(-S, axis=None):
        i, j = divmod(int(flat), S.shape[1])
        if S[i, j] == 0 or len(pairs) == k:
            break
        if i in used or j in used:
            continue
        pairs.append((i, j, D[i, j] > 0))
        used |= {i, j}
    return pairs


def ktsp_votes(X, pairs):
    v = np.zeros(len(X))
    for i, j, s in pairs:
        lt = X[:, i] < X[:, j]
        v += lt if s else ~lt
    return v                                                           # votes for class 1


def tsp_transfer(k):
    Xtr = R.loc[top0, TR].values.T
    pairs = ktsp_fit(Xtr, (y_tr == "DLBCL").astype(int), k)
    return ktsp_votes(A.loc[top0].values.T, pairs), pairs


def tsp_probe_fit(k):
    def fit(Xtr, ytr, Xte):
        sel = np.argsort(-Xtr.var(axis=0))[:N_TOP]
        return ktsp_votes(Xte[:, sel], ktsp_fit(Xtr[:, sel], ytr, k)) - k / 2
    return fit


dec3, pairs3 = tsp_transfer(K_TSP)
print("   (M3 pairs: " + ", ".join(f"{top0[i]}<{top0[j]}" for i, j, _ in pairs3) + ")")
record(f"M3 k-TSP gene pairs (k={K_TSP})", R, top0, lambda inA: dec3, probe(R, all_genes, tsp_probe_fit(K_TSP)),
       values_changed=False)

# %% Step 9. Sensitivity to the one tuning number of M2 and M3 (reported, not used to pick).
sens = []
for k in [1, 2, 5, 10, 20]:
    a = [auc(transfer_lr(ruv4(k, c), all_genes), m_bl, m_dl, ~inA) for inA, c in zip(splits, ctrl_sets)]
    sens.append({"method": "M2 RUV-4", "k": k, "auc_morph": np.mean(a)})
for k in [1, 3, 5, 9, 25]:
    d, _ = tsp_transfer(k)
    sens.append({"method": "M3 k-TSP", "k": k, "auc_morph": np.mean([auc(d, m_bl, m_dl, ~inA) for inA in splits])})
sens = pd.DataFrame(sens)
print("\nSensitivity (morphology AUC, half B mean):")
print(sens.round(3).to_string(index=False))

# %% Save
res = pd.DataFrame(rows)
res.round(4).to_csv(OUT + "stage08_handwriting_results.csv", index=False)
sens.round(4).to_csv(OUT + "stage08_sensitivity.csv", index=False)
print("\nsaved stage08_handwriting_results.csv, stage08_sensitivity.csv")
