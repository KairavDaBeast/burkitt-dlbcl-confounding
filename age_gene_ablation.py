"""Age-gene ablation (follow-up to age_partial_check.py).

Question: if genes that track age within BL are removed from the candidate pool, does the
BL-vs-DLBCL classifier still work?

Strictly nested. For each 5-fold CV split (same folds as 06_rebuild_results_nested.py):
  1. Normalization is fitted on the training fold only (as in 06_rebuild_results_nested.py).
  2. Age rule, computed from TRAINING BL samples that have age only (no held-out data, no DLBCL):
     Spearman rho of each gene with age, Benjamini-Hochberg FDR across genes.
     Primary rule: remove genes with FDR < 0.05. Secondary rule: remove genes with |rho| >= 0.3.
  3. The same pipeline runs on the remaining genes: top-500 variance (label-free), StandardScaler,
     L2 logistic regression. Labels of held-out samples are never used by any step.
Controls: baseline (nothing removed) and 20 random removals of the same size as the primary rule.

Age is never a feature of the classifier. It is used only to choose which genes to remove.

Run: python age_gene_ablation.py   (from the project folder)
"""
import json
import urllib.request
import warnings
warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, balanced_accuracy_score, confusion_matrix
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler

DATA = "data/full/"
OUT = ""
N_TOP = 500
N_CONTROL = 20
rng = np.random.default_rng(7)

# %% Step 1. Ages from GDC (same query as age_partial_check.py)
body = json.dumps({
    "filters": {"op": "in", "content": {"field": "project.project_id",
                                        "value": ["CGCI-BLGSP", "NCICCR-DLBCL", "TCGA-DLBC"]}},
    "fields": "case_id,project.project_id,demographic.age_at_index",
    "size": 3000}).encode()
req = urllib.request.Request("https://api.gdc.cancer.gov/cases", data=body,
                             headers={"Content-Type": "application/json"})
hits = json.load(urllib.request.urlopen(req, timeout=120))["data"]["hits"]
age_map = {h["case_id"]: (h.get("demographic") or {}).get("age_at_index") for h in hits}

# %% Step 2. Expression (same preprocessing as 06_rebuild_results_nested.py)
# read_pickle is safe here: the file is written by 05_download_full_tpm.py on this PC from GDC data.
full = pd.read_pickle(DATA + "gdc_tpm_all.pkl")
full = full[full.index.isin(pd.read_csv("data/gdc_tpm.csv.gz", index_col=0, usecols=[0]).index)]  # Stage 09 switch (2026-11-20): protein-coding genes only
proj = pd.read_csv(DATA + "samples_all.csv", index_col="sample")["project"].reindex(full.columns)
bl_cols = full.columns[proj == "CGCI-BLGSP"]
nc_cols = full.columns[proj == "NCICCR-DLBCL"]
train_raw = full[list(bl_cols) + list(nc_cols)].dropna(axis=0, how="any")
genes = train_raw.index
cols = train_raw.columns
M = np.log2(train_raw.values.astype(float) + 1)          # genes x samples
bl_set = set(bl_cols)
y = np.array(["BL" if c in bl_set else "DLBCL" for c in cols], dtype=object)
age = np.array([np.nan if age_map.get(c) is None else float(age_map[c]) for c in cols])
G = len(genes)
print(f"Genes {G} | BL {(y == 'BL').sum()} (with age {int(np.sum((y == 'BL') & ~np.isnan(age)))}) | DLBCL {(y == 'DLBCL').sum()}")


def reference_from(M_train):
    return np.sort(M_train, axis=0).mean(axis=1)


def map_to_reference(M_in, ref):
    ranks = pd.DataFrame(M_in).rank(method="average", axis=0).values
    pos = np.arange(1, len(ref) + 1)
    out = np.empty(M_in.shape, dtype=float)
    for c in range(M_in.shape[1]):
        out[:, c] = np.interp(ranks[:, c], pos, ref)
    return out


def spearman_per_gene(X, a):
    """X: genes x n, a: length n. Returns Spearman rho and two-sided p for every gene."""
    Rx = stats.rankdata(X, axis=1)
    Ra = stats.rankdata(a)
    Rx = Rx - Rx.mean(axis=1, keepdims=True)
    Ra = Ra - Ra.mean()
    r = (Rx @ Ra) / (np.sqrt((Rx ** 2).sum(axis=1)) * np.sqrt((Ra ** 2).sum()))
    n = len(a)
    t = r * np.sqrt((n - 2) / np.clip(1 - r ** 2, 1e-12, None))
    return r, 2 * stats.t.sf(np.abs(t), n - 2)


def fit_predict(Xtr, ytr, Xte, pool, sel_base=None):
    """Same pipeline on the genes in `pool` (boolean over all genes)."""
    Xtr_p, Xte_p = Xtr[:, pool], Xte[:, pool]
    sel = np.argsort(-Xtr_p.var(axis=0))[:N_TOP]        # top-500 variance, label-free
    scaler = StandardScaler().fit(Xtr_p[:, sel])
    clf = LogisticRegression(max_iter=1000, penalty="l2").fit(scaler.transform(Xtr_p[:, sel]), ytr)
    return clf.predict(scaler.transform(Xte_p[:, sel])), np.where(pool)[0][sel]


# %% Step 3. Nested CV
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
folds = list(cv.split(np.zeros(len(y)), y))
pred = {"baseline": np.empty(len(y), dtype=object),
        "fdr05": np.empty(len(y), dtype=object),
        "rho03": np.empty(len(y), dtype=object)}
ctrl_pred = [np.empty(len(y), dtype=object) for _ in range(N_CONTROL)]
stats_rows = []
for k, (tr, te) in enumerate(folds):
    ref = reference_from(M[:, tr])
    Ntr = map_to_reference(M[:, tr], ref)                # genes x training samples
    Xtr = Ntr.T
    Xte = map_to_reference(M[:, te], ref).T
    ytr = y[tr]

    # Age rule from TRAINING BL samples with age only
    pos_bl = np.where((ytr == "BL") & ~np.isnan(age[tr]))[0]
    r, p = spearman_per_gene(Ntr[:, pos_bl], age[tr][pos_bl])
    q = stats.false_discovery_control(p)
    rm_fdr = q < 0.05
    rm_rho = np.abs(r) >= 0.3

    all_genes = np.ones(G, dtype=bool)
    pred["baseline"][te], sel_base = fit_predict(Xtr, ytr, Xte, all_genes)
    pred["fdr05"][te], _ = fit_predict(Xtr, ytr, Xte, ~rm_fdr)
    pred["rho03"][te], _ = fit_predict(Xtr, ytr, Xte, ~rm_rho)

    n_fdr = int(rm_fdr.sum())
    for c in range(N_CONTROL):
        rm_rand = np.zeros(G, dtype=bool)
        rm_rand[rng.choice(G, size=n_fdr, replace=False)] = True
        ctrl_pred[c][te], _ = fit_predict(Xtr, ytr, Xte, ~rm_rand)

    stats_rows.append({"fold": k, "bl_train_with_age": len(pos_bl), "removed_fdr05": n_fdr,
                       "removed_rho03": int(rm_rho.sum()),
                       "baseline_top500_removed_fdr05": int(rm_fdr[sel_base].sum()),
                       "baseline_top500_removed_rho03": int(rm_rho[sel_base].sum())})
    print(f"fold {k}: BL with age in train {len(pos_bl)}; removed FDR<0.05 {n_fdr}; "
          f"removed |rho|>=0.3 {int(rm_rho.sum())}; of baseline top-500, removed {int(rm_fdr[sel_base].sum())} (FDR) "
          f"/ {int(rm_rho[sel_base].sum())} (rho)")


# %% Step 4. Results
def summarise(name, p_):
    cm = confusion_matrix(y, p_, labels=["BL", "DLBCL"])
    return {"model": name, "accuracy": accuracy_score(y, p_), "balanced_accuracy": balanced_accuracy_score(y, p_),
            "BL_correct": int(cm[0, 0]), "BL_total": int(cm[0].sum()),
            "DLBCL_correct": int(cm[1, 1]), "DLBCL_total": int(cm[1].sum())}


rows = [summarise("baseline (nothing removed)", pred["baseline"]),
        summarise("remove age genes, FDR < 0.05 (primary)", pred["fdr05"]),
        summarise("remove age genes, |rho| >= 0.3 (secondary)", pred["rho03"])]
ctrl = pd.DataFrame([summarise(f"control: random removal, draw {c}", ctrl_pred[c]) for c in range(N_CONTROL)])
res = pd.DataFrame(rows)
print("\n" + res.round(4).to_string(index=False))
print(f"\nControl (random removal of the same count as FDR rule, {N_CONTROL} draws):")
print(f"   accuracy mean {ctrl['accuracy'].mean():.4f}, min {ctrl['accuracy'].min():.4f}, max {ctrl['accuracy'].max():.4f}")
print(f"   balanced mean {ctrl['balanced_accuracy'].mean():.4f}, min {ctrl['balanced_accuracy'].min():.4f}")
st = pd.DataFrame(stats_rows)
print("\nPer-fold removal counts:")
print(st.to_string(index=False))

res.to_csv(OUT + "age_ablation_results.csv", index=False)
ctrl.to_csv(OUT + "age_ablation_control_draws.csv", index=False)
st.to_csv(OUT + "age_ablation_fold_counts.csv", index=False)
print("saved age_ablation_results.csv, age_ablation_control_draws.csv, age_ablation_fold_counts.csv")
