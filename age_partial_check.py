"""Age partial checks (draft Limitation 5 / Author note B).

Age is available for BL (CGCI-BLGSP, 248 of 283 expected) and TCGA-DLBC from GDC.
It is not available for NCICCR-DLBCL, so the BL-vs-DLBCL age confound cannot be tested directly.
Two partial checks:
  1. Within BL: Spearman(out-of-fold decision value from the nested CV, age).
  2. Within TCGA-DLBC (external, never used in training): Spearman(final-model decision value, age).
A strong correlation would suggest an age-related signal in the score. A weak one does not clear
the between-group confound.

Run after 06_rebuild_results_nested.py (needs oof_scores_nested.csv).
"""
import json
import urllib.request
import warnings
warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

DATA = "data/full/"
OUT = ""
N_TOP = 500

# %% Step 1. Pull age from GDC (age_at_index, years)
body = json.dumps({
    "filters": {"op": "in", "content": {"field": "project.project_id",
                                        "value": ["CGCI-BLGSP", "NCICCR-DLBCL", "TCGA-DLBC"]}},
    "fields": "case_id,project.project_id,demographic.age_at_index",
    "size": 3000}).encode()
req = urllib.request.Request("https://api.gdc.cancer.gov/cases", data=body,
                             headers={"Content-Type": "application/json"})
hits = json.load(urllib.request.urlopen(req, timeout=120))["data"]["hits"]
rows = [{"case_id": h["case_id"], "project": h["project"]["project_id"],
         "age": (h.get("demographic") or {}).get("age_at_index")} for h in hits]
ages = pd.DataFrame(rows).set_index("case_id")
for proj_name, grp in ages.groupby("project"):
    a = grp["age"].dropna()
    print(f"{proj_name}: {len(grp)} cases, age available for {len(a)}"
          + (f", median {a.median():.1f}, IQR {a.quantile(.25):.1f}-{a.quantile(.75):.1f}, range {a.min():.0f}-{a.max():.0f}" if len(a) else ""))
# Validation of the field: the facts sheet says BL age is available for 248 of 283, median 10.5.

# %% Step 2. Within BL: out-of-fold nested-CV score vs age
oof = pd.read_csv(OUT + "oof_scores_nested.csv", index_col="sample")
bl = oof[oof["label"] == "BL"].join(ages["age"], how="inner").dropna(subset=["age"])
rho_bl, p_bl = stats.spearmanr(bl["oof_decision"], bl["age"])
print(f"\nWithin BL (n={len(bl)} with age): Spearman(oof score, age) = {rho_bl:.3f}, p = {p_bl:.3g}")
print("   (positive = higher age, more DLBCL-like score)")

# %% Step 3. Within TCGA-DLBC: final model (fit on all 764, joint normalization as in Stage 05) vs age
# read_pickle is safe here: the file is written by 05_download_full_tpm.py on this PC from GDC data.
full = pd.read_pickle(DATA + "gdc_tpm_all.pkl")
full = full[full.index.isin(pd.read_csv("data/gdc_tpm.csv.gz", index_col=0, usecols=[0]).index)]  # Stage 09 switch (2026-11-20): protein-coding genes only
proj = pd.read_csv(DATA + "samples_all.csv", index_col="sample")["project"].reindex(full.columns)
bl_cols = full.columns[proj == "CGCI-BLGSP"]
nc_cols = full.columns[proj == "NCICCR-DLBCL"]
tc_cols = full.columns[proj == "TCGA-DLBC"]
train_raw = full[list(bl_cols) + list(nc_cols)].dropna(axis=0, how="any")
genes = train_raw.index
M = np.log2(train_raw.values.astype(float) + 1)
bl_set = set(bl_cols)
y = np.array(["BL" if c in bl_set else "DLBCL" for c in train_raw.columns], dtype=object)


def reference_from(M_train):
    return np.sort(M_train, axis=0).mean(axis=1)


def map_to_reference(M_in, ref):
    ranks = pd.DataFrame(M_in).rank(method="average", axis=0).values
    pos = np.arange(1, len(ref) + 1)
    out = np.empty(M_in.shape, dtype=float)
    for c in range(M_in.shape[1]):
        out[:, c] = np.interp(ranks[:, c], pos, ref)
    return out


ref_all = reference_from(M)
Xall = map_to_reference(M, ref_all).T
sel = np.argsort(-Xall.var(axis=0))[:N_TOP]
scaler = StandardScaler().fit(Xall[:, sel])
clf = LogisticRegression(max_iter=1000, penalty="l2").fit(scaler.transform(Xall[:, sel]), y)

tc_raw = full[tc_cols].reindex(genes).fillna(0.0)          # Stage 05 rule: missing gene -> 0
Xtc = map_to_reference(np.log2(tc_raw.values.astype(float) + 1), ref_all).T
dec_tc = clf.decision_function(scaler.transform(Xtc[:, sel]))
tc = pd.DataFrame({"decision": dec_tc}, index=tc_cols).join(ages["age"], how="inner").dropna(subset=["age"])
rho_tc, p_tc = stats.spearmanr(tc["decision"], tc["age"])
print(f"\nWithin TCGA-DLBC (n={len(tc)} with age, external): Spearman(decision, age) = {rho_tc:.3f}, p = {p_tc:.3g}")

# %% Step 4. Save
pd.DataFrame([
    {"check": "BL nested oof score vs age", "n": len(bl), "spearman": rho_bl, "p": p_bl},
    {"check": "TCGA-DLBC final-model score vs age", "n": len(tc), "spearman": rho_tc, "p": p_tc},
]).to_csv(OUT + "age_partial_check.csv", index=False)
print("saved age_partial_check.csv")
