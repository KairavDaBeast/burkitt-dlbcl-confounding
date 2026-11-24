"""Stage 10b: fix the BL training cohort.

Problem found 2026-11-20: the rule "first file in sorted file-ID order per patient" picked 6 normal lymphoid
samples and 23 cell-line samples as "BL" training profiles (GDC sample_type), plus slides, FFPE scrolls,
a bone-marrow sample and recurrent tumours.

Rule (fixed before any result was seen): per CGCI-BLGSP patient, use a tumour sample only.
Order of preference: Primary Tumor > FFPE Scrolls > Slides > Recurrent Tumor > Primary Blood Derived Cancer -
Bone Marrow; ties broken by sorted file ID. Lymphoid Normal and Cell Lines are never used. Patients with no
tumour sample are dropped. Diagnoses are saved so a BL-NOS-only sensitivity check can be run.

Writes the cleaned tables under the SAME file names the pipeline reads, after backing up the originals to
full_v1_before_clean/ and gdc_tpm_v1_before_clean.csv.gz, so every script picks up the fix unchanged.
Run once: python 10b_clean_bl_cohort.py
"""
import io, json, os, shutil, tarfile, urllib.request
import pandas as pd

BASE = "data/"
API = "https://api.gdc.cancer.gov/"
PREF = ["Primary Tumor", "FFPE Scrolls", "Slides", "Recurrent Tumor", "Primary Blood Derived Cancer - Bone Marrow"]


def post(endpoint, body, raw=False):
    req = urllib.request.Request(API + endpoint, data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    r = urllib.request.urlopen(req, timeout=600)
    return r.read() if raw else json.load(r)["data"]


# %% 1. Every open CGCI-BLGSP STAR-Counts file with its sample type and diagnosis
filt = {"op": "and", "content": [
    {"op": "in", "content": {"field": "cases.project.project_id", "value": ["CGCI-BLGSP"]}},
    {"op": "in", "content": {"field": "files.data_type", "value": ["Gene Expression Quantification"]}},
    {"op": "in", "content": {"field": "files.analysis.workflow_type", "value": ["STAR - Counts"]}},
    {"op": "in", "content": {"field": "access", "value": ["open"]}}]}
hits = post("files", {"filters": filt, "size": 1000,
                      "fields": "file_id,cases.case_id,cases.samples.sample_type,cases.samples.preservation_method,"
                                "cases.diagnoses.primary_diagnosis"})["hits"]
allf = pd.DataFrame([{"file_id": h["file_id"], "case_id": h["cases"][0]["case_id"],
                      "sample_type": h["cases"][0]["samples"][0].get("sample_type"),
                      "preservation": h["cases"][0]["samples"][0].get("preservation_method"),
                      "diagnosis": (h["cases"][0].get("diagnoses") or [{}])[0].get("primary_diagnosis")} for h in hits])
print(f"CGCI-BLGSP STAR-Counts files: {len(allf)} from {allf.case_id.nunique()} patients")
print(allf.sample_type.value_counts().to_string())

# %% 2. Choose one tumour file per patient
tum = allf[allf.sample_type.isin(PREF)].copy()
tum["rank"] = tum.sample_type.map({s: i for i, s in enumerate(PREF)})
chosen = tum.sort_values(["case_id", "rank", "file_id"]).groupby("case_id").head(1)
old = pd.read_csv(BASE + "full/samples_all.csv")
old_bl = old[old.project == "CGCI-BLGSP"]
dropped = sorted(set(old_bl["sample"]) - set(chosen.case_id))
changed = chosen[chosen.case_id.isin(old_bl["sample"]) &
                 ~chosen.file_id.isin(old_bl["file_id"])]
print(f"\nPatients kept: {len(chosen)} (was {len(old_bl)}); dropped (no tumour sample): {len(dropped)}; "
      f"file changed: {len(changed)}")
print(chosen.sample_type.value_counts().to_string())
print("Diagnoses of kept patients:\n" + chosen.diagnosis.value_counts().to_string())

# %% 3. Download TPM for newly chosen files
need = list(chosen.file_id[~chosen.file_id.isin(old_bl["file_id"])])
new_cols = {}
if need:
    tar_bytes = post("data", {"ids": need}, raw=True)
    with tarfile.open(fileobj=io.BytesIO(tar_bytes), mode="r:gz") as tf:
        for m in tf.getmembers():
            if m.isfile() and m.name.endswith(".tsv"):
                df = pd.read_csv(tf.extractfile(m), sep="\t", comment="#").set_index("gene_id")
                new_cols[m.name.split("/")[0]] = df["tpm_unstranded"]
print(f"Downloaded {len(new_cols)} new files")

# %% 4. Rebuild the full table and the protein-coding table under the same names (originals backed up)
bk = BASE + "full_v1_before_clean/"
if not os.path.exists(bk):
    shutil.copytree(BASE + "full", bk)
    shutil.copy(BASE + "gdc_tpm.csv.gz", BASE + "gdc_tpm_v1_before_clean.csv.gz")
    shutil.copy(BASE + "gdc_samples.csv", BASE + "gdc_samples_v1_before_clean.csv")
# read_pickle is safe here: the file was written by 05_download_full_tpm.py on this PC from GDC data.
full = pd.read_pickle(bk + "gdc_tpm_all.pkl")
fid_to_case = dict(zip(old["file_id"], old["sample"]))
keep_cols = [c for c in full.columns if not (c in set(old_bl["sample"]))]          # non-BL columns unchanged
bl_cols = {}
for r in chosen.itertuples():
    if r.file_id in set(old_bl["file_id"]):
        bl_cols[r.case_id] = full[fid_to_case[r.file_id]]
    else:
        bl_cols[r.case_id] = new_cols[r.file_id].reindex(full.index)
full_new = pd.concat([pd.DataFrame(bl_cols), full[keep_cols]], axis=1)
full_new.to_pickle(BASE + "full/gdc_tpm_all.pkl")
samples_new = pd.concat([pd.DataFrame({"sample": chosen.case_id, "project": "CGCI-BLGSP", "file_id": chosen.file_id}),
                         old[old.project != "CGCI-BLGSP"]])
samples_new.to_csv(BASE + "full/samples_all.csv", index=False)
chosen.to_csv(BASE + "full/bl_chosen_files.csv", index=False)

gs = pd.read_csv(bk.replace("full_v1_before_clean/", "") + "gdc_samples_v1_before_clean.csv")
tpm_old = pd.read_csv(BASE + "gdc_tpm_v1_before_clean.csv.gz", index_col=0)
pc = full_new.loc[tpm_old.index]
cols3 = [c for c in full_new.columns if c in set(samples_new[samples_new.project.isin(
    ["CGCI-BLGSP", "NCICCR-DLBCL", "TCGA-DLBC"])]["sample"])]
out = pd.concat([tpm_old[["gene_name"]], pc[cols3]], axis=1)
out.to_csv(BASE + "gdc_tpm.csv.gz", compression="gzip")
pd.DataFrame({"sample": cols3, "project": samples_new.set_index("sample").loc[cols3, "project"].values}) \
    .to_csv(BASE + "gdc_samples.csv", index=False)
print("Rewrote full/gdc_tpm_all.pkl, full/samples_all.csv, gdc_tpm.csv.gz, gdc_samples.csv; originals backed up")
assert full_new.shape[1] == len(samples_new) and not full_new[list(bl_cols)].loc[tpm_old.index].isna().any().any()
