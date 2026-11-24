"""Stage 07, part 1: re-download the RNA-seq cohort from GDC as one compact TPM table.

Same open STAR-Counts files as Stage 01 (CGCI-BLGSP Burkitt, NCICCR-DLBCL), plus TCGA-DLBC
(a second DLBCL lab, used as the lab-only control in Stage 07). Each file is streamed, only the
tpm_unstranded column for protein-coding genes is kept, and the raw file is never saved.
One profile per Burkitt patient (first file_id in sorted order), like Stage 01.

Output: data/gdc_tpm.csv.gz  (genes x samples)
        data/gdc_samples.csv  (sample -> project)
Run:    python 07_download_gdc_tpm.py
"""
import io, json, os, time, urllib.request
from concurrent.futures import ThreadPoolExecutor
import pandas as pd

OUT = "data/"
API = "https://api.gdc.cancer.gov/"
PROJECTS = ["CGCI-BLGSP", "NCICCR-DLBCL", "TCGA-DLBC"]


def list_files(project):
    filt = {"op": "and", "content": [
        {"op": "in", "content": {"field": "cases.project.project_id", "value": [project]}},
        {"op": "in", "content": {"field": "data_type", "value": ["Gene Expression Quantification"]}},
        {"op": "in", "content": {"field": "analysis.workflow_type", "value": ["STAR - Counts"]}},
        {"op": "in", "content": {"field": "access", "value": ["open"]}}]}
    body = json.dumps({"filters": filt, "fields": "file_id,cases.case_id", "size": 5000}).encode()
    req = urllib.request.Request(API + "files", data=body, headers={"Content-Type": "application/json"})
    hits = json.load(urllib.request.urlopen(req, timeout=120))["data"]["hits"]
    one_per_patient = {}
    for h in sorted(hits, key=lambda h: h["file_id"]):
        one_per_patient.setdefault(h["cases"][0]["case_id"], h["file_id"])
    return [(fid, case, project) for case, fid in one_per_patient.items()]


def fetch_tpm(fid):
    # GDC throws occasional 500s on single files: retry with backoff, then skip (see project memory).
    for attempt in range(4):
        try:
            raw = urllib.request.urlopen(API + "data/" + fid, timeout=300).read().decode()
            df = pd.read_csv(io.StringIO(raw), sep="\t", comment="#")
            df = df[df["gene_type"] == "protein_coding"]
            return df.set_index("gene_id")["tpm_unstranded"].astype("float32"), df.set_index("gene_id")["gene_name"]
        except Exception as e:
            last_err = e
            time.sleep(5 * (attempt + 1))
    print("SKIPPED", fid, last_err, flush=True)
    return None, None


if __name__ == "__main__":
    os.makedirs(OUT, exist_ok=True)
    files = [f for p in PROJECTS for f in list_files(p)]
    print(len(files), "files to fetch")
    cols, names, meta = {}, None, []
    with ThreadPoolExecutor(8) as pool:
        for i, ((fid, case, project), (tpm, gene_names)) in enumerate(
                zip(files, pool.map(lambda f: fetch_tpm(f[0]), files))):
            if tpm is None:
                continue
            cols[case] = tpm
            names = gene_names if names is None else names
            meta.append({"sample": case, "project": project, "file_id": fid})
            if i % 100 == 0:
                print(i, "done", flush=True)
    mat = pd.DataFrame(cols)
    mat.insert(0, "gene_name", names.reindex(mat.index))
    mat.to_csv(OUT + "gdc_tpm.csv.gz", float_format="%.3f")
    pd.DataFrame(meta).to_csv(OUT + "gdc_samples.csv", index=False)
    print("saved", mat.shape, pd.DataFrame(meta)["project"].value_counts().to_dict())
