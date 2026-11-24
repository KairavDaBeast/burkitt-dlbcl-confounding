# Burkitt lymphoma vs DLBCL: what does a classifier trained on confounded public RNA-seq learn?

This is the code for the paper "A Classifier Trained on Confounded Public RNA-seq Data Transfers to an Unconfounded Cohort: Burkitt Lymphoma versus Diffuse Large B-Cell Lymphoma" (K. Sapru, manuscript in preparation).

In the NCI GDC, every Burkitt lymphoma (BL) sample comes from one project (CGCI-BLGSP) and every DLBCL sample from others, so disease and laboratory are perfectly confounded. These scripts:
- train an L2 logistic regression;
- audit what it learned (laboratory probes, gene-type and batch-correction tests);
- test it on same-laboratory cohorts, including pre-registered tests (`STAGE10_PREREGISTRATION.md`).

## Requirements

Python 3.12 with numpy 2.5.2, pandas 3.0.5, scikit-learn 1.9.0, scipy 1.18.0 and matplotlib 3.11.1 (`pip install -r requirements.txt`).

## Data (not included; all public)

Scripts read from `data/`.

1. GDC RNA-seq (STAR-Counts `tpm_unstranded`): `python 05_download_full_tpm.py`, then `python 07_download_gdc_tpm.py`.
2. GEO GSE4475 and the GPL96 annotation go into `data/geo/`:
   - `curl -o data/geo/GSE4475_series_matrix.txt.gz https://ftp.ncbi.nlm.nih.gov/geo/series/GSE4nnn/GSE4475/matrix/GSE4475_series_matrix.txt.gz`
   - `curl -o data/geo/GPL96.annot.gz https://ftp.ncbi.nlm.nih.gov/geo/platforms/GPLnnn/GPL96/annot/GPL96.annot.gz`
3. Stage 10 cohorts: `python 10_download_new_cohorts.py` downloads CGCI-HTMCP-DLBCL from the GDC. If its GEO step is refused, download into `data/new/`:
   - the series matrices for GSE10172, GSE69049 and GSE69051, from `https://ftp.ncbi.nlm.nih.gov/geo/series/<stub>/<acc>/matrix/`;
   - the platform tables `GPL8432_table.txt` and `GPL14951_table.txt`, from `https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=<GPL>&targ=self&form=text&view=data`.
4. Clean the BL cohort: `python 10b_clean_bl_cohort.py`. This drops 29 BL patients whose only samples were cell lines (23) or normal tissue (6).

## Run order (after the data steps)

```
python 06_rebuild_results.py
python 06_rebuild_results_nested.py      # main cross-validation and permutation test
python 05_batch_confound_check_local.py  # external DLBCL cohorts, lab probe, top-20
python 05_fig7_batch_check.py
python 06b_hbb_malat1_check.py
python 07_same_lab_check.py              # gene-level same-lab check, Fig 8
python age_partial_check.py
python age_gene_ablation.py
python biology_gene_set_model.py         # gene-set controls, transfer to GSE4475
python fig9_transfer.py
python 06c_figures_1_to_6.py
python 09_protein_coding_model.py        # final model, applied directly to GSE4475
python 08b_probe_gene_space.py           # lab signal by gene type
python 08_handwriting_fix.py             # batch-correction attempts
python 10c_external_tests.py             # pre-registered Stage 10 tests
python 10d_same_lab_ranking.py           # post hoc ranking check
python 11_top_gene_table.py              # top-50 replication table, model kit
```

All model scripts use protein-coding genes only (the line marked "Stage 09 switch").

## Contents

- `results/`: summary tables behind the paper's numbers. Patient-level outputs (per-sample scores, demographics) are not included.
- `model_kit/`: the frozen model as CSV files plus `score_cohort.py`, so another lab can score its own data without sharing it. See `model_kit/README.md`.
- `STAGE10_PREREGISTRATION.md`: predictions written before the Stage 10 tests were run.

## Use of AI

The analysis code was written with Claude (Anthropic), as disclosed in the paper.
