# Stage 10 pre-registration (written 2026-11-20, before any Stage 10 test was run)

**Model under test (frozen):** the protein-coding final model refit on the cleaned training cohort (254 BL tumour samples, 481 NCICCR-DLBCL; Stage 10b). It uses the top-500 variance genes, StandardScaler and L2 logistic regression, the same as 09_protein_coding_model.py Step 4. Nothing about it is changed after seeing the results below.

**Scoring rules (fixed now):**
- RNA-seq cohorts are quantile-mapped onto the training reference, the same as TCGA and CTSP.
- Array cohorts are z-scored per gene within the cohort. Model genes absent from the array are set to 0, the mean.
- The score is AUC (BL vs DLBCL), or the fraction called DLBCL for a DLBCL-only cohort. CIs come from 1,000 within-class bootstraps.
- Every result is reported whatever it is.

| Test | Cohort | Why it matters | Prediction if the model reads disease | Result that would count against it |
|---|---|---|---|---|
| A | CGCI-HTMCP-DLBCL (55 patients, GDC). HIV+ DLBCL sequenced at BCGSC with rRNA depletion, the same center and protocol as the BL training data. | DLBCL written in BL's "handwriting" | At least 80% called DLBCL. TCGA was 100% and CTSP 93%. | Most called BL, which would mean the model reads BCGSC handwriting |
| A2 | Same cohort, old all-gene model | Did non-coding genes carry BCGSC handwriting? | Lower DLBCL call rate than the protein-coding model | — |
| B | GSE10172 (Kiel, U133A). Pediatric BL and DLBCL, consensus morphology. | Age-matched: children in both groups | AUC above 0.75 | AUC near 0.5 |
| C | GSE69049 + GSE69051 (Leeds HMDS, Illumina DASL, FFPE). Clinical diagnosis labels; replicate and relapse samples removed. | Second large same-lab judge, different platform | AUC above 0.75 overall and in adults | AUC near 0.5 |
| D | Within BCGSC: CGCI-BLGSP BL vs HTMCP-DLBCL, cross-validated | BL vs DLBCL with the lab held constant | Clearly separable (balanced accuracy above 85%) | Near 50% |

**Known caveats, stated now:**
- HTMCP patients are HIV+, and some HIV-associated DLBCL resemble BL, so a lower call rate than TCGA is expected even if the model reads disease.
- GSE69051's own "bdc.class" labels are expression-defined, so only the clinical diagnosis is used.
- GSE10172 is small (36 samples).

**Exploratory, no prediction:**
- Within-BL scores for FFPE vs frozen samples.
- A BL-NOS-only sensitivity check, which excludes Burkitt-like and other diagnoses.
