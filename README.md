# Metabolomics Annotation & RT-Prediction Pipeline

A reproducible, end-to-end workflow for annotating untargeted LC–MS features against HMDB, rescuing unannotated peaks via a comprehensive adduct list, and prioritizing candidates by predicted retention time using machine learning (Random Forest / XGBoost).

---

## Project Overview

1. **Initial Monoisotopic Matching**  
   — Load your significant features (`sig_met_list.txt`, containing `mode`, `metabolite`, `MW`, `m/z`, `RT`)  
   — Stream-parse HMDB XML (v5.0) and match monoisotopic masses within ±5 ppm  
   — Output ~250 direct MW matches

2. **Adduct-Augmented Rescue**  
   — For every feature’s **observed** m/z window (±5 ppm), compute theoretical m/z for an expanded set of adducts & dimers  
   — Positive (ESI⁺): `[M+H]+`, `[M+Na]+`, `[M+K]+`, `[M+NH4]+`,  `[M+ACN+H]+`, `[M+ACN+Na]+`, `[M+ACN+K]+`, `[M+H2O+H]+`
   — Negative (ESI⁻): `[M–H]-`, `[M+HCOO]-`, `[M+CH3COO]-`, `[M+Cl]-`, `[2M–H]-`
   — Collate rescued candidates into `sig_met_candidates_full_adducts.csv`

3. **Feature Scoring & RT-Prediction**  
   — Compute mass‐error and parse HMDB chemical properties (log P, PSA, H-bonds, rotatable bonds, class, 3D‐shape descriptors via RDKit)  
   — Train RF & XGBoost on **manually confirmed** annotations (and singletons) to predict RT  
   — Score every candidate by mass‐error, adduct likelihood, and RT Gaussian likelihood  
   — Ensemble predictions for maximum accuracy  
   — Output ranked candidates for each feature in `sig_met_candidates_final_scored_ensemble_v2.csv`

---

## Quick Start

```bash
# 1) Clone this repo
git clone https://github.com/<your-username>/metabolomics-annotation-pipeline.git
cd metabolomics-annotation-pipeline

# 2) Create & activate conda env
conda create -n met-pqn python=3.10
conda activate met-pqn

# 3) Install dependencies
pip install -r requirements.txt

# 4) Place your data files in this folder:
#    - sig_met_list.txt
#    - hmdb_metabolites.xml (HMDB v5.0 dump)
#    - Any manual annotation CSVs (for RT training)

# 5) Run initial MW matching
python rescue_unannotated.py \
    --sig sig_met_list.txt \
    --hmdb hmdb_metabolites.xml \
    --out sig_met_candidates_mono.tsv

# 6) Run adduct-augmented rescue
python annotate_with_hmdb_adducts.py \
    --sig sig_met_list.txt \
    --hmdb hmdb_metabolites.xml \
    --out sig_met_candidates_full_adducts.csv

# 7) Add chemical properties
python add_properties_from_hmdb.py \
    --in sig_met_candidates_full_adducts.csv \
    --hmdb hmdb_metabolites.xml \
    --out sig_met_candidates_with_props.csv

# 8) Add RDKit 3D descriptors
python add_rdkit_descriptors.py \
    --in sig_met_candidates_with_props.csv \
    --out sig_met_candidates_with_rdkit3d.csv

# 9) Score & rank with RT model
python compare_and_train_xgb.py \
    --in sig_met_candidates_with_rdkit3d_v2.csv \
    --out sig_met_candidates_final_scored_ensemble_v2.csv
# metabolomics-annotation-pipeline
