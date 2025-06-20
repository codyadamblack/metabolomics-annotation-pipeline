#!/usr/bin/env python3
"""
add_rdkit_descriptors.py

Reads in 'sig_met_candidates_with_extended_features.csv', computes additional
3D shape descriptors (PMI1, PMI2, asphericity) via RDKit, and writes out
'sig_met_candidates_with_rdkit3d.csv'.
"""

import pandas as pd
from rdkit import Chem
from rdkit.Chem import Descriptors, rdMolDescriptors, AllChem

def calc_3d_shape(mol, embed_iter=10):
    """
    Generate one 3D conformer and return:
      - PMI1, PMI2 (principal moments of inertia ratios)
      - asphericity
    If embedding fails, returns (None, None, None).
    """
    m3d = Chem.AddHs(mol)
    # embed a single conformer, with up to embed_iter attempts
    res = AllChem.EmbedMultipleConfs(
        m3d,
        numConfs=1,
        maxAttempts=embed_iter,
        pruneRmsThresh=0.5,
        randomSeed=42
    )
    if not res:
        return None, None, None

    conf_id = res[0]
    AllChem.UFFOptimizeMolecule(m3d, confId=conf_id)

    # calculate shape descriptors
    pmi1 = rdMolDescriptors.CalcPMI1(m3d, confId=conf_id)
    pmi2 = rdMolDescriptors.CalcPMI2(m3d, confId=conf_id)
    asp  = rdMolDescriptors.CalcAsphericity(m3d, confId=conf_id)
    return pmi1, pmi2, asp

def main():
    # 1) load the extended-features table
    infile  = "sig_met_candidates_with_extended_features.csv"
    outfile = "sig_met_candidates_with_rdkit3d.csv"

    df = pd.read_csv(infile)

    # 2) for each row, parse SMILES and compute 3D shape descriptors
    results = []
    for idx, row in df.iterrows():
        smi = row.get("smiles", "")
        if not smi or pd.isna(smi):
            results.append((None, None, None))
            continue

        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            results.append((None, None, None))
            continue

        pmi1, pmi2, asp = calc_3d_shape(mol)
        results.append((pmi1, pmi2, asp))

    # 3) attach results and save
    df[["PMI1", "PMI2", "asphericity"]] = pd.DataFrame(
        results, columns=["PMI1", "PMI2", "asphericity"], index=df.index
    )
    df.to_csv(outfile, index=False)
    print(f"Wrote {len(df)} rows of shape descriptors → {outfile}")

if __name__ == "__main__":
    main()
