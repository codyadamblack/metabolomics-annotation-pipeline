import pandas as pd
import xml.etree.ElementTree as ET
import re

# ─── Settings & File Paths ────────────────────────────────────────────────────
INPUT_CSV    = "sig_met_candidates_with_props_v2.csv"
HMDB_XML     = "hmdb_metabolites.xml"
OUTPUT_CSV   = "sig_met_candidates_with_extended_features.csv"

# ─── 1) Load existing candidate table
df = pd.read_csv(INPUT_CSV)

# ─── 2) Parse HMDB XML for taxonomy & formula & SMILES
ns = "{http://www.hmdb.ca}"
hmdb_data = {}

for _, elem in ET.iterparse(HMDB_XML, events=("end",)):
    if elem.tag == f"{ns}metabolite":
        acc = elem.findtext(f"{ns}accession")
        # extract chemical formula, SMILES
        formula = elem.findtext(f".//{ns}chemical_formula")
        smiles  = elem.findtext(f".//{ns}smiles")
        # taxonomy
        tax_elem = elem.find(f"{ns}taxonomy")
        cls = tax_elem.findtext(f"{ns}class") if tax_elem is not None else None
        sub = tax_elem.findtext(f"{ns}sub_class") if tax_elem is not None else None
        
        hmdb_data[acc] = {
            "formula": formula,
            "smiles": smiles,
            "chem_class": cls,
            "chem_sub_class": sub
        }
        elem.clear()

# Convert hmdb_data to DataFrame
hmdb_df = pd.DataFrame.from_dict(hmdb_data, orient="index").reset_index()
hmdb_df.rename(columns={"index": "accession"}, inplace=True)

# ─── 3) Merge with candidate DataFrame
df = df.merge(hmdb_df, on="accession", how="left")

# ─── 4) Derive formula‐based features
def parse_formula(formula):
    """Parse elemental counts from a formula string."""
    counts = {"C":0, "H":0, "N":0, "O":0, "P":0, "S":0}
    if pd.isna(formula):
        return counts
    for elem, num in re.findall(r"([A-Z][a-z]?)(\d*)", formula):
        if elem in counts:
            counts[elem] = int(num) if num else 1
    return counts

# Apply parsing
elem_counts = df["formula"].apply(parse_formula).apply(pd.Series)
df = pd.concat([df, elem_counts], axis=1)

# H/C ratio and heteroatom count
df["H_C_ratio"] = df["H"] / df["C"].replace(0, pd.NA)
df["heteroatom_count"] = df[["N", "O", "P", "S"]].sum(axis=1)

# ─── 5) Export the enriched table
df.to_csv(OUTPUT_CSV, index=False)
print(f"Saved enriched features to {OUTPUT_CSV}")
