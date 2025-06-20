#!/usr/bin/env python3
import pandas as pd
import xml.etree.ElementTree as ET

# ─── File paths ──────────────────────────────────────────────────────────────
SCORED_CSV = "sig_met_candidates_scored.csv"
HMDB_XML    = "hmdb_metabolites.xml"
OUTPUT_CSV  = "sig_met_candidates_with_props.csv"
#───────────────────────────────────────────────────────────────────────────────

# 1) Load your scored candidates
df = pd.read_csv(SCORED_CSV)

# 2) Stream-parse HMDB and collect key predicted properties
ns = "{http://www.hmdb.ca}"
wanted = {
    "logp": float,
    "logs": float,
    "polar_surface_area": float,
    "donor_count": int,
    "acceptor_count": int,
    "rotatable_bond_count": int
}

hmdb_data = {}
for _, elem in ET.iterparse(HMDB_XML, events=("end",)):
    if elem.tag == f"{ns}metabolite":
        acc = elem.findtext(f"{ns}accession")
        # initialize
        entry = {k: None for k in wanted}
        # grab each <property>
        for p in elem.findall(f"{ns}predicted_properties/{ns}property"):
            kind = p.findtext(f"{ns}kind")
            val  = p.findtext(f"{ns}value")
            if kind in wanted and val is not None:
                try:
                    entry[kind] = wanted[kind](val)
                except ValueError:
                    pass
        hmdb_data[acc] = entry
        elem.clear()

# 3) Build a DataFrame of HMDB props
props_df = (
    pd.DataFrame.from_dict(hmdb_data, orient="index")
      .rename_axis("accession")
      .reset_index()
)

# 4) Merge with your scored table
merged = df.merge(props_df, on="accession", how="left")

# 5) Save the enriched table
merged.to_csv(OUTPUT_CSV, index=False)

print(f"Wrote enriched candidates to {OUTPUT_CSV}")
