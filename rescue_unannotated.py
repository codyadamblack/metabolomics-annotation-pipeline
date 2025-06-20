#!/usr/bin/env python3
"""
Rescue and annotate “unannotated” features (±5 ppm) by matching
raw m/z values (with common adducts) against HMDB’s monoisotopic masses.

Inputs (same folder):
  • unannotated_at_5ppm_sig_list.txt
  • hmdb_metabolites.xml

Output:
  • rescued_annotations.csv
"""

import pandas as pd
import xml.etree.ElementTree as ET

# ─── Settings ───────────────────────────────────────────────────────────────
FEATURE_FILE = "unannotated_at_5ppm_sig_list.txt"
HMDB_FILE    = "hmdb_metabolites.xml"
OUTPUT_CSV   = "rescued_annotations.csv"
PPM_WINDOW   = 5.0   # ±5 ppm tolerance
#───────────────────────────────────────────────────────────────────────────────

# Adduct mass offsets in Da
ADDUCTS_POS = {
    "[M+H]+":   1.007276,
    "[M+Na]+": 22.989218,
    "[M+K]+":   38.963158,
    "[M+NH4]+": 18.033823,
}
ADDUCTS_NEG = {
    "[M-H]-":   -1.007276,
    "[M+Cl]-":  34.969402,
}

def load_hmdb(xml_path):
    """
    Stream‐parse hmdb_metabolites.xml. For each <metabolite> element,
    manually iterate its direct children, strip the namespace off
    each tag name, and extract accession, name, and monoisotopic mass.
    Returns a DataFrame with columns [accession, hmdb_name, mono_mw].
    """
    records = []
    # iterparse with 'end' so elements are complete
    for _, elem in ET.iterparse(xml_path, events=("end",)):
        # strip off '{namespace}' if present
        tag = elem.tag.split('}', 1)[-1]
        if tag == "metabolite":
            acc = None
            name = None
            mono = None
            # only look at direct children, not grandchildren
            for child in elem:
                ctag = child.tag.split('}', 1)[-1]
                if ctag == "accession":
                    acc = child.text
                elif ctag == "name":
                    name = child.text
                elif ctag == "monoisotopic_molecular_weight":
                    txt = child.text
                    try:
                        mono = float(txt)
                    except (TypeError, ValueError):
                        mono = None
            # only keep fully populated records
            if acc and name and mono is not None:
                records.append((acc, name, mono))
            # clear from memory
            elem.clear()
    return pd.DataFrame(records, columns=["accession","hmdb_name","mono_mw"])

def expand_adducts(hmdb_df, adducts):
    """
    Given hmdb_df with (accession, hmdb_name, mono_mw) and
    a dict of {adduct:mass_offset}, return DataFrame with
    [accession, hmdb_name, mono_mw, adduct, theo_mz].
    """
    frames = []
    for adduct, delta in adducts.items():
        tmp = hmdb_df.copy()
        tmp["adduct"]  = adduct
        tmp["theo_mz"] = tmp["mono_mw"] + delta
        frames.append(tmp)
    return pd.concat(frames, ignore_index=True)

def main():
    # 1) Load HMDB metabolite masses
    print("Loading HMDB entries…")
    hmdb_df = load_hmdb(HMDB_FILE)
    print(f"  • {len(hmdb_df)} metabolites loaded")

    # 2) Expand to adduct tables
    print("Expanding adduct masses…")
    hmdb_pos = expand_adducts(hmdb_df, ADDUCTS_POS).sort_values("theo_mz")
    hmdb_neg = expand_adducts(hmdb_df, ADDUCTS_NEG).sort_values("theo_mz")

    # 3) Read your unannotated features & compute ±ppm windows
    print("Reading feature list…")
    feats = pd.read_csv(FEATURE_FILE, sep="\t", engine="python")
    feats["tol_Da"] = feats["m/z"] * (PPM_WINDOW / 1e6)
    feats["lo"]     = feats["m/z"] - feats["tol_Da"]
    feats["hi"]     = feats["m/z"] + feats["tol_Da"]

    # 4) Annotate each feature against the correct adduct table
    print("Annotating features…")
    out = []
    for _, f in feats.iterrows():
        mode = f["mode"].split("_",1)[0].lower()
        lookup = hmdb_pos if mode == "pos" else hmdb_neg if mode == "neg" else None
        if lookup is None:
            continue

        # slice by theoretical m/z window
        hits = lookup[
            (lookup["theo_mz"] >= f["lo"]) &
            (lookup["theo_mz"] <= f["hi"])
        ]
        for _, h in hits.iterrows():
            rec = f.to_dict()
            rec.update({
                "accession": h["accession"],
                "hmdb_name": h["hmdb_name"],
                "adduct":    h["adduct"],
                "theo_mz":   h["theo_mz"],
            })
            out.append(rec)

    # 5) Write out rescued annotations
    rescued = pd.DataFrame(out)
    rescued.to_csv(OUTPUT_CSV, index=False)
    print(f"Done — recovered {len(rescued)} annotations → {OUTPUT_CSV}")

if __name__ == "__main__":
    main()
