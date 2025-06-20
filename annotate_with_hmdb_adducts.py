#!/usr/bin/env python3
import pandas as pd
import xml.etree.ElementTree as ET

# ─── Settings ────────────────────────────────────────────────────────────────
SIG_FILE   = "sig_met_list.txt"       # your 555-feature list
HMDB_XML   = "hmdb_metabolites.xml"   # HMDB dump
PPM_WINDOW = 5.0                      # ±5 ppm tolerance
OUT_CSV    = "sig_met_candidates_full_adducts.csv"
#───────────────────────────────────────────────────────────────────────────────

# namespace prefix exactly as in the HMDB dump
ns_prefix = "{http://www.hmdb.ca}"

# All adduct mass offsets (Da); None for the 2M dimers
ADDUCTS = {
    # ESI⁺
    "[M+H]+":    1.007276,
    "[M+Na]+":  22.989218,
    "[M+K]+":   38.963158,
    "[M+NH4]+": 18.033823,
    "[M+Li]+":   7.015455,
    "[M+Cs]+": 132.905452,
    # ESI⁻
    "[M-H]-":    -1.007276,
    "[M+HCOO]-": 44.998201,
    "[M+Cl]-":   34.969402,
    # dimers (handled below)
    "[2M+H]+":   None,
    "[2M+Na]+":  None,
    "[2M-H]-":   None,
}

def compute_theo_mz(mono, adduct):
    d = ADDUCTS[adduct]
    if d is not None:
        return mono + d
    # 2M dimers
    if adduct == "[2M+H]+":
        return 2*mono + ADDUCTS["[M+H]+"]
    if adduct == "[2M+Na]+":
        return 2*mono + ADDUCTS["[M+Na]+"]
    if adduct == "[2M-H]-":
        return 2*mono + ADDUCTS["[M-H]-"]
    return None

def main():
    # 1) Read features with tab delimiter so that 'm/z' is parsed correctly
    feats = pd.read_csv(SIG_FILE, sep="\t", engine="python")
    # trim whitespace from column names
    feats.columns = feats.columns.str.strip()

    # 2) Build ±ppm windows on the OBSERVED m/z
    feats["tol"] = feats["m/z"] * (PPM_WINDOW / 1e6)
    feats["lo"]  = feats["m/z"] - feats["tol"]
    feats["hi"]  = feats["m/z"] + feats["tol"]

    out = []

    # 3) Stream-parse HMDB and annotate
    for _, elem in ET.iterparse(HMDB_XML, events=("end",)):
        if elem.tag == f"{ns_prefix}metabolite":
            acc       = elem.findtext(f"{ns_prefix}accession")
            name      = elem.findtext(f"{ns_prefix}name")
            mono_txt  = elem.findtext(f"{ns_prefix}monisotopic_molecular_weight")
            try:
                mono = float(mono_txt)
            except (TypeError, ValueError):
                elem.clear()
                continue

            # 4) For each feature, choose only + adducts on pos rows, – on neg rows
            for _, feat in feats.iterrows():
                mode = feat["mode"].split("_",1)[0].lower()
                if mode == "pos":
                    adsets = [a for a in ADDUCTS if a.endswith("]+")]
                elif mode == "neg":
                    adsets = [a for a in ADDUCTS if a.endswith("]-")]
                else:
                    continue

                # 5) Compute and match each theoretical m/z
                for adduct in adsets:
                    mz = compute_theo_mz(mono, adduct)
                    if mz is None:
                        continue
                    if feat.lo <= mz <= feat.hi:
                        rec = feat.to_dict()
                        rec.update({
                            "accession": acc,
                            "hmdb_name": name,
                            "adduct":    adduct,
                            "theo_mz":   mz,
                        })
                        out.append(rec)

            elem.clear()

    # 6) Save rescued annotations
    pd.DataFrame(out).to_csv(OUT_CSV, index=False, encoding="utf-8-sig")
    print(f"Wrote {len(out)} candidate annotations to {OUT_CSV}")

if __name__ == "__main__":
    main()
