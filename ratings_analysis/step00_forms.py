# -*- coding: utf-8 -*-
"""Step 00. Rebuild the long rating table from the eight raw Google Form tabs of the ratings workbook.
Each form tab has 5 leading columns (Item ID, Timestamp, Email Address, Student Email, Labeler_ID) followed by
30 blocks of 9 question columns (PU, SU, UN, PP, CS, GT, AC, FW, GN), one block per response in the condition.
E-mail columns are read only to be discarded; nothing identifying is written.
Output: form_long.parquet (one row per rater x response submission, full timestamps kept)."""
import os, re, sys, pandas as pd

XLSX = os.environ.get("RATINGS_XLSX", "combined_label_data.xlsx")
FORM_TABS = {"PSGM Persona_and_Steps_L": "Persona_and_Steps_L", "PRGM Persona_Role_L": "Persona_Role_L",
             "ASGM Action_Steps_L": "Action_Steps_L", "BPGM Base_Prompts_L": "Base_Prompts_L",
             "PSAR Persona_and_Steps_AR": "Persona_and_Steps_AR", "BPAR Base_Prompts_AR": "Base_Prompts_AR",
             "PRAR Persona_Role_AR": "Persona_Role_AR", "ASAR Action_Steps_AR": "Action_Steps_AR"}
Q = ["PU", "SU", "UN", "PP", "CS", "GT", "AC", "FW", "GN"]

def main():
    xl = pd.ExcelFile(XLSX)
    rows = []
    for tab, prefix in FORM_TABS.items():
        t = xl.parse(tab)
        cols = list(t.columns)
        assert cols[0] == "Item ID" and str(cols[4]).lower().startswith("labeler"), f"{tab}: unexpected leading columns {cols[:5]}"
        blocks = cols[5:]
        assert len(blocks) % 9 == 0, f"{tab}: {len(blocks)} question columns is not a multiple of 9"
        n_items = len(blocks) // 9
        for _, r in t.iterrows():
            if pd.isna(r["Labeler_ID"]):
                continue
            for i in range(n_items):
                b = blocks[9 * i: 9 * i + 9]
                rec = dict(form_item_id=r["Item ID"], ts=pd.to_datetime(r["Timestamp"]), LABELER_ID=str(r["Labeler_ID"]).strip(),
                           Prompt_ID=f"{prefix}_{i + 1:03d}")
                for q, c in zip(Q, b):
                    rec[q] = r[c]
                rows.append(rec)
    fl = pd.DataFrame(rows)
    for d in ["PU", "PP", "CS", "GT", "AC", "FW"]:
        fl[d] = pd.to_numeric(fl[d], errors="coerce")
    missing = fl[["PU", "PP", "CS", "GT", "AC", "FW"]].isna().any(axis=1).sum()
    print(f"form rows: {len(fl)}; raters: {fl.LABELER_ID.nunique()}; responses: {fl.Prompt_ID.nunique()}; rows with a missing rating: {missing}")
    # cross-check against MAIN (date-only timestamps there), excluding double submissions which MAIN cannot disambiguate
    main = xl.parse("MAIN"); main.columns = [c.strip() for c in main.columns]
    dup_f = fl.duplicated(["Prompt_ID", "LABELER_ID"], keep=False); dup_m = main.duplicated(["Prompt_ID", "LABELER_ID"], keep=False)
    m = main[~dup_m].merge(fl[~dup_f], on=["Prompt_ID", "LABELER_ID"], suffixes=("_m", "_f"), how="outer", indicator=True)
    bad = {d: int((pd.to_numeric(m[d + "_m"], errors="coerce") != m[d + "_f"]).sum()) for d in ["PU", "PP", "CS", "GT", "AC", "FW"]}
    print(f"cross-check vs MAIN on non-duplicated rows: unmatched {int((m._merge != 'both').sum())}, value mismatches {bad}")
    fl.to_parquet("form_long.parquet")

if __name__ == "__main__":
    main()
