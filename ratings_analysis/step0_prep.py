# -*- coding: utf-8 -*-
"""Step 0. Build the analysis tables.
Ratings come from the eight raw form tabs (form_long.parquet, full timestamps) so that the earlier of two
submissions is identified reliably; prompt/response text comes from MAIN. Outputs:
 ratings.parquet   900 unique rater-response ratings (first submission kept)
 rerate.parquet    120 rows = 60 (Prompt_ID, rater) pairs rated twice (test-retest)
 responses.parquet 240 responses with canonical text and text features
 out/00_data_summary.txt
"""
import re, numpy as np, pandas as pd, os
from common import DIMS, parse_su, SU_SHORT
os.makedirs("out", exist_ok=True)
xl = pd.ExcelFile(os.environ.get("RATINGS_XLSX", "combined_label_data.xlsx"))
main = xl.parse("MAIN"); main.columns = [c.strip() for c in main.columns]
fl = pd.read_parquet("form_long.parquet")
for d in DIMS:
    fl[d] = pd.to_numeric(fl[d], errors="coerce").astype(int)
log = [f"form rows: {len(fl)}; raters: {fl.LABELER_ID.nunique()}; responses: {fl.Prompt_ID.nunique()}"]

def factors(df):
    df["pipeline"] = df["Prompt_ID"].str.extract(r"_(AR|L)_", expand=False).map({"L": "base_llm", "AR": "autorag"})
    df["persona"] = df["Prompt_ID"].str.contains("Persona", case=False)
    df["steps"] = df["Prompt_ID"].str.contains("Steps", case=False)
    df["base_prompt_id"] = df["Prompt_ID"].str.extract(r"(\d+)$")[0].astype(int)
    df["condition"] = df["pipeline"] + "|" + np.where(df.persona, "persona", "noPersona") + "|" + np.where(df.steps, "steps", "noSteps")
    assert df["pipeline"].notna().all()
    return df
fl = factors(fl)

# ---- duplicates: keep the earliest submission per (Prompt_ID, rater); keep both for test-retest ----
fl = fl.sort_values(["Prompt_ID", "LABELER_ID", "ts"])
dup = fl.duplicated(["Prompt_ID", "LABELER_ID"], keep=False)
rerate = fl[dup].copy()
rerate["submission"] = rerate.groupby(["Prompt_ID", "LABELER_ID"]).cumcount() + 1
ratings = fl.drop_duplicates(["Prompt_ID", "LABELER_ID"], keep="first").copy()
gaps = rerate.groupby(["LABELER_ID"]).apply(lambda g: (g[g.submission == 2].ts.min() - g[g.submission == 1].ts.max()).days)
log.append(f"double submissions: {int(dup.sum())} rows, {rerate.groupby(['Prompt_ID','LABELER_ID']).ngroups} pairs; by rater x condition:\n" +
           rerate[rerate.submission == 1].groupby(["LABELER_ID", "condition"]).size().to_string() + f"\nretest interval (days): {gaps.to_dict()}")
log.append(f"unique ratings kept: {len(ratings)}")

# ---- SU tags ----
for df in (ratings, rerate):
    tags = df["SU"].apply(parse_su)
    for t in SU_SHORT.values():
        df["su_" + t] = tags.apply(lambda L: int(t in L))
    df["su_any"] = (tags.apply(len) > 0).astype(int)
    df["su_n"] = tags.apply(len)

# ---- canonical response text and features (from MAIN) ----
tmp = main.assign(_len=main["Response"].fillna("").astype(str).str.len()).sort_values("_len", ascending=False)
canon = tmp.drop_duplicates("Prompt_ID")[["Prompt_ID", "Prompt", "Response"]].reset_index(drop=True)
canon = factors(canon)
# recover missing base prompt text from the base-LLM twin (same base prompt)
bp_text = canon[canon.condition == "base_llm|noPersona|noSteps"].set_index("base_prompt_id")["Prompt"]
miss_p = canon["Prompt"].fillna("").astype(str).str.strip() == ""
canon.loc[miss_p, "Prompt"] = canon.loc[miss_p, "base_prompt_id"].map(bp_text)
log.append(f"prompt text recovered from base-LLM twin for {int(miss_p.sum())} responses")

DOC_REF = re.compile(r"\b(?:DoDI|DODI|DoDD|DODD|DoDM|DODM|OPNAVINST|SECNAVINST|NAVADMIN|COMNAVSURFORINST|CJCSI|CJCSM|MIL-STD|NTTP|NWP|NTRP|JP|AR|USC|U\.S\.C\.|CFR|Title)\s?\d[\d.\-]*\b|\b\d+\s?(?:U\.S\.C\.|USC|CFR)\b")
CITE_CUE = re.compile(r"\b(according to|based on the|as (?:stated|outlined|described|specified|noted) in|in accordance with|per (?:the )?[A-Z]|the (?:provided |given |retrieved )?(?:passages?|documents?|sources?|context|text|excerpts?)|referenc\w+|cit\w+)\b", re.I)
MISS_ADMIT = re.compile(r"(without referencing the passages|passages? (?:do(?:es)? not|don't|did not) (?:provide|contain|mention|address|cover|include|specify)|not (?:explicitly )?(?:mentioned|addressed|covered|provided|specified|found) in the (?:provided |given )?(?:passages?|documents?|context|text)|no (?:direct |specific )?(?:information|mention|guidance) (?:on|about|regarding) .{0,60} in the (?:passages?|documents?|context)|(?:passages?|documents?|context) (?:provided )?(?:does not|do not|doesn't|don't) (?:directly |specifically )?(?:address|answer|cover|mention)|cannot (?:be )?(?:answer|determin|found)\w* (?:from|based on|using) the (?:provided |given )?(?:passages?|documents?|context)|I (?:don't|do not|cannot|can't) (?:have|find|see) (?:enough |sufficient |any )?(?:information|context))", re.I)
AUTH = re.compile(r"\b(doctrine|doctrinal|regulations?|policy|policies|directives?|instructions?|manuals?|protocols?|standard operating procedures?|SOPs?|rules of engagement|ROE|international (?:humanitarian |maritime )?law|law of (?:armed conflict|the sea)|LOAC|UNCLOS|COLREGS?|SOLAS|Geneva Conventions?|UCMJ|Navy Regulations|OPNAV\w*|NAVSEA|SECNAV\w*|DoD\w*|DOD\w*|CJCS\w*|Joint Publication|JP \d|Title \d+|U\.S\. Code|USC|CFR)\b", re.I)
HEDGE = re.compile(r"\b(may|might|could|possibly|potentially|uncertain|unclear|likely|unlikely|appears?|seems?|suggests?|assum\w+|cannot confirm|not (?:certain|clear|sure)|generally|typically|if applicable|as appropriate)\b", re.I)
_SENT = re.compile(r"(?<=[.!?])\s+(?=[A-Z*\d])|\n+")

def feats(t):
    t = "" if pd.isna(t) else str(t)
    words = re.findall(r"\b[\w'-]+\b", t)
    sents = [s for s in _SENT.split(t) if len(re.findall(r"\w+", s)) >= 3]
    return dict(wc=len(words), n_sent=len(sents), wps=(len(words) / len(sents) if sents else np.nan),
                bold_headers=len(re.findall(r"\*\*[^*]+\*\*", t)),
                numbered=len(re.findall(r"(?:^|\s)\d{1,2}[.)]\s", t)),
                bullets=len(re.findall(r"(?:^|\n)\s*[\*\-•]\s", t)),
                hedges=len(HEDGE.findall(t)), doc_refs=len(DOC_REF.findall(t)), authority=len(AUTH.findall(t)), cite_cues=len(CITE_CUE.findall(t)),
                miss_admit=int(bool(MISS_ADMIT.search(t))), is_empty=int(len(words) == 0),
                refusal=int(bool(re.match(r"^\s*I\s+(?:can(?:no|')t|can’t|am unable|'m unable)\b", t))),
                acronyms=len(set(re.findall(r"\b[A-Z]{3,}\b", t))))
F = pd.DataFrame([feats(t) for t in canon["Response"]])
responses = pd.concat([canon, F], axis=1)
responses["hedges_per100"] = 100 * responses.hedges / responses.wc.replace(0, np.nan)
responses["doc_refs_per100"] = 100 * responses.doc_refs / responses.wc.replace(0, np.nan)
responses["authority_per100"] = 100 * responses.authority / responses.wc.replace(0, np.nan)
responses["log_wc"] = np.log(responses.wc.replace(0, np.nan))
responses["hard_refusal"] = ((responses.refusal == 1) & (responses.wc < 50)).astype(int)
log.append(f"responses: {len(responses)}; missing response text: {int(responses.is_empty.sum())} " +
           f"({', '.join(responses[responses.is_empty==1].Prompt_ID.tolist()[:3])} ... )")
log.append("refusals (response opens with 'I can't/cannot'): total %d\n" % int(responses.refusal.sum()) +
           responses.groupby("condition").refusal.sum().loc[lambda s: s.index].to_string() +
           "\nrefusal Prompt_IDs: " + ", ".join(responses[responses.refusal == 1].Prompt_ID.tolist()))
log.append("text features by condition (mean over responses with text):\n" +
           responses[responses.is_empty == 0].groupby("condition")[["wc", "n_sent", "wps", "bold_headers", "numbered", "bullets", "hedges_per100", "doc_refs", "authority_per100", "cite_cues", "miss_admit", "acronyms"]].mean().round(2).to_string())

featcols = [c for c in responses.columns if c not in ["Prompt_ID", "Prompt", "Response", "pipeline", "persona", "steps", "base_prompt_id", "condition"]]
ratings = ratings.merge(responses[["Prompt_ID"] + featcols], on="Prompt_ID", how="left")
rerate = rerate.merge(responses[["Prompt_ID"] + featcols], on="Prompt_ID", how="left")

cov = ratings.groupby("LABELER_ID")["Prompt_ID"].nunique().sort_values(ascending=False)
rpr = ratings.groupby("Prompt_ID")["LABELER_ID"].nunique()
log.append("responses rated per rater:\n" + cov.to_string())
log.append(f"raters per response: mean {rpr.mean():.2f}, min {rpr.min()}, max {rpr.max()}; distribution {rpr.value_counts().sort_index().to_dict()}")
log.append("rater x condition (unique responses):\n" + pd.crosstab(ratings.LABELER_ID, ratings.condition).to_string())
log.append("rating value counts:\n" + pd.DataFrame({d: ratings[d].value_counts().sort_index() for d in DIMS}).to_string())

ratings.to_parquet("ratings.parquet"); rerate.to_parquet("rerate.parquet"); responses.to_parquet("responses.parquet")
open("out/00_data_summary.txt", "w").write("\n\n".join(log)); print("\n\n".join(log))
