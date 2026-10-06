# -*- coding: utf-8 -*-
"""Step 5. Secondary analyses.
 1. Source-of-uncertainty tags by condition (rating level), with rater-adjusted logistic mixed models for 'no source' and 'vague'.
 2. Surface quality (mean of PP, CS, AC) versus grounding transparency, and the gap by condition.
 3. Per-base-prompt RAG effect heterogeneity (paired differences within persona x steps cell).
 4. Response text features versus ratings (rater-adjusted): document references, refusal, word count, hedging, headers.
 5. Refusals: how raters scored them, and the RAG effect with and without them.
 6. Qualitative notes: counts and example comments by condition (for the discussion).
"""
import warnings, re, numpy as np, pandas as pd, statsmodels.formula.api as smf, matplotlib
warnings.filterwarnings("ignore"); matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import chi2_contingency, spearmanr, wilcoxon
from common import *
ratings, responses, rerate = load(); ratings = add_codes(ratings)
log = []
SU = ["su_no_source", "su_vague", "su_unexpected", "su_multi_interp", "su_conflict", "su_other", "su_any"]

# ---------- 1. SU tags ----------
su_cond = ratings.groupby("condition")[SU].mean().loc[COND_ORDER] * 100
su_pipe = ratings.groupby("pipeline")[SU].mean() * 100; su_per = ratings.groupby("persona")[SU].mean() * 100
su_cond.to_csv(f"{OUT}/05_su_by_condition.csv")
log.append("Source-of-uncertainty tags, % of ratings flagged, by condition:\n" + su_cond.round(1).to_string() +
           "\nby pipeline:\n" + su_pipe.round(1).to_string() + "\nby persona:\n" + su_per.round(1).to_string())
# how does tagging relate to PU?
log.append("PU by whether any SU tag was given: " + ratings.groupby("su_any").PU.mean().round(2).to_dict().__str__() +
           f"; share of ratings with PU>=3 that carry a tag: {ratings[ratings.PU>=3].su_any.mean():.2f}; PU<=2 with a tag: {ratings[ratings.PU<=2].su_any.mean():.2f}")
# rater-adjusted linear probability mixed models (same crossed random intercepts as the primary analysis); effects in percentage points
rows = []
for tag in ["su_no_source", "su_vague", "su_unexpected", "su_any"]:
    df = ratings.copy(); df["g"] = 1; df[tag] = 100.0 * df[tag]
    m = smf.mixedlm(f"{tag} ~ rag + per + stp + rag_per + rag_stp + per_stp + rag_per_stp", df, groups="g", re_formula="0",
                    vc_formula={"rater": "0 + C(LABELER_ID)", "bp": "0 + C(base_prompt_id)", "resp": "0 + C(Prompt_ID)"})
    for meth in (["lbfgs", "powell"], ["powell"], ["nm"]):
        try:
            r = m.fit(reml=True, method=meth, maxiter=3000)
            if r.converged: break
        except Exception: pass
    ps = [r.pvalues[t] for t in TERMS]; adj = holm(ps)
    for t, pa in zip(TERMS, adj):
        rows.append(dict(tag=tag, term=t, est=r.params[t], lo=r.conf_int().loc[t, 0], hi=r.conf_int().loc[t, 1], p=r.pvalues[t], p_holm=pa))
su_or = pd.DataFrame(rows); su_or.to_csv(f"{OUT}/05_su_lpm.csv", index=False)
su_or["cell"] = su_or.apply(lambda r: f"{r.est:+.1f} [{r.lo:+.1f},{r.hi:+.1f}] p={fmt_p(r.p)} H={fmt_p(r.p_holm)}", axis=1)
log.append("Rater-adjusted linear probability mixed models for tag presence (percentage points; Holm within tag):\n" +
           su_or.pivot(index="term", columns="tag", values="cell").reindex(TERMS).to_string())

# ---------- 2. surface quality vs grounding ----------
ratings["surface"] = ratings[["PP", "CS", "AC"]].mean(axis=1)
ratings["gap"] = ratings.surface - ratings.GT
resp = ratings.groupby(["Prompt_ID", "condition", "pipeline", "persona", "steps", "base_prompt_id"])[["surface", "gap"] + DIMS].mean().reset_index()
resp = resp.merge(responses[["Prompt_ID", "wc", "log_wc", "doc_refs", "cite_cues", "hedges_per100", "bold_headers", "numbered", "refusal", "hard_refusal", "is_empty", "miss_admit", "acronyms"]], on="Prompt_ID")
resp.to_parquet(f"{OUT}/05_response_table.parquet")
g = resp.groupby("condition")[["surface", "GT", "gap"]].mean().loc[COND_ORDER]
log.append("Surface quality (mean PP,CS,AC) vs grounding transparency, response level, by condition:\n" + g.round(2).to_string())
rows = []
for d in ["gap"]:
    df = ratings.copy(); df["g"] = 1
    m = smf.mixedlm("gap ~ rag + per + stp + rag_per + rag_stp + per_stp + rag_per_stp", df, groups="g", re_formula="0",
                    vc_formula={"rater": "0 + C(LABELER_ID)", "bp": "0 + C(base_prompt_id)", "resp": "0 + C(Prompt_ID)"})
    for meth in (["lbfgs", "powell"], ["powell"], ["nm"]):
        try:
            r = m.fit(reml=True, method=meth, maxiter=3000)
            if r.converged: break
        except Exception: pass
    for t in TERMS:
        rows.append(dict(term=t, est=r.params[t], lo=r.conf_int().loc[t, 0], hi=r.conf_int().loc[t, 1], p=r.pvalues[t]))
gap = pd.DataFrame(rows); gap["p_holm"] = holm(gap.p); gap.to_csv(f"{OUT}/05_gap_model.csv", index=False)
log.append("Surface-minus-grounding gap, rater-adjusted mixed model (positive = polished but ungrounded):\n" +
           "\n".join(f"{r.term}: {r.est:+.2f} [{r.lo:+.2f},{r.hi:+.2f}] p={fmt_p(r.p)} Holm={fmt_p(r.p_holm)}" for _, r in gap.iterrows()))
log.append(f"Correlation surface vs GT across responses: r={resp.surface.corr(resp.GT):.2f}; within base LLM r={resp[resp.pipeline=='base_llm'].surface.corr(resp[resp.pipeline=='base_llm'].GT):.2f}; within AutoRAG r={resp[resp.pipeline=='autorag'].surface.corr(resp[resp.pipeline=='autorag'].GT):.2f}")
# share of responses rated polished (surface>=4) but ungrounded (GT<=2)
resp["polished_ungrounded"] = ((resp.surface >= 4) & (resp.GT <= 2.5)).astype(int)
log.append("Share of responses with surface>=4 and GT<=2.5 by condition:\n" + (resp.groupby("condition").polished_ungrounded.mean().loc[COND_ORDER] * 100).round(1).to_string())

# ---------- 3. per-base-prompt heterogeneity of the RAG effect ----------
wide = resp.pivot_table(index=["base_prompt_id", "persona", "steps"], columns="pipeline", values=DIMS)
rows = []
for d in DIMS:
    diff = (wide[(d, "autorag")] - wide[(d, "base_llm")]).reset_index().rename(columns={0: "diff"})
    diff.columns = ["base_prompt_id", "persona", "steps", "diff"]
    per_bp = diff.groupby("base_prompt_id")["diff"].mean()
    rows.append(dict(dim=d, mean=per_bp.mean(), sd=per_bp.std(ddof=1), pct_positive=100 * (per_bp > 0).mean(), min=per_bp.min(), max=per_bp.max(),
                     wilcoxon_p=wilcoxon(per_bp).pvalue))
het = pd.DataFrame(rows).set_index("dim"); het.to_csv(f"{OUT}/05_rag_heterogeneity.csv")
log.append("RAG effect per base prompt (AutoRAG minus base LLM, averaged over the four prompt-wrapper cells; n=30 base prompts):\n" + het.round(3).to_string())
# figure: per-prompt RAG effect for GT and PU
fig, axes = plt.subplots(1, 2, figsize=(10, 4.2), sharey=False)
for ax, d in zip(axes, ["GT", "PU"]):
    diff = (wide[(d, "autorag")] - wide[(d, "base_llm")]).groupby(level="base_prompt_id").mean().sort_values()
    ax.barh([str(i) for i in diff.index], diff.values, color=np.where(diff.values > 0, "#1f77b4", "#d62728"))
    ax.axvline(0, color="k", lw=0.8); ax.set_xlabel(f"AutoRAG minus base LLM, {d} (mean over 4 wrapper cells)"); ax.set_ylabel("base prompt"); ax.tick_params(axis="y", labelsize=6)
fig.tight_layout(); fig.savefig(f"{OUT}/fig3_rag_by_prompt.pdf"); fig.savefig(f"{OUT}/fig3_rag_by_prompt.png", dpi=200)

# ---------- 4. text features vs ratings (rater-adjusted, responses with text) ----------
sub = ratings[ratings.is_empty == 0].copy(); sub["g"] = 1
sub["log_wc_c"] = sub.log_wc - sub.log_wc.mean(); sub["auth_c"] = sub.authority_per100 - sub.authority_per100.mean()
sub["headers_c"] = sub.bold_headers - sub.bold_headers.mean(); sub["hedges_c"] = sub.hedges_per100 - sub.hedges_per100.mean()
rows = []
for d in DIMS:
    m = smf.mixedlm(f"{d} ~ log_wc_c + auth_c + hard_refusal + headers_c + hedges_c + rag + per + stp", sub, groups="g", re_formula="0",
                    vc_formula={"rater": "0 + C(LABELER_ID)", "bp": "0 + C(base_prompt_id)", "resp": "0 + C(Prompt_ID)"})
    for meth in (["lbfgs", "powell"], ["powell"], ["nm"]):
        try:
            r = m.fit(reml=True, method=meth, maxiter=3000)
            if r.converged: break
        except Exception: pass
    for t in ["log_wc_c", "auth_c", "hard_refusal", "headers_c", "hedges_c", "rag", "per", "stp"]:
        rows.append(dict(dim=d, term=t, est=r.params[t], lo=r.conf_int().loc[t, 0], hi=r.conf_int().loc[t, 1], p=r.pvalues[t]))
tf = pd.DataFrame(rows); tf.to_csv(f"{OUT}/05_text_features_model.csv", index=False)
tf["cell"] = tf.apply(lambda r: f"{r.est:+.2f} [{r.lo:+.2f},{r.hi:+.2f}] {'*' if r.p < .05 else ''}", axis=1)
log.append("Ratings regressed on response features plus design main effects (rater, base prompt, response random intercepts; * p<.05 unadjusted):\n" +
           tf.pivot(index="term", columns="dim", values="cell").loc[["log_wc_c", "auth_c", "hard_refusal", "headers_c", "hedges_c", "rag", "per", "stp"]].to_string())
log.append("Authority-vocabulary mentions per 100 words and %% with a numbered directive citation, by condition:\n" + responses[responses.is_empty == 0].groupby("condition").agg(auth=("authority_per100", "mean"), cite=("doc_refs", lambda s: 100 * (s > 0).mean())).loc[COND_ORDER].round(2).to_string())

# ---------- 5. refusals ----------
ref = resp[resp.hard_refusal == 1]; nonref = resp[(resp.hard_refusal == 0) & (resp.pipeline == "base_llm")]
log.append(f"Hard refusals: {len(ref)} base-LLM responses. Mean ratings on refusals vs other base-LLM responses:\n" +
           pd.DataFrame({"refusal": ref[DIMS].mean(), "other_base": nonref[DIMS].mean(), "autorag": resp[resp.pipeline == 'autorag'][DIMS].mean()}).round(2).T.to_string())
log.append("Refusals by condition: " + responses.groupby("condition").hard_refusal.sum().loc[COND_ORDER].to_dict().__str__() +
           f"; refusal rate base LLM without persona {responses[(responses.pipeline=='base_llm')&(~responses.persona)].hard_refusal.mean():.2f} vs with persona {responses[(responses.pipeline=='base_llm')&(responses.persona)].hard_refusal.mean():.2f}")
# SU tags on refusals
log.append("SU tags on ratings of refusals (%): " + (ratings[ratings.hard_refusal == 1][SU].mean() * 100).round(1).to_dict().__str__())

# ---------- 6. qualitative notes ----------
notes = ratings[["Prompt_ID", "LABELER_ID", "condition", "pipeline", "persona", "steps", "PU", "GT", "UN", "GN", "hard_refusal", "doc_refs"]].copy()
notes["has_UN"] = notes.UN.notna() & (notes.UN.astype(str).str.strip() != ""); notes["has_GN"] = notes.GN.notna() & (notes.GN.astype(str).str.strip() != "")
log.append("Free-text notes: uncertainty notes in %d ratings, general notes in %d ratings; by pipeline (%% with a general note): %s" %
           (notes.has_UN.sum(), notes.has_GN.sum(), (notes.groupby("pipeline").has_GN.mean() * 100).round(1).to_dict()))
# keyword coding of notes
KW = {"source/grounding": r"\b(source|sources|cite|cited|citation|reference|references|grounded|grounding|where .{0,20} from|policy|policies|directive|regulation|document)\b",
      "vague/generic": r"\b(vague|generic|general|broad|surface|superficial|not specific|unspecific|boilerplate|fluff)\b",
      "structure/clarity": r"\b(structure|structured|organized|clear|clearly|easy to (?:read|follow|understand)|format|bullet|numbered|concise|step)\b",
      "refusal/misread": r"\b(refus|declin|can't fulfill|cannot fulfill|didn't answer|did not answer|misread|misunderstood|misinterpret|illegal|no response|unhelpful)\b",
      "actionable/specific": r"\b(actionable|specific|concrete|detailed|practical|useful|helpful|thorough)\b",
      "acronym/jargon": r"\b(acronym|acronyms|jargon|abbreviation|terminology|didn't know what|did not know what|unfamiliar)\b"}
all_notes = pd.concat([notes[notes.has_UN].assign(text=lambda x: x.UN.astype(str)), notes[notes.has_GN].assign(text=lambda x: x.GN.astype(str))])
for k, pat in KW.items():
    all_notes[k] = all_notes.text.str.contains(pat, case=False, regex=True)
kw = all_notes.groupby("pipeline")[list(KW)].mean() * 100
log.append("Keyword coding of free-text notes (% of notes mentioning), by pipeline:\n" + kw.round(1).to_string())
kw2 = all_notes.groupby("persona")[list(KW)].mean() * 100
log.append("... by persona:\n" + kw2.round(1).to_string())
all_notes[["Prompt_ID", "LABELER_ID", "condition", "PU", "GT", "text"] + list(KW)].to_csv(f"{OUT}/05_notes_coded.csv", index=False)
# example quotes (short, informative) per pipeline
ex = []
for pipe in ["base_llm", "autorag"]:
    s = all_notes[(all_notes.pipeline == pipe) & (all_notes.text.str.len().between(40, 220))]
    for k in ["source/grounding", "vague/generic", "actionable/specific", "acronym/jargon", "refusal/misread"]:
        q = s[s[k]].head(3)
        for _, r in q.iterrows(): ex.append(f"[{pipe} | {r.condition} | {r.LABELER_ID} | PU={r.PU} GT={r.GT} | {k}] {r.text.strip()}")
log.append("Example rater comments:\n" + "\n".join(ex))
open(f"{OUT}/05_secondary.txt", "w").write("\n\n".join(log)); print("\n\n".join(log))
