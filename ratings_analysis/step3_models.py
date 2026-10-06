# -*- coding: utf-8 -*-
"""Step 3. Mixed-effects models of the 2x2x2 design.
 A. Primary: rating-level linear mixed model, y ~ rag*per*stp (+-0.5 coding) with crossed random intercepts for
    rater, base prompt and response (statsmodels MixedLM, single group, variance components).
 B. Response-level model (mean rating per response) with base-prompt random intercept (matches the published approach).
 C. Robustness: rater as fixed effect; core five raters only; hard refusals excluded; length-adjusted (log word count).
 D. Variance decomposition (rater / base prompt / response / residual).
 E. Leave-one-rater-out for the three main effects.
Holm adjustment is applied within dimension across the seven design terms."""
import warnings, numpy as np, pandas as pd, statsmodels.formula.api as smf, statsmodels.api as sm
warnings.filterwarnings("ignore")
from common import *
ratings, responses, rerate = load()
ratings = add_codes(ratings); responses = add_codes(responses)
ratings["log_wc_c"] = ratings.log_wc - ratings.log_wc.mean()
log = []
FX = "rag + per + stp + rag_per + rag_stp + per_stp + rag_per_stp"

def fit_rating_level(df, d, extra="", rater="random", reml=True):
    df = df.copy(); df["g"] = 1
    vc = {"bp": "0 + C(base_prompt_id)", "resp": "0 + C(Prompt_ID)"}
    rhs = FX + (" + " + extra if extra else "")
    if rater == "random": vc["rater"] = "0 + C(LABELER_ID)"
    elif rater == "fixed": rhs += " + C(LABELER_ID)"
    m = smf.mixedlm(f"{d} ~ {rhs}", df, groups="g", re_formula="0", vc_formula=vc)
    for method in (["lbfgs", "powell"], ["powell"], ["nm"], ["cg"]):
        try:
            r = m.fit(reml=reml, method=method, maxiter=3000)
            if r.converged: return r
        except Exception: pass
    return r

def _safe_fit(m, reml=True):
    for method in (["lbfgs", "powell"], ["powell"], ["nm"], ["cg"]):
        try:
            r = m.fit(reml=reml, method=method, maxiter=3000)
            if r.converged: return r
        except Exception: pass
    return r

def fit_response_level(df, d, extra=""):
    rhs = FX + (" + " + extra if extra else "")
    m = smf.mixedlm(f"{d} ~ {rhs}", df, groups="base_prompt_id")
    for meth in (["powell"], ["nm"], ["cg"]):
        try:
            r = m.fit(reml=True, method=meth, maxiter=3000)
            if r.converged: return r
        except Exception: pass
    return r

def coef_table(res, d, model, terms=TERMS):
    rows = []
    ps = [res.pvalues[t] for t in terms]; adj = holm(ps)
    for t, pa in zip(terms, adj):
        rows.append(dict(model=model, dim=d, term=t, est=res.params[t], se=res.bse[t], lo=res.conf_int().loc[t, 0], hi=res.conf_int().loc[t, 1],
                         z=res.tvalues[t], p=res.pvalues[t], p_holm=pa))
    return rows

def vc_summary(res):
    out = {}
    for k in ["rater", "bp", "resp"]:
        if k in res.vcomp_names if hasattr(res, "vcomp_names") else False: pass
    names = list(res.model.exog_vc.names) if hasattr(res.model, "exog_vc") else []
    for nm, v in zip(names, res.vcomp): out[nm] = float(v)
    out["resid"] = float(res.scale)
    return out

# ---------- A. primary rating-level model ----------
tabs = []; vcs = []
for d in DIMS:
    r = fit_rating_level(ratings, d)
    tabs += coef_table(r, d, "A_rating_crossed")
    v = vc_summary(r); v["dim"] = d; v["model"] = "A"; vcs.append(v)
    log.append(f"[A] {d}: converged={r.converged}; vc={ {k: round(x,3) for k,x in v.items() if k not in ('dim','model')} }")
prim = pd.DataFrame([t for t in tabs if t["model"] == "A_rating_crossed"])

# ---------- B. response-level model ----------
resp_mean = ratings.groupby(["Prompt_ID", "base_prompt_id", "condition", "pipeline", "persona", "steps"])[DIMS].mean().reset_index()
resp_mean = add_codes(resp_mean).merge(responses[["Prompt_ID", "log_wc", "wc", "wps", "hard_refusal", "refusal", "is_empty", "doc_refs", "cite_cues", "hedges_per100", "miss_admit"]], on="Prompt_ID")
resp_mean["log_wc_c"] = resp_mean.log_wc - resp_mean.log_wc.mean()
for d in DIMS:
    r = fit_response_level(resp_mean, d); tabs += coef_table(r, d, "B_response_level")

# ---------- C. robustness ----------
for d in DIMS:
    tabs += coef_table(fit_rating_level(ratings, d, rater="fixed"), d, "C1_rater_fixed")
    tabs += coef_table(fit_rating_level(ratings[ratings.LABELER_ID.isin(["L02", "L03", "L05", "L07", "L08"])], d), d, "C2_core5")
    tabs += coef_table(fit_rating_level(ratings[ratings.hard_refusal == 0], d), d, "C3_no_refusals")
    tabs += coef_table(fit_rating_level(ratings[ratings.refusal == 0], d), d, "C3b_no_refusal_openings")
    sub = ratings[ratings.is_empty == 0]
    tabs += coef_table(fit_rating_level(sub, d, extra="log_wc_c"), d, "C4_length_adj", terms=TERMS + ["log_wc_c"])
    tabs += coef_table(fit_rating_level(ratings, d, reml=False), d, "C5_ML")
tab = pd.DataFrame(tabs); tab.to_csv(f"{OUT}/03_model_coefficients.csv", index=False)

def show(model):
    t = tab[tab.model == model].pivot(index="term", columns="dim", values=["est", "p_holm"])
    s = pd.DataFrame(index=[x for x in TERMS + ["log_wc_c"] if x in t.index])
    for d in DIMS:
        s[d] = [f"{t.loc[i, ('est', d)]:+.2f}{'***' if t.loc[i, ('p_holm', d)] < .001 else '**' if t.loc[i, ('p_holm', d)] < .01 else '*' if t.loc[i, ('p_holm', d)] < .05 else ''}" for i in s.index]
    return s.to_string()
for m in ["A_rating_crossed", "B_response_level", "C1_rater_fixed", "C2_core5", "C3_no_refusals", "C3b_no_refusal_openings", "C4_length_adj", "C5_ML"]:
    log.append(f"Model {m} (estimates in rating points; stars = Holm-adjusted p within dimension: * <.05, ** <.01, *** <.001):\n" + show(m))

# Bonferroni across all 42 primary tests (6 dimensions x 7 terms), for the multiple-testing note
prim["p_bonf42"] = (prim.p * 42).clip(upper=1.0)
log.append("Primary model: terms surviving Bonferroni across all 42 tests (p*42 < .05):\n" + prim[prim.p_bonf42 < .05][["dim", "term", "est", "p", "p_bonf42"]].round(4).to_string())
# full primary table with CIs
pt = prim.copy(); pt["cell"] = pt.apply(lambda r: f"{r.est:+.2f} [{r.lo:+.2f}, {r.hi:+.2f}] p={fmt_p(r.p)} (Holm {fmt_p(r.p_holm)})", axis=1)
log.append("Primary model, full detail:\n" + pt.pivot(index="term", columns="dim", values="cell").loc[TERMS].to_string())

# ---------- D. variance decomposition ----------
rows = []
for d in DIMS:
    df = ratings.copy(); df["g"] = 1
    for nm, f in [("null", f"{d} ~ 1"), ("design", f"{d} ~ {FX}")]:
        m = smf.mixedlm(f, df, groups="g", re_formula="0", vc_formula={"rater": "0 + C(LABELER_ID)", "bp": "0 + C(base_prompt_id)", "resp": "0 + C(Prompt_ID)"})
        r = _safe_fit(m)
        v = vc_summary(r); tot = sum(v.values())
        rows.append(dict(dim=d, model=nm, **{k: v[k] for k in v}, **{k + "_share": v[k] / tot for k in v}))
vd = pd.DataFrame(rows); vd.to_csv(f"{OUT}/03_variance_decomposition.csv", index=False)
log.append("Variance decomposition (share of total rating variance; 'null' = no fixed effects, 'design' = with the 2x2x2 terms):\n" +
           vd.set_index(["dim", "model"])[["rater_share", "bp_share", "resp_share", "resid_share"]].round(3).to_string())

# ---------- E. leave-one-rater-out on the primary model ----------
rows = []
for rid in sorted(ratings.LABELER_ID.unique()):
    sub = ratings[ratings.LABELER_ID != rid]
    for d in DIMS:
        r = fit_rating_level(sub, d)
        for t in ["rag", "per", "stp", "rag_per"]:
            rows.append(dict(dropped=rid, dim=d, term=t, est=r.params[t], lo=r.conf_int().loc[t, 0], hi=r.conf_int().loc[t, 1], p=r.pvalues[t]))
loro = pd.DataFrame(rows); loro.to_csv(f"{OUT}/03_leave_one_rater_out.csv", index=False)
summ = loro.groupby(["dim", "term"]).agg(est_min=("est", "min"), est_max=("est", "max"), n_sig=("p", lambda p: int((p < .05).sum()))).reset_index()
log.append("Leave-one-rater-out (primary model): range of estimates and number of the 8 refits with p<.05 (unadjusted):\n" + summ.pivot(index="term", columns="dim", values=["est_min", "est_max", "n_sig"]).round(2).to_string())

# ---------- simple-effects: RAG effect within each persona x steps cell (primary model re-parameterised) ----------
rows = []
for (p, s) in [(False, False), (False, True), (True, False), (True, True)]:
    sub = ratings[(ratings.persona == p) & (ratings.steps == s)].copy(); sub["g"] = 1
    for d in DIMS:
        m = smf.mixedlm(f"{d} ~ rag", sub, groups="g", re_formula="0", vc_formula={"rater": "0 + C(LABELER_ID)", "bp": "0 + C(base_prompt_id)", "resp": "0 + C(Prompt_ID)"})
        r = _safe_fit(m)
        rows.append(dict(persona=p, steps=s, dim=d, est=r.params["rag"], lo=r.conf_int().loc["rag", 0], hi=r.conf_int().loc["rag", 1], p=r.pvalues["rag"]))
se = pd.DataFrame(rows); se.to_csv(f"{OUT}/03_simple_effects_rag.csv", index=False)
se["cell"] = se.apply(lambda r: f"{r.est:+.2f} [{r.lo:+.2f},{r.hi:+.2f}] p={fmt_p(r.p)}", axis=1)
log.append("Simple effect of RAG (AutoRAG minus base LLM) within each persona x steps cell, rater/base-prompt/response random intercepts:\n" +
           se.pivot(index=["persona", "steps"], columns="dim", values="cell").to_string())
# persona simple effect within pipeline
rows = []
for pipe in ["base_llm", "autorag"]:
    sub = ratings[ratings.pipeline == pipe].copy(); sub["g"] = 1
    for d in DIMS:
        m = smf.mixedlm(f"{d} ~ per + stp + per_stp", sub, groups="g", re_formula="0", vc_formula={"rater": "0 + C(LABELER_ID)", "bp": "0 + C(base_prompt_id)", "resp": "0 + C(Prompt_ID)"})
        r = _safe_fit(m)
        rows.append(dict(pipeline=pipe, dim=d, est=r.params["per"], lo=r.conf_int().loc["per", 0], hi=r.conf_int().loc["per", 1], p=r.pvalues["per"]))
sp = pd.DataFrame(rows); sp.to_csv(f"{OUT}/03_simple_effects_persona.csv", index=False)
sp["cell"] = sp.apply(lambda r: f"{r.est:+.2f} [{r.lo:+.2f},{r.hi:+.2f}] p={fmt_p(r.p)}", axis=1)
log.append("Simple effect of persona within pipeline (rater-adjusted):\n" + sp.pivot(index="pipeline", columns="dim", values="cell").to_string())

open(f"{OUT}/03_models.txt", "w").write("\n\n".join(log)); print("\n\n".join(log))
