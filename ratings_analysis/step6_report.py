# -*- coding: utf-8 -*-
"""Step 6. Assemble LaTeX tables (booktabs) and the remaining figures from the step outputs."""
import os, numpy as np, pandas as pd, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from common import *
ratings, responses, rerate = load()
RMAP = {"L01": "R1", "L02": "R2", "L03": "R3", "L05": "R4", "L06": "R5", "L07": "R6", "L08": "R7", "L09": "R8"}
ratings["LABELER_ID"] = ratings.LABELER_ID.map(RMAP)
T = {}
COND_TEX = {"base_llm|noPersona|noSteps": "Base LLM, none", "base_llm|noPersona|steps": "Base LLM, steps", "base_llm|persona|noSteps": "Base LLM, persona",
            "base_llm|persona|steps": "Base LLM, persona+steps", "autorag|noPersona|noSteps": "AutoRAG, none", "autorag|noPersona|steps": "AutoRAG, steps",
            "autorag|persona|noSteps": "AutoRAG, persona", "autorag|persona|steps": "AutoRAG, persona+steps"}
def stars(p): return "$^{***}$" if p < .001 else "$^{**}$" if p < .01 else "$^{*}$" if p < .05 else ""

# ---- Table: rater allocation ----
ct = pd.crosstab(ratings.LABELER_ID, ratings.condition).reindex(columns=COND_ORDER)
ref = responses.groupby("condition").hard_refusal.sum().reindex(COND_ORDER)
lines = ["\\begin{tabular}{l" + "c" * 8 + "r}", "\\toprule", "Rater & " + " & ".join(["\\rotatebox{60}{" + COND_TEX[c].replace(", ", " / ") + "}" for c in COND_ORDER]) + " & Total \\\\", "\\midrule"]
for rid, row in ct.iterrows():
    lines.append(f"{rid} & " + " & ".join(str(v) if v else "--" for v in row.values) + f" & {row.sum()} \\\\")
lines += ["\\midrule", "Raters per response & " + " & ".join(str(int((ct[c] > 0).sum())) for c in COND_ORDER) + " & \\\\",
          "Hard refusals (of 30) & " + " & ".join(str(int(ref[c])) for c in COND_ORDER) + " & 19 \\\\", "\\bottomrule", "\\end{tabular}"]
T["rater_allocation"] = "\n".join(lines)

# ---- Table: descriptives + reliability ----
desc = pd.read_csv(f"{OUT}/01_descriptives.csv", index_col=0); ka = pd.read_csv(f"{OUT}/02_krippendorff.csv", index_col=0)
icc = pd.read_csv(f"{OUT}/02_icc_by_condition.csv", index_col=0); pw = pd.read_csv(f"{OUT}/02_pairwise.csv"); vd = pd.read_csv(f"{OUT}/03_variance_decomposition.csv", keep_default_na=False)
lines = ["\\begin{tabular}{llcccccc}", "\\toprule", "Dim. & Construct & Mean (SD) & \\% $\\leq$2 & \\% $\\geq$4 & $\\alpha_{\\mathrm{ord}}$ [95\\% CI] & ICC(2,1) range & Rater / response variance share \\\\", "\\midrule"]
for d in DIMS:
    v = vd[(vd.dim == d) & (vd.model == "null")].iloc[0]
    lines.append(f"{d} & {DIM_NAMES[d]} & {desc.loc[d,'mean']:.2f} ({desc.loc[d,'sd']:.2f}) & {desc.loc[d,'pct_low']:.0f} & {desc.loc[d,'pct_high']:.0f} & "
                 f"{ka.loc[d,'alpha_ordinal']:.2f} [{ka.loc[d,'lo']:.2f}, {ka.loc[d,'hi']:.2f}] & {icc[d+'_ICC2_1'].min():.2f} to {icc[d+'_ICC2_1'].max():.2f} & {100*v.rater_share:.0f}\\% / {100*v.resp_share:.0f}\\% \\\\")
lines += ["\\bottomrule", "\\end{tabular}"]
T["descriptives"] = "\n".join(lines)

# ---- Table: cell means ----
cells = pd.read_csv(f"{OUT}/01_cell_means.csv", index_col=0)
lines = ["\\begin{tabular}{l" + "c" * 6 + "}", "\\toprule", "Condition & " + " & ".join(DIMS) + " \\\\", "\\midrule"]
for c in COND_ORDER:
    lines.append(COND_TEX[c] + " & " + " & ".join(f"{cells.loc[c,d]:.2f} [{cells.loc[c,d+'_lo']:.2f}, {cells.loc[c,d+'_hi']:.2f}]" for d in DIMS) + " \\\\")
marg = pd.read_csv(f"{OUT}/01_marginal_means.csv")
lines.append("\\midrule")
for fac, lev, lab in [("pipeline", "base_llm", "Base LLM (all)"), ("pipeline", "autorag", "AutoRAG (all)"), ("persona", "False", "No persona"), ("persona", "True", "Persona"), ("steps", "False", "No steps"), ("steps", "True", "Steps")]:
    r = marg[(marg.factor == fac) & (marg.level == lev)].iloc[0]
    lines.append(lab + " & " + " & ".join(f"{r[d]:.2f}" for d in DIMS) + " \\\\")
lines += ["\\bottomrule", "\\end{tabular}"]
T["cell_means"] = "\n".join(lines)

# ---- Table: primary model ----
coef = pd.read_csv(f"{OUT}/03_model_coefficients.csv")
A = coef[coef.model == "A_rating_crossed"].set_index(["term", "dim"])
lines = ["\\begin{tabular}{l" + "c" * 6 + "}", "\\toprule", "Term & " + " & ".join(DIMS) + " \\\\", "\\midrule"]
for t in TERMS:
    lines.append(TERM_LABELS[t] + " & " + " & ".join(f"{A.loc[(t,d),'est']:+.2f} [{A.loc[(t,d),'lo']:+.2f}, {A.loc[(t,d),'hi']:+.2f}]{stars(A.loc[(t,d),'p_holm'])}" for d in DIMS) + " \\\\")
vc = vd[vd.model == "design"].set_index("dim")
lines += ["\\midrule", "Rater SD & " + " & ".join(f"{np.sqrt(vc.loc[d,'rater']):.2f}" for d in DIMS) + " \\\\",
          "Base-prompt SD & " + " & ".join(f"{np.sqrt(vc.loc[d,'bp']):.2f}" for d in DIMS) + " \\\\",
          "Response SD & " + " & ".join(f"{np.sqrt(vc.loc[d,'resp']):.2f}" for d in DIMS) + " \\\\",
          "Residual SD & " + " & ".join(f"{np.sqrt(vc.loc[d,'resid']):.2f}" for d in DIMS) + " \\\\", "\\bottomrule", "\\end{tabular}"]
T["primary_model"] = "\n".join(lines)

# ---- Table: robustness of the three main effects across specifications ----
ordf = f"{OUT}/04_ordinal_coefficients.csv"
ordc = pd.read_csv(ordf).set_index(["term", "dim"]) if os.path.exists(ordf) else None
specs = [("A_rating_crossed", "A: rating level, crossed random intercepts (primary)"), ("C1_rater_fixed", "Rater fixed effects"), ("C2_core5", "Core five raters only"),
         ("C3_no_refusals", "Hard refusals excluded"), ("C3b_no_refusal_openings", "All 23 refusal openings excluded"), ("C4_length_adj", "Adjusted for log word count"), ("B_response_level", "Response-level means")]
lines = ["\\begin{tabular}{ll" + "c" * 6 + "}", "\\toprule", "Effect & Specification & " + " & ".join(DIMS) + " \\\\", "\\midrule"]
for t in ["rag", "per", "stp"]:
    first = True
    for m, lab in specs:
        S = coef[coef.model == m].set_index(["term", "dim"])
        lines.append((TERM_LABELS[t] if first else "") + " & " + lab + " & " + " & ".join(f"{S.loc[(t,d),'est']:+.2f}{stars(S.loc[(t,d),'p_holm'])}" for d in DIMS) + " \\\\"); first = False
    if ordc is not None:
        lines.append(" & Cumulative-link (logit; 94\\% HDI excludes 0 marked $^{\\dagger}$) & " + " & ".join((f"{ordc.loc[(t,d),'mean']:+.2f}" + ("$^{\\dagger}$" if (ordc.loc[(t,d),'hdi_lo'] > 0 or ordc.loc[(t,d),'hdi_hi'] < 0) else "")) if (t, d) in ordc.index else "" for d in DIMS) + " \\\\")
    if t != "stp": lines.append("\\midrule")
lines += ["\\bottomrule", "\\end{tabular}"]
T["robustness"] = "\n".join(lines)

# ---- Table: text features and refusals by condition ----
tx = responses.copy(); tx = tx[tx.is_empty == 0]
g = tx.groupby("condition").agg(wc=("wc", "mean"), wc_sd=("wc", "std"), headers=("bold_headers", "mean"), numbered=("numbered", "mean"), docref=("doc_refs", lambda s: 100 * (s > 0).mean()), auth=("authority_per100", "mean"), hedges=("hedges_per100", "mean")).reindex(COND_ORDER)
ref = responses.groupby("condition").hard_refusal.sum().reindex(COND_ORDER); n_text = responses[responses.is_empty == 0].groupby("condition").size().reindex(COND_ORDER)
lines = ["\\begin{tabular}{lrrrrrrr}", "\\toprule", "Condition & $n$ with text & Words, mean (SD) & Bold headers & Numbered items & Authority terms per 100 words & Hedges per 100 words & Hard refusals \\\\", "\\midrule"]
for c in COND_ORDER:
    lines.append(f"{COND_TEX[c]} & {n_text[c]} & {g.loc[c,'wc']:.0f} ({g.loc[c,'wc_sd']:.0f}) & {g.loc[c,'headers']:.1f} & {g.loc[c,'numbered']:.1f} & {g.loc[c,'auth']:.2f} & {g.loc[c,'hedges']:.2f} & {int(ref[c])} \\\\")
lines += ["\\bottomrule", "\\end{tabular}"]
T["text_features"] = "\n".join(lines)

# ---- Table: SU tags ----
su = pd.read_csv(f"{OUT}/05_su_by_condition.csv", index_col=0); lpm = pd.read_csv(f"{OUT}/05_su_lpm.csv").set_index(["tag", "term"])
tags = [("su_no_source", "No source / unclear grounding"), ("su_vague", "Vague or ambiguous language"), ("su_unexpected", "Unexpected answer"), ("su_multi_interp", "Multiple interpretations"), ("su_conflict", "Conflicting information"), ("su_any", "Any tag")]
lines = ["\\begin{tabular}{l" + "c" * 8 + "ccc}", "\\toprule", "Tag & " + " & ".join("\\rotatebox{60}{" + COND_TEX[c].replace(", ", " / ") + "}" for c in COND_ORDER) + " & RAG & Persona & Steps \\\\", "\\midrule"]
for k, lab in tags:
    eff = " & ".join(f"{lpm.loc[(k,t),'est']:+.0f}{stars(lpm.loc[(k,t),'p_holm'])}" if (k, t) in lpm.index else "" for t in ["rag", "per", "stp"])
    lines.append(lab + " & " + " & ".join(f"{su.loc[c,k]:.0f}" for c in COND_ORDER) + " & " + eff + " \\\\")
lines += ["\\bottomrule", "\\end{tabular}"]
T["su_tags"] = "\n".join(lines)

# ---- Table: test-retest ----
tr = pd.read_csv(f"{OUT}/02_test_retest.csv")
lines = ["\\begin{tabular}{llc" + "c" * 6 + "}", "\\toprule", "Rater & Condition (interval) & Statistic & " + " & ".join(DIMS) + " \\\\", "\\midrule"]
for _, r in tr.iterrows():
    lines.append(f"{RMAP[r.rater]} & {COND_TEX[r.condition]} ({r.interval_days} d) & $\\kappa_w$ & " + " & ".join(f"{r[d+'_wkappa']:.2f}" for d in DIMS) + " \\\\")
    lines.append(f" &  & within 1 point & " + " & ".join(f"{100*r[d+'_within1']:.0f}\\%" for d in DIMS) + " \\\\")
    lines.append(f" &  & mean shift & " + " & ".join(f"{r[d+'_shift']:+.2f}" for d in DIMS) + " \\\\")
lines += ["\\bottomrule", "\\end{tabular}"]
T["test_retest"] = "\n".join(lines)

with open(f"{OUT}/tables.tex", "w") as f:
    for k, v in T.items():
        f.write(f"% ==== {k} ====\n{v}\n\n")

# ---- Figure 4: coefficient forest plot (primary model, main effects + interactions) ----
fig, axes = plt.subplots(1, 6, figsize=(13, 3.6), sharey=True)
for ax, d in zip(axes, DIMS):
    for i, t in enumerate(TERMS):
        r = A.loc[(t, d)]
        col = "#1f77b4" if r.p_holm < .05 else "#999999"
        ax.errorbar(r.est, len(TERMS) - i, xerr=[[r.est - r.lo], [r.hi - r.est]], fmt="o", color=col, capsize=2, ms=4)
    ax.axvline(0, color="k", lw=0.8); ax.set_title(d, fontsize=10); ax.grid(alpha=0.3, axis="x")
    if d == "PU": ax.set_xlim(-1.6, 1.2)
axes[0].set_yticks(range(len(TERMS), 0, -1)); axes[0].set_yticklabels([TERM_LABELS[t] for t in TERMS], fontsize=8)
fig.supxlabel("Estimate in rating points (+-0.5 contrast coding), 95% CI; blue = Holm-adjusted p < .05", fontsize=9)
fig.tight_layout(); fig.savefig(f"{OUT}/fig4_coefficients.pdf"); fig.savefig(f"{OUT}/fig4_coefficients.png", dpi=200)

# ---- Figure 5: variance decomposition ----
fig, ax = plt.subplots(figsize=(7, 3))
vn = vd[vd.model == "null"].set_index("dim").reindex(DIMS)
bottom = np.zeros(6)
for k, lab, col in [("resp_share", "Response", "#1f77b4"), ("bp_share", "Base prompt", "#2ca02c"), ("rater_share", "Rater", "#ff7f0e"), ("resid_share", "Residual (rater x response)", "#c7c7c7")]:
    ax.bar(DIMS, 100 * vn[k], bottom=bottom, label=lab, color=col); bottom += 100 * vn[k].values
ax.set_ylabel("% of rating variance"); ax.legend(fontsize=8, ncol=2, loc="upper center", bbox_to_anchor=(0.5, -0.15)); ax.set_ylim(0, 100)
fig.tight_layout(); fig.savefig(f"{OUT}/fig5_variance.pdf"); fig.savefig(f"{OUT}/fig5_variance.png", dpi=200)
print("tables written:", list(T)); print(open(f"{OUT}/tables.tex").read()[:3000])
