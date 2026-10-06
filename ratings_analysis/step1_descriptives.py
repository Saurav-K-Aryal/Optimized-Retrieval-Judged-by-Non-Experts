# -*- coding: utf-8 -*-
"""Step 1. Descriptives: rating distributions, cell means with cluster-bootstrap CIs, marginal means,
length by condition, dimension correlations and PCA, interaction figure, distribution figure."""
import numpy as np, pandas as pd, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from common import *
ratings, responses, rerate = load()
log = []

# ---------- rating-level distributions ----------
rows = []
for d in DIMS:
    v = ratings[d].astype(float)
    rows.append(dict(dim=d, name=DIM_NAMES[d], mean=v.mean(), sd=v.std(ddof=1), median=v.median(),
                     pct_low=100 * (v <= 2).mean(), pct_high=100 * (v >= 4).mean()))
desc = pd.DataFrame(rows).set_index("dim"); log.append("Rating-level descriptives (n=900):\n" + desc.round(2).to_string())
desc.to_csv(f"{OUT}/01_descriptives.csv")

# ---------- response-level means ----------
resp_mean = ratings.groupby(["Prompt_ID", "condition", "pipeline", "persona", "steps", "base_prompt_id"])[DIMS].mean().reset_index()
resp_mean = resp_mean.merge(responses[["Prompt_ID", "wc", "wps", "n_sent", "is_empty", "doc_refs", "cite_cues", "hedges_per100", "miss_admit", "bold_headers", "numbered"]], on="Prompt_ID")
resp_mean.to_parquet(f"{OUT}/response_means.parquet")

# cell means with cluster bootstrap over base prompts (the shared experimental units)
cells = []
for c in COND_ORDER:
    sub = resp_mean[resp_mean.condition == c]
    row = dict(condition=c, n=len(sub))
    for d in DIMS:
        est, lo, hi = cluster_boot_ci(sub, d, "base_prompt_id", n=4000, seed=1)
        row[d] = est; row[d + "_lo"] = lo; row[d + "_hi"] = hi
    cells.append(row)
cells = pd.DataFrame(cells).set_index("condition")
log.append("Cell means (response-level, 95% cluster-bootstrap CI over base prompts):\n" +
           "\n".join(f"{c}: " + ", ".join(f"{d}={cells.loc[c,d]:.2f} [{cells.loc[c,d+'_lo']:.2f},{cells.loc[c,d+'_hi']:.2f}]" for d in DIMS) for c in COND_ORDER))
cells.to_csv(f"{OUT}/01_cell_means.csv")

# marginal means by factor
marg = []
for fac, lab in [("pipeline", None), ("persona", None), ("steps", None)]:
    for lev, sub in resp_mean.groupby(fac):
        row = dict(factor=fac, level=str(lev), n=len(sub))
        for d in DIMS:
            est, lo, hi = cluster_boot_ci(sub, d, "base_prompt_id", n=4000, seed=2)
            row[d] = est; row[d + "_lo"] = lo; row[d + "_hi"] = hi
        marg.append(row)
marg = pd.DataFrame(marg); marg.to_csv(f"{OUT}/01_marginal_means.csv", index=False)
log.append("Marginal means by factor (response-level, cluster-bootstrap CI):\n" +
           "\n".join(f"{r.factor}={r.level}: " + ", ".join(f"{d}={r[d]:.2f} [{r[d+'_lo']:.2f},{r[d+'_hi']:.2f}]" for d in DIMS) for _, r in marg.iterrows()))

# raw marginal differences (paired within base prompt: mean over the 4 cells on each side)
diffs = []
for fac in ["pipeline", "persona", "steps"]:
    hi_lev = {"pipeline": "autorag", "persona": True, "steps": True}[fac]
    per_bp = resp_mean.groupby(["base_prompt_id", fac])[DIMS].mean().unstack(fac)
    row = dict(factor=fac)
    for d in DIMS:
        dd = per_bp[(d, hi_lev)] - per_bp[(d, [x for x in per_bp[d].columns if x != hi_lev][0])]
        est, lo, hi = boot_ci(dd.values, n=4000, seed=3)
        row[d] = est; row[d + "_lo"] = lo; row[d + "_hi"] = hi
        row[d + "_dz"] = dd.mean() / dd.std(ddof=1)
    diffs.append(row)
diffs = pd.DataFrame(diffs).set_index("factor"); diffs.to_csv(f"{OUT}/01_marginal_diffs.csv")
log.append("Marginal differences (high minus low level, paired within base prompt; 95% bootstrap CI; dz = paired Cohen's d):\n" +
           "\n".join(f"{f}: " + ", ".join(f"{d}={diffs.loc[f,d]:+.2f} [{diffs.loc[f,d+'_lo']:+.2f},{diffs.loc[f,d+'_hi']:+.2f}] dz={diffs.loc[f,d+'_dz']:+.2f}" for d in DIMS) for f in diffs.index))

# ---------- length and text features by condition ----------
txt = responses[responses.is_empty == 0].groupby("condition")[["wc", "n_sent", "wps", "bold_headers", "numbered", "doc_refs", "cite_cues", "hedges_per100", "miss_admit", "acronyms"]].agg(["mean", "std"])
txt.to_csv(f"{OUT}/01_text_by_condition.csv")
log.append("Text features by condition (responses with text):\n" + responses[responses.is_empty == 0].groupby("condition")[["wc", "wps", "bold_headers", "numbered", "doc_refs", "cite_cues", "hedges_per100", "miss_admit"]].mean().round(2).loc[COND_ORDER].to_string())
log.append("Word count by pipeline: " + responses[responses.is_empty == 0].groupby("pipeline").wc.describe()[["mean", "std", "min", "50%", "max"]].round(1).to_string())

# ---------- correlations and PCA ----------
rl = ratings[DIMS].astype(float)
corr_rl = rl.corr(method="spearman"); corr_rl.to_csv(f"{OUT}/01_corr_rating_level.csv")
corr_rp = resp_mean[DIMS].corr(method="pearson"); corr_rp.to_csv(f"{OUT}/01_corr_response_level.csv")
log.append("Spearman correlations (rating level):\n" + corr_rl.round(2).to_string())
log.append("Pearson correlations (response-level means):\n" + corr_rp.round(2).to_string())
from sklearn.decomposition import PCA
for nm, X in [("rating", rl), ("response", resp_mean[DIMS])]:
    Z = (X - X.mean()) / X.std(ddof=1)
    p = PCA().fit(Z)
    load = pd.DataFrame(p.components_.T * np.sqrt(p.explained_variance_), index=DIMS, columns=[f"PC{i+1}" for i in range(6)])
    log.append(f"PCA ({nm} level) variance explained: {np.round(p.explained_variance_ratio_, 3).tolist()}\nloadings:\n" + load.round(2).iloc[:, :3].to_string())
    load.to_csv(f"{OUT}/01_pca_loadings_{nm}.csv")
# Cronbach alpha of the five positively keyed items plus reversed PU
X = rl.copy(); X["PU"] = 6 - X["PU"]
k = X.shape[1]; alpha = k / (k - 1) * (1 - X.var(ddof=1).sum() / X.sum(axis=1).var(ddof=1))
log.append(f"Cronbach alpha over six items (PU reversed): {alpha:.3f}")

# ---------- Figure 1: interaction plot ----------
fig, axes = plt.subplots(2, 3, figsize=(11, 6.2), sharex=True)
xlab = ["none", "steps", "persona", "persona+steps"]
xkey = [("noPersona", "noSteps"), ("noPersona", "steps"), ("persona", "noSteps"), ("persona", "steps")]
for ax, d in zip(axes.ravel(), DIMS):
    for pipe, col, mk in [("base_llm", "#7f7f7f", "o"), ("autorag", "#1f77b4", "s")]:
        ys = [cells.loc[f"{pipe}|{p}|{s}", d] for p, s in xkey]
        lo = [cells.loc[f"{pipe}|{p}|{s}", d + "_lo"] for p, s in xkey]
        hi = [cells.loc[f"{pipe}|{p}|{s}", d + "_hi"] for p, s in xkey]
        off = -0.06 if pipe == "base_llm" else 0.06
        ax.errorbar(np.arange(4) + off, ys, yerr=[np.array(ys) - np.array(lo), np.array(hi) - np.array(ys)], fmt=mk + "-", color=col,
                    capsize=3, label={"base_llm": "Base LLM", "autorag": "AutoRAG-optimized RAG"}[pipe])
    ax.set_title(f"{d}: {DIM_NAMES[d]}", fontsize=10); ax.set_xticks(range(4)); ax.set_xticklabels(xlab, rotation=20, fontsize=8)
    ax.set_ylim(1, 5); ax.grid(alpha=0.3)
axes[0, 0].legend(fontsize=8, loc="lower left"); fig.supylabel("Mean rating (1-5), response level, 95% CI")
fig.tight_layout(); fig.savefig(f"{OUT}/fig1_interaction.pdf"); fig.savefig(f"{OUT}/fig1_interaction.png", dpi=200)

# ---------- Figure 2: stacked rating distributions by pipeline ----------
fig, axes = plt.subplots(1, 6, figsize=(12, 2.8), sharey=True)
cmap = plt.get_cmap("RdYlGn")
for ax, d in zip(axes, DIMS):
    tab = pd.crosstab(ratings.pipeline, ratings[d], normalize="index").reindex(["base_llm", "autorag"]) * 100
    left = np.zeros(2)
    for v in range(1, 6):
        vals = tab.get(v, pd.Series([0, 0], index=tab.index)).values
        colr = cmap((6 - v) / 5) if d == "PU" else cmap(v / 5)
        ax.barh(["Base LLM", "AutoRAG"], vals, left=left, color=colr, edgecolor="white", label=str(v)); left += vals
    ax.set_title(d, fontsize=10); ax.set_xlim(0, 100); ax.set_xlabel("% of ratings", fontsize=8)
axes[-1].legend(title="rating", fontsize=7, title_fontsize=7, bbox_to_anchor=(1.02, 1), loc="upper left")
fig.tight_layout(); fig.savefig(f"{OUT}/fig2_distributions.pdf"); fig.savefig(f"{OUT}/fig2_distributions.png", dpi=200)

open(f"{OUT}/01_descriptives.txt", "w").write("\n\n".join(log)); print("\n\n".join(log))
