# -*- coding: utf-8 -*-
"""Figures and LaTeX tables for the AutoRAG search space and the selection of the pipeline used in the study.

    python autorag_figures.py [analysis_dir] [out_dir]

Reads the CSV tables written by autorag_parse.py (trials.csv, retrieval_configs.csv, reranker_configs.csv,
prompt_configs.csv, generator_configs.csv, generator_by_model.csv, variance_decomposition.csv) and writes
  fig6_autorag_selection.pdf/.png   four panels: retrieval, reranker, generator, sampling grid
  tab_autorag_search.tex            main-text table: search space, outcome, and selection per node
  tab_autorag_trials.tex            supplement: every trial run, including the pilot families
  tab_autorag_generators.tex        supplement: generator models by mean and best composite
  tab_autorag_topconfigs.tex        supplement: the ten best sampling configurations
"""
import os, sys, ast, numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

A = sys.argv[1] if len(sys.argv) > 1 else "analysis"
OUT = sys.argv[2] if len(sys.argv) > 2 else A
os.makedirs(OUT, exist_ok=True)
plt.rcParams.update({"font.size": 8, "axes.titlesize": 9, "axes.labelsize": 8, "legend.fontsize": 7, "xtick.labelsize": 7, "ytick.labelsize": 7,
                     "font.family": "serif", "pdf.fonttype": 42})

trials = pd.read_csv(os.path.join(A, "trials.csv"), dtype={"trial": str})
ret = pd.read_csv(os.path.join(A, "retrieval_configs.csv"), dtype={"trial": str})
rer = pd.read_csv(os.path.join(A, "reranker_configs.csv"), dtype={"trial": str})
prm = pd.read_csv(os.path.join(A, "prompt_configs.csv"), dtype={"trial": str})
gen = pd.read_csv(os.path.join(A, "generator_configs.csv"), dtype={"trial": str})
gbm = pd.read_csv(os.path.join(A, "generator_by_model.csv"))
var = pd.read_csv(os.path.join(A, "variance_decomposition.csv"))
MAIN = trials[trials.family == "F1"].trial.tolist()
ret = ret[ret.trial.isin(MAIN)]; rer = rer[rer.trial.isin(MAIN)]; prm = prm[prm.trial.isin(MAIN)]; gen = gen[gen.trial.isin(MAIN)]

MODEL_LABEL = {"llama3.1": "Llama 3.1 8B", "deepseek-r1": "DeepSeek-R1 7.6B", "deepseek-r1:14b": "DeepSeek-R1 14B", "qwen2.5": "Qwen 2.5 7B", "qwen2.5:14b": "Qwen 2.5 14B"}
RET_LABEL = {"VectorDB": "Dense\n(10 embeddings\n× 3 settings)", "BM25": "BM25", "HybridRRF": "Hybrid rank\nfusion (k=3,5,10)", "HybridCC": "Hybrid convex\n(3 weightings)"}
RER_LABEL = {"PassReranker": "Pass-through", "ColbertReranker": "ColBERT", "Tart": "TART", "KoReranker": "KoReranker", "Upr": "UPR", "RankGPT": "RankGPT"}
BLUE, GRAY, RED, LIGHT, ORANGE = "#2b5d9b", "#8c8c8c", "#b2182b", "#c9d7ea", "#e07b00"

# the configuration used for the rated responses: AutoRAG's own within-trial choice for Llama 3.1 at a 512 window
gen["key"] = gen.model + "@" + gen.context_length.astype(int).astype(str)
USED = gen[(gen.model == "llama3.1") & (gen.context_length == 512) & (gen.is_best == True)].iloc[0]
BEST_COMPOSITE = gen.sort_values("composite_minmax", ascending=False).iloc[0]

# ------------------------------------------------------------------ figure
fig, axes = plt.subplots(2, 2, figsize=(6.9, 6.3))
fig.subplots_adjust(hspace=0.6, wspace=0.42, left=0.085, right=0.95, top=0.95, bottom=0.13)

# (a) retrieval: rank metrics by module, median over trials; configurations within a module tie
ax = axes[0, 0]
order = ["VectorDB", "BM25", "HybridRRF", "HybridCC"]
med = ret.groupby("module_name")[["retrieval_mrr", "retrieval_map", "retrieval_ndcg"]].median().loc[order]
ncfg = ret.groupby(["module_name", "trial"]).size().groupby("module_name").first().loc[order]
x = np.arange(len(order)); w = 0.26
for i, (m, lab) in enumerate([("retrieval_mrr", "MRR"), ("retrieval_map", "MAP"), ("retrieval_ndcg", "NDCG")]):
    ax.bar(x + (i - 1) * w, med[m], w, color=[BLUE if o == "HybridCC" else GRAY for o in order], alpha=1 - 0.28 * i, edgecolor="white", label=lab)
for xi, o in zip(x, order):
    ax.text(xi, med.loc[o].max() + 0.015, f"{int(ncfg[o])}/trial", ha="center", va="bottom", fontsize=6.5, color="#333")
ax.set_xticks(x); ax.set_xticklabels(["Dense\n(10 emb.)", "BM25", "Hybrid RRF\n(k = 3, 5, 10)", "Hybrid CC\n(3 weights)"], fontsize=6.3)
ax.set_ylim(0, 0.88); ax.set_ylabel("Median over 10 trials"); ax.set_title("(a) Retrieval node, top-k 20", loc="left")
ax.legend(frameon=False, ncol=3, loc="upper left", handlelength=1.2, columnspacing=0.8, bbox_to_anchor=(0, 1.02))
ax.text(x[-1], 0.80, "selected", ha="center", fontsize=7, color=BLUE)
ax.spines[["top", "right"]].set_visible(False)

# (b) reranker: metrics tie; execution time breaks the tie
ax = axes[0, 1]
rorder = ["PassReranker", "ColbertReranker", "Tart", "KoReranker", "Upr", "RankGPT"]
rs = rer.groupby("module_name").agg(recall=("passage_reranker_retrieval_recall", "median"), precision=("passage_reranker_retrieval_precision", "median"),
                                    t_med=("execution_time", "median"), t_min=("execution_time", "min"), t_max=("execution_time", "max")).loc[rorder]
x = np.arange(len(rorder))
ax.bar(x - 0.18, rs.recall, 0.36, color=[BLUE if r == "PassReranker" else GRAY for r in rorder], edgecolor="white", label="Recall@3")
ax.bar(x + 0.18, rs.precision, 0.36, color=[BLUE if r == "PassReranker" else GRAY for r in rorder], alpha=0.55, edgecolor="white", label="Precision@3")
ax.set_ylim(0, 1.05); ax.set_ylabel("Median over 10 trials")
ax.set_xticks(x); ax.set_xticklabels([RER_LABEL[r] for r in rorder], rotation=30, ha="right", fontsize=6.5)
ax2 = ax.twinx()
ax2.errorbar(x, rs.t_med, yerr=[rs.t_med - rs.t_min, rs.t_max - rs.t_med], fmt="o", color=RED, ms=3.5, capsize=2, lw=0.8, label="Exec. time (s)")
ax2.set_yscale("log"); ax2.set_ylim(1e-3, 8); ax2.set_ylabel("Execution time (s, log)", color=RED); ax2.tick_params(axis="y", colors=RED, labelsize=6.5)
ax.set_title("(b) Reranker node, top-k 3", loc="left")
h1, l1 = ax.get_legend_handles_labels(); h2, l2 = ax2.get_legend_handles_labels()
ax.legend(h1 + h2, l1 + l2, frameon=False, loc="upper left", ncol=3, bbox_to_anchor=(-0.02, 1.03), handlelength=1.0, columnspacing=0.6, fontsize=6)
ax.text(0, 0.72, "selected", ha="center", fontsize=7, color=BLUE)
ax.spines[["top"]].set_visible(False); ax2.spines[["top"]].set_visible(False)

# (c) generator: composite over the 18-configuration grid per model and window
ax = axes[1, 0]
gbm["key"] = gbm.model + "@" + gbm.context_length.astype(int).astype(str)
gorder = gbm.sort_values("composite_mean", ascending=False).key.tolist()
data = [gen[gen.key == k].composite_minmax.values for k in gorder]
bp = ax.boxplot(data, positions=np.arange(len(gorder)), widths=0.55, patch_artist=True, showfliers=False, medianprops=dict(color="black", lw=1))
for patch, k in zip(bp["boxes"], gorder):
    patch.set_facecolor(LIGHT if k.startswith("llama") else "#e6e6e6"); patch.set_edgecolor("#555")
rng = np.random.default_rng(3)
for i, k in enumerate(gorder):
    v = gen[gen.key == k]
    ax.scatter(i + rng.uniform(-0.18, 0.18, len(v)), v.composite_minmax, s=6, color="#444", alpha=0.6, zorder=3)
iu = gorder.index(USED.key)
ax.scatter([iu], [USED.composite_minmax], marker="*", s=130, color=ORANGE, edgecolor="white", zorder=5, label="used in the study")
ax.scatter([gorder.index(BEST_COMPOSITE.key)], [BEST_COMPOSITE.composite_minmax], marker="D", s=26, facecolor="none", edgecolor=RED, zorder=5, label="best composite")
think = gbm.set_index("key").think_share
for i, k in enumerate(gorder):
    if think[k] > 0:
        ax.text(i, 0.96, f"{think[k]*100:.0f}%\nthink", ha="center", va="top", fontsize=5.5, color="#555")
SHORT = {"llama3.1": "Llama 3.1 8B", "deepseek-r1": "DS-R1 7.6B", "deepseek-r1:14b": "DS-R1 14B", "qwen2.5": "Qwen 2.5 7B", "qwen2.5:14b": "Qwen 2.5 14B"}
ax.set_xticks(np.arange(len(gorder)))
ax.set_xticklabels([f"{SHORT[k.split('@')[0]]} / {k.split('@')[1]}" for k in gorder], rotation=40, ha="right", fontsize=6.3)
ax.set_ylabel("Composite of 5 generation metrics"); ax.set_ylim(-0.02, 0.98)
ax.set_title("(c) Generator / window, 18 configurations each", loc="left")
ax.legend(frameon=False, loc="upper right", fontsize=6.3, handlelength=1.2, bbox_to_anchor=(1.0, 0.82))
ax.spines[["top", "right"]].set_visible(False)

# (d) sampling grid for the selected model and window
ax = axes[1, 1]
sub = gen[(gen.model == "llama3.1") & (gen.context_length == 512)]
temps, tps = [0.1, 0.5, 1.0], [0.1, 0.5, 1.0]
blocks = [(10.0, 0.1), (90.0, 0.9)]
M = np.full((3, 7), np.nan)
for b, (tk, mp) in enumerate(blocks):
    for i, T in enumerate(temps):
        for j, P in enumerate(tps):
            r = sub[(sub.temperature == T) & (sub.top_p == P) & (sub.top_k == tk) & (sub.min_p == mp)]
            if len(r):
                M[i, j + 4 * b] = r.composite_minmax.iloc[0]
im = ax.imshow(M, cmap="Blues", vmin=0, vmax=0.9, aspect="auto")
for i in range(3):
    for j in range(7):
        if not np.isnan(M[i, j]):
            ax.text(j, i, f"{M[i, j]:.2f}", ha="center", va="center", fontsize=7, color="white" if M[i, j] > 0.5 else "#222")
ax.set_xticks([0, 1, 2, 4, 5, 6]); ax.set_xticklabels([f"{p}" for p in tps] * 2)
ax.set_yticks(range(3)); ax.set_yticklabels([f"{t}" for t in temps]); ax.set_ylabel("temperature"); ax.set_xlabel("top-p")
ax.text(1, -0.62, "top-k 10, min-p 0.1", ha="center", va="bottom", fontsize=6.5); ax.text(5, -0.62, "top-k 90, min-p 0.9", ha="center", va="bottom", fontsize=6.5)
ax.axvline(3, color="white", lw=8)
ju = tps.index(USED.top_p) + (4 if USED.top_k == 90 else 0); iu = temps.index(USED.temperature)
ax.add_patch(plt.Rectangle((ju - 0.5, iu - 0.5), 1, 1, fill=False, edgecolor=ORANGE, lw=2.6))
jb = tps.index(BEST_COMPOSITE.top_p) + (4 if BEST_COMPOSITE.top_k == 90 else 0); ib = temps.index(BEST_COMPOSITE.temperature)
ax.add_patch(plt.Rectangle((jb - 0.5, ib - 0.5), 1, 1, fill=False, edgecolor=RED, lw=1.4, ls="--"))
ax.set_title("(d) Llama 3.1 8B / 512: sampling grid", loc="left", pad=20)
ax.legend(handles=[Patch(facecolor="none", edgecolor=ORANGE, lw=2.6, label="used in the study"), Patch(facecolor="none", edgecolor=RED, lw=1.4, ls="--", label="best composite")],
          frameon=False, fontsize=6.3, loc="upper center", bbox_to_anchor=(0.5, -0.2), ncol=2)
for sp in ax.spines.values(): sp.set_visible(False)
cb = fig.colorbar(im, ax=ax, fraction=0.04, pad=0.03); cb.ax.tick_params(labelsize=6)

fig.savefig(os.path.join(OUT, "fig6_autorag_selection.pdf")); fig.savefig(os.path.join(OUT, "fig6_autorag_selection.png"), dpi=200)
plt.close(fig)

# ------------------------------------------------------------------ numbers used in the tables
n_trials = len(MAIN)
ret_per_trial = ret.groupby("trial").size().iloc[0]
cc_sep = (ret[ret.module_name == "HybridCC"].groupby("trial").retrieval_mrr.max() > ret[ret.module_name == "BM25"].groupby("trial").retrieval_mrr.max()).sum()
rer_time = rer.groupby("module_name").execution_time.agg(["median", "min", "max"])
pass_t = rer_time.loc["PassReranker", "median"]; other_min = rer_time.drop("PassReranker")["min"].min(); other_max = rer_time.drop("PassReranker")["max"].max()
lm = gbm.set_index("key")
within_bleu = var.set_index("metric").loc["bleu", "within_share"]
within_other = var.set_index("metric").drop("bleu").within_share
top3_temps = sub.sort_values("composite_minmax", ascending=False).head(3).temperature.tolist()
n14 = gbm[gbm.model.str.contains("14b")].rank_by_mean.tolist()
pilot = trials[trials.family != "F1"]

def esc(s):
    return str(s).replace("%", r"\%").replace("&", r"\&").replace("_", r"\_")

# ------------------------------------------------------------------ main-text table
rows = [
    ("Retrieval (top-$k$ 20)",
     f"{ret_per_trial} per trial ({len(ret)} in all); dense retrieval with 10 embedding models ($\\times$ 3 metadata settings), BM25, hybrid convex combination with dense/BM25 weights 0.5/0.5, 0.7/0.3, 0.3/0.7, and hybrid reciprocal rank fusion with $k$ = 3, 5, 10",
     "F1, recall, precision, MRR, MAP, NDCG",
     f"All ten embedding models returned identical passages in every trial (MRR 0.33); BM25 and rank fusion reached 0.50; the convex combination reached 0.67 in {cc_sep} of {n_trials} trials, with its three weightings tied",
     "Hybrid convex combination, equal weights (all-mpnet-base-v2 + BM25), the only choice that separated on the rank metrics"),
    ("Passage reranker (top-$k$ 3)",
     f"6 per trial ({len(rer)} in all); pass-through, TART, UPR, KoReranker, RankGPT, ColBERT",
     "F1, recall, precision after reranking",
     f"All six tied on every metric in every trial (recall 0.67, precision 0.22 at top-$k$ 3)",
     f"Pass-through (no reranking), by execution time ({pass_t:.3f} s against {other_min:.2f} to {other_max:.1f} s for the others)"),
    ("Prompt template",
     f"3 per trial ({len(prm)} in all); fixed-string templates wrapping the passages",
     "BLEU, ROUGE, METEOR, semantic similarity of the generated answer",
     "The templates hard-coded a question and did not insert the evaluation query, so their scores do not measure template quality and are identical within a trial",
     "The first template, which AutoRAG selected in every trial; treated as a fixed element of the pipeline"),
    ("Generator model and context window",
     f"{gbm.model.nunique()} models $\\times$ 2 windows, one trial each; Llama 3.1 8B, DeepSeek-R1 7.6B and 14B, Qwen 2.5 7B and 14B, at 512 and 1,024 tokens",
     "BLEU, ROUGE, METEOR, semantic similarity, BERTScore",
     f"Llama 3.1 at 512 had the highest mean composite over its grid ({lm.loc['llama3.1@512','composite_mean']:.2f}) and the best single configuration ({lm.loc['llama3.1@512','composite_best']:.2f}); DeepSeek-R1 7.6B at 512 matched the mean ({lm.loc['deepseek-r1@512','composite_mean']:.2f}) by leading the two semantic metrics while reasoning traces made up {lm.loc['deepseek-r1@512','think_share']*100:.0f} percent of its output and depressed its $n$-gram scores; the four 14B runs took four of the five lowest ranks",
     "Llama 3.1 8B with a 512-token window"),
    ("Sampling grid",
     f"18 per model and window ({len(gen)} in all); temperature \\{{0.1, 0.5, 1.0\\}} $\\times$ top-$p$ \\{{0.1, 0.5, 1.0\\}} $\\times$ (top-$k$, min-$p$) $\\in$ \\{{(10, 0.1), (90, 0.9)\\}}, one sample per query",
     "as above",
     f"Variation within a grid accounted for {within_bleu*100:.0f} percent of the BLEU variance and {within_other.min()*100:.0f} to {within_other.max()*100:.0f} percent for the other metrics; the three best Llama 3.1 configurations all used temperature 0.1",
     f"Temperature {USED.temperature}, top-$p$ {USED.top_p}, top-$k$ {int(USED.top_k)}, min-$p$ {USED.min_p}, AutoRAG's within-trial choice (composite {USED.composite_minmax:.2f}; the top-$p$ {BEST_COMPOSITE.top_p} neighbour scored {BEST_COMPOSITE.composite_minmax:.2f})"),
    ("Pilot runs (not comparable)",
     f"{len(pilot)} earlier trials with Llama 3 and Mistral generators, one embedding model, three rerankers, and different templates (supplement, Table~S5)",
     "same metric families",
     "Exploratory runs preceding the main search; their templates differ, so their scores are not comparable with the main search, and the three two-generator trials did not write a best configuration",
     "Not used for selection"),
]
L = [r"\begin{table}[!htbp]", r"\caption{The AutoRAG search over the doctrinal corpus (ten trials, three generated evaluation queries) and the choice made at each node for the pipeline used in the study. Counts are configurations evaluated; metrics are means over the three queries.}",
     r"\label{tab:autorag}", r"\footnotesize", r"\setlength{\tabcolsep}{4pt}", r"\begin{tabular}{@{}>{\raggedright\arraybackslash}p{0.10\textwidth}>{\raggedright\arraybackslash}p{0.23\textwidth}>{\raggedright\arraybackslash}p{0.10\textwidth}>{\raggedright\arraybackslash}p{0.26\textwidth}>{\raggedright\arraybackslash}p{0.17\textwidth}@{}}", r"\toprule",
     r"Node & Candidates evaluated & Metrics & Outcome & Selected, and why \\", r"\midrule"]
for r in rows:
    L.append(" & ".join(r) + r" \\[2pt]")
L += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
open(os.path.join(OUT, "tab_autorag_search.tex"), "w").write("\n".join(L) + "\n")

# ------------------------------------------------------------------ supplement: every trial
def fam_label(f):
    return {"F1": "Main search", "F2": "Pilot B", "F3": "Pilot C", "F4": "Pilot A"}.get(f, f)
def models_label(s):
    return "; ".join(MODEL_LABEL.get(m.strip(), {"llama3": "Llama 3 8B", "mistral": "Mistral 7B"}.get(m.strip(), m.strip())) for m in str(s).split(";"))
def n_emb(s):
    return len([x for x in str(s).split(";") if x.strip()]) if isinstance(s, str) else 0
def n_rer(s):
    return len([x for x in str(s).split(";") if x.strip()]) if isinstance(s, str) else 0
def grid_label(r):
    if pd.isna(r.temperatures):
        return "default"
    return f"{len(r.temperatures.split(';'))} $\\times$ {len(r.top_ps.split(';'))} $\\times$ 2 (18)"
def best_gen(r):
    return "" if pd.isna(r.autorag_best_generator) else "yes"
t2 = trials.copy()
t2["sortkey"] = t2.trial.str.replace(" copy", ".5").astype(float)
t2 = t2.sort_values("sortkey")
L = [r"\begin{table}[t]", r"\caption{Every AutoRAG trial found in the results folder. The main search (ten trials) shares one prompt-template set and metric set and is the basis of Table~\ref{tab:autorag}; the three pilot families used different templates and are not comparable with it or with each other. The sampling grid is temperature $\times$ top-$p$ $\times$ two (top-$k$, min-$p$) pairs, with the number of generator configurations in parentheses; completed means that AutoRAG wrote a best configuration for every node.}",
     r"\label{tab:autorag-trials}", r"\footnotesize", r"\begin{tabular}{@{}llllrrrl@{}}", r"\toprule",
     r"Trial & Family & Generator(s) & Context window & Embeddings & Rerankers & Sampling grid & Completed \\", r"\midrule"]
for _, r in t2.iterrows():
    ctx = "default" if pd.isna(r.context_lengths) else str(int(r.context_lengths))
    L.append(f"{esc(r.trial)} & {fam_label(r.family)} & {models_label(r.gen_models)} & {ctx} & {n_emb(r.embedding_models)} & {n_rer(r.rerankers)} & {grid_label(r)} & {best_gen(r)} \\\\")
L += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
open(os.path.join(OUT, "tab_autorag_trials.tex"), "w").write("\n".join(L) + "\n")

# ------------------------------------------------------------------ supplement: generator by model
g2 = gbm.sort_values("rank_by_mean")
L = [r"\begin{table}[t]", r"\caption{Generator models in the main search, ranked by mean composite over their 18 sampling configurations. Metric columns are means over the grid (three evaluation queries each); composite is the min-max normalized mean of the five metrics across all 180 configurations; think share is the fraction of output characters inside reasoning tags, which were scored as part of the answer.}",
     r"\label{tab:autorag-generators}", r"\footnotesize", r"\begin{tabular}{@{}llrrrrrrrrr@{}}", r"\toprule",
     r"Model & Window & BLEU & ROUGE & METEOR & Sem. sim. & BERTScore & Comp. (mean) & Comp. (best) & Out. tokens & Think \\", r"\midrule"]
for _, r in g2.iterrows():
    L.append(f"{MODEL_LABEL[r.model]} & {int(r.context_length)} & {r.bleu_mean:.2f} & {r.rouge_mean:.3f} & {r.meteor_mean:.3f} & {r.sem_score_mean:.3f} & {r.bert_score_mean:.3f} & {r.composite_mean:.2f} & {r.composite_best:.2f} & {r.out_tokens:.0f} & {r.think_share*100:.0f}\\% \\\\")
L += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
open(os.path.join(OUT, "tab_autorag_generators.tex"), "w").write("\n".join(L) + "\n")

# ------------------------------------------------------------------ supplement: top configurations
top = gen.sort_values("composite_minmax", ascending=False).head(10)
L = [r"\begin{table}[t]", r"\caption{The ten best generator configurations in the main search by composite. ``Used'' marks the configuration that generated the rated responses, which is the one AutoRAG selected within the Llama 3.1 trial at a 512-token window.}",
     r"\label{tab:autorag-top}", r"\footnotesize", r"\begin{tabular}{@{}lrrrrrrrrrrl@{}}", r"\toprule",
     r"Model & Window & T & top-$p$ & top-$k$ & min-$p$ & BLEU & ROUGE & METEOR & Sem. sim. & BERTScore & Composite \\", r"\midrule"]
for _, r in top.iterrows():
    used = " (used)" if (r.trial == USED.trial and r.filename == USED.filename) else ""
    L.append(f"{MODEL_LABEL[r.model]} & {int(r.context_length)} & {r.temperature} & {r.top_p} & {int(r.top_k)} & {r.min_p} & {r.bleu:.2f} & {r.rouge:.3f} & {r.meteor:.3f} & {r.sem_score:.3f} & {r.bert_score:.3f} & {r.composite_minmax:.2f}{used} \\\\")
L += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
open(os.path.join(OUT, "tab_autorag_topconfigs.tex"), "w").write("\n".join(L) + "\n")

print(f"main trials: {MAIN}; retrieval configs {len(ret)}, reranker {len(rer)}, prompt {len(prm)}, generator {len(gen)}; pilot trials {len(pilot)}")
print(f"used config: trial {USED.trial} {USED.filename} T={USED.temperature} top_p={USED.top_p} top_k={USED.top_k} min_p={USED.min_p} composite={USED.composite_minmax:.3f}")
print(f"best composite: trial {BEST_COMPOSITE.trial} {BEST_COMPOSITE.filename} T={BEST_COMPOSITE.temperature} top_p={BEST_COMPOSITE.top_p} composite={BEST_COMPOSITE.composite_minmax:.3f}")
print(f"CC separated in {cc_sep}/{n_trials} trials; pass-through {pass_t:.3f}s vs {other_min:.2f}-{other_max:.1f}s; top-3 llama temps {top3_temps}; 14B ranks {n14}")
