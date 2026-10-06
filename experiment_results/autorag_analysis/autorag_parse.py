#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
autorag_parse.py  --  Parse an AutoRAG project folder (one or more trials) and find the best full pipeline.

Usage
    python autorag_parse.py <results_root> [--out OUTDIR] [--metrics bleu,rouge,meteor,sem_score,bert_score]
                            [--strip-think]  (re-weights nothing; only reports the share of <think> text)

What it reads (standard AutoRAG layout)
    <root>/<trial>/config.yaml
    <root>/<trial>/summary.csv                          trial-level best module per node (AutoRAG's own choice)
    <root>/<trial>/<node_line>/<node>/summary.csv       one row per module configuration, with mean metrics
    <root>/<trial>/<node_line>/<node>/<k>.parquet       per-query results for configuration k

What it writes to OUTDIR
    trials.csv                 one row per trial: generator model(s), context length, number of eval queries, AutoRAG's picks
    retrieval_configs.csv      every retrieval configuration in every trial with metrics and parsed params
    reranker_configs.csv       every reranker configuration
    prompt_configs.csv         every prompt-maker configuration
    generator_configs.csv      every generator configuration (model, sampling params, context length, metrics, composite)
    generator_by_model.csv     per-model summary across its configurations (mean, sd, best, rank)
    generator_per_query.csv    per-query metric values for every generator configuration (long format)
    model_pairwise_wins.csv    per-query paired comparison of models (median over each model's configurations)
    variance_decomposition.csv between-model vs within-model variance for each generation metric
    optimal_pipeline.yaml      the selected full pipeline in AutoRAG config style
    optimal_pipeline.json      the same selection with the metric values that justify it
    report.md                  human-readable report with caveats

Selection rules (stated so they can be argued with)
    retrieval  : highest retrieval_ndcg, then retrieval_map, then retrieval_mrr, then retrieval_f1, then lowest execution time
    reranker   : highest passage_reranker_retrieval_f1, then recall, then precision, then lowest execution time
    prompt     : highest prompt_maker_sem_score, then meteor, then rouge, then bleu, then lowest execution time
    generator  : highest composite = mean over metrics of the min-max normalized value across ALL generator
                 configurations in ALL trials (so models are compared on a common scale), with the rank-average
                 composite reported alongside; ties broken by bert_score then sem_score
"""
import argparse, ast, glob, json, os, re, sys, hashlib
import numpy as np, pandas as pd, yaml

GEN_METRICS_DEFAULT = ["bleu", "rouge", "meteor", "sem_score", "bert_score"]
RET_PRIORITY = ["retrieval_ndcg", "retrieval_map", "retrieval_mrr", "retrieval_f1", "retrieval_recall", "retrieval_precision"]
RER_PRIORITY = ["passage_reranker_retrieval_f1", "passage_reranker_retrieval_recall", "passage_reranker_retrieval_precision"]
PM_PRIORITY = ["prompt_maker_sem_score", "prompt_maker_meteor", "prompt_maker_rouge", "prompt_maker_bleu"]


def safe_eval(s):
    try:
        return ast.literal_eval(s)
    except Exception:
        return {"_raw": str(s)}


def flat_params(p):
    """Flatten the module_params dict into scalar columns useful for grouping."""
    out = {}
    if not isinstance(p, dict):
        return out
    for k, v in p.items():
        if k == "target_module_params":
            # hybrid modules: pull the embedding model of the vector side
            try:
                for sub in v:
                    if isinstance(sub, dict) and "embedding_model" in sub:
                        out["embedding_model"] = str(sub["embedding_model"])
            except Exception:
                pass
        elif k == "prompt":
            out["prompt_text"] = str(v)
            out["prompt_hash"] = hashlib.md5(str(v).encode()).hexdigest()[:8]
            m = re.search(r"approaching (\w+(?: \w+)?)", str(v))
            out["prompt_topic"] = m.group(1) if m else str(v)[:40]
        elif isinstance(v, (list, tuple, dict)):
            out[k] = str(v)
        else:
            out[k] = v
    return out


def find_trials(root):
    trials = []
    for cfg in sorted(glob.glob(os.path.join(root, "**", "config.yaml"), recursive=True)):
        tdir = os.path.dirname(cfg)
        if not any(os.path.isdir(os.path.join(tdir, d)) for d in os.listdir(tdir) if "node_line" in d):
            continue
        trials.append(tdir)
    return trials


def content_hash(tdir):
    """Hash of all summary.csv contents, used to drop duplicated copies of the same trial."""
    h = hashlib.md5()
    for f in sorted(glob.glob(os.path.join(tdir, "**", "summary.csv"), recursive=True)):
        h.update(open(f, "rb").read())
    return h.hexdigest()


def parse_config(cfg_path):
    """Return generator models, context lengths, sampling grids and retrieval top_k declared in config.yaml."""
    info = dict(gen_models=[], context_lengths=[], temperatures=[], top_ps=[], retrieval_top_k=None, reranker_top_k=None, n_prompts=None,
                embedding_models=[], rerankers=[])
    try:
        cfg = yaml.safe_load(open(cfg_path))
    except Exception as e:
        info["config_error"] = str(e); return info
    for nl in cfg.get("node_lines", []):
        for node in nl.get("nodes", []):
            nt = node.get("node_type")
            if nt == "retrieval":
                info["retrieval_top_k"] = node.get("top_k")
                for m in node.get("modules", []):
                    if m.get("module_type") == "vectordb":
                        em = m.get("embedding_model", [])
                        info["embedding_models"] += [str(x) for x in (em if isinstance(em, list) else [em])]
            elif nt == "passage_reranker":
                info["reranker_top_k"] = node.get("top_k")
                info["rerankers"] = [m.get("module_type") for m in node.get("modules", [])]
            elif nt == "prompt_maker":
                info["n_prompts"] = sum(len(m.get("prompt", [])) if isinstance(m.get("prompt"), list) else 1 for m in node.get("modules", []))
            elif nt == "generator":
                for m in node.get("modules", []):
                    info["gen_models"].append(str(m.get("model", m.get("llm"))))
                    cl = m.get("context_length"); info["context_lengths"].append(cl)
                    t = m.get("temperature"); info["temperatures"] += t if isinstance(t, list) else [t]
                    tp = m.get("top_p"); info["top_ps"] += tp if isinstance(tp, list) else [tp]
    for k in ["gen_models", "context_lengths", "temperatures", "top_ps", "embedding_models"]:
        info[k] = sorted({str(x) for x in info[k] if x is not None})
    return info


def read_node(tdir, node_line, node):
    f = os.path.join(tdir, node_line, node, "summary.csv")
    if not os.path.exists(f):
        return None
    s = pd.read_csv(f)
    s["module_params_dict"] = s["module_params"].apply(safe_eval)
    P = pd.DataFrame([flat_params(p) for p in s["module_params_dict"]])
    s = pd.concat([s.drop(columns=["module_params_dict"]), P], axis=1)
    s.insert(0, "node", node); s.insert(0, "node_line", node_line); s.insert(0, "trial", os.path.basename(tdir))
    return s


def per_query(tdir, node_line, node, filenames, metrics):
    rows = []
    for fn in filenames:
        f = os.path.join(tdir, node_line, node, fn)
        if not os.path.exists(f):
            continue
        d = pd.read_parquet(f)
        for qi in range(len(d)):
            r = dict(trial=os.path.basename(tdir), filename=fn, query_idx=qi)
            for m in metrics:
                if m in d.columns:
                    r[m] = float(d[m].iloc[qi])
            if "generated_texts" in d.columns:
                txt = str(d["generated_texts"].iloc[qi])
                think = re.findall(r"<think>.*?</think>", txt, flags=re.S)
                r["chars"] = len(txt); r["think_char_share"] = (sum(len(t) for t in think) / len(txt)) if txt else 0.0
                r["has_think"] = int("<think>" in txt)
            rows.append(r)
    return pd.DataFrame(rows)


def pick(df, priority, lower_is_better_time=True):
    """Sort by the priority metrics (descending) then execution time (ascending); return the top row and the tie set."""
    cols = [c for c in priority if c in df.columns]
    d = df.copy()
    for c in cols:
        d[c] = pd.to_numeric(d[c], errors="coerce").round(6)
    d = d.sort_values(cols + ["execution_time"], ascending=[False] * len(cols) + [True])
    top = d.iloc[0]
    tie = d[(d[cols] == top[cols]).all(axis=1)]
    return top, tie, cols


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("root"); ap.add_argument("--out", default="autorag_analysis")
    ap.add_argument("--metrics", default=",".join(GEN_METRICS_DEFAULT))
    ap.add_argument("--include-duplicates", action="store_true", help="keep trial folders whose summaries are identical to another trial")
    ap.add_argument("--family", default=None, help="family id (F1, F2, ...) to select within; default = the family with most trials")
    args = ap.parse_args()
    metrics = [m for m in args.metrics.split(",") if m]
    os.makedirs(args.out, exist_ok=True)
    root = os.path.abspath(args.root)

    # ---------- trials ----------
    trials = find_trials(root)
    seen, keep = {}, []
    for t in trials:
        h = content_hash(t)
        if h in seen and not args.include_duplicates:
            print(f"skip duplicate trial {os.path.relpath(t, root)} (same summaries as {os.path.relpath(seen[h], root)})")
            continue
        seen[h] = t; keep.append(t)
    trials = keep
    print(f"{len(trials)} distinct trials")

    T, RET, RER, PM, GEN, PQ = [], [], [], [], [], []
    for t in trials:
        name = os.path.relpath(t, root)
        info = parse_config(os.path.join(t, "config.yaml"))
        row = dict(trial=name, path=t, **{k: ("; ".join(v) if isinstance(v, list) else v) for k, v in info.items()})
        ts = os.path.join(t, "summary.csv")
        if os.path.exists(ts):
            s = pd.read_csv(ts)
            for _, r in s.iterrows():
                row[f"autorag_best_{r.node_type}"] = f"{r.best_module_name} {r.best_module_filename}"
                row[f"autorag_best_{r.node_type}_params"] = r.best_module_params
        n_q = None
        for nl in sorted(os.listdir(t)):
            if "node_line" not in nl or not os.path.isdir(os.path.join(t, nl)):
                continue
            for node in sorted(os.listdir(os.path.join(t, nl))):
                s = read_node(t, nl, node)
                if s is None:
                    continue
                s["trial"] = name
                pq_files = s["filename"].tolist()
                if node == "retrieval":
                    RET.append(s)
                    f0 = os.path.join(t, nl, node, pq_files[0])
                    if os.path.exists(f0): n_q = len(pd.read_parquet(f0))
                elif node == "passage_reranker": RER.append(s)
                elif node == "prompt_maker": PM.append(s)
                elif node == "generator":
                    GEN.append(s)
                    pq = per_query(t, nl, node, pq_files, metrics); pq["trial"] = name; PQ.append(pq)
        row["n_eval_queries"] = n_q
        T.append(row)
    trials_df = pd.DataFrame(T); trials_df.to_csv(os.path.join(args.out, "trials.csv"), index=False)
    ret = pd.concat(RET, ignore_index=True) if RET else pd.DataFrame()
    rer = pd.concat(RER, ignore_index=True) if RER else pd.DataFrame()
    pm = pd.concat(PM, ignore_index=True) if PM else pd.DataFrame()
    gen = pd.concat(GEN, ignore_index=True) if GEN else pd.DataFrame()
    pq = pd.concat(PQ, ignore_index=True) if PQ else pd.DataFrame()

    # ---------- experiment families: trials that share the same prompt templates and metric set are comparable ----------
    fam_key = {}
    for t in trials_df.trial:
        ph = tuple(sorted(pm[pm.trial == t].prompt_hash.dropna().unique())) if not pm.empty and "prompt_hash" in pm else ()
        gcols = tuple(sorted(c for c in metrics if not gen.empty and c in gen.columns and gen.loc[gen.trial == t, c].notna().any()))
        fam_key[t] = (ph, gcols)
    fam_ids = {k: f"F{i+1}" for i, k in enumerate(sorted(set(fam_key.values()), key=lambda k: -sum(1 for v in fam_key.values() if v == k)))}
    trials_df["family"] = trials_df.trial.map(lambda t: fam_ids[fam_key[t]])
    for df in (ret, rer, pm, gen, pq):
        if not df.empty: df["family"] = df.trial.map(lambda t: fam_ids[fam_key[t]])
    main_family = args.family or trials_df.family.value_counts().idxmax()
    print("families:", trials_df.groupby("family").trial.apply(list).to_dict(), "-> selecting within", main_family)
    trials_df.to_csv(os.path.join(args.out, "trials.csv"), index=False)
    gen_all = gen.copy()
    gen = gen[gen.family == main_family].copy(); pq = pq[pq.family == main_family].copy() if not pq.empty else pq
    ret_all, rer_all, pm_all = ret.copy(), rer.copy(), pm.copy()
    ret = ret[ret.family == main_family].copy(); rer = rer[rer.family == main_family].copy(); pm = pm[pm.family == main_family].copy()

    # ---------- generator: composite and rankings (within the main family) ----------
    gm = [m for m in metrics if m in gen.columns]
    for m in gm:
        lo, hi = gen[m].min(), gen[m].max()
        gen[m + "_norm"] = (gen[m] - lo) / (hi - lo) if hi > lo else 0.0
        gen[m + "_rank"] = gen[m].rank(ascending=False, method="average")
    gen["composite_minmax"] = gen[[m + "_norm" for m in gm]].mean(axis=1)
    gen["composite_rank"] = gen[[m + "_rank" for m in gm]].mean(axis=1)
    ngram = [m for m in ["bleu", "rouge", "meteor"] if m in gm]; sem = [m for m in ["sem_score", "bert_score"] if m in gm]
    gen["composite_ngram"] = gen[[m + "_norm" for m in ngram]].mean(axis=1) if ngram else np.nan
    gen["composite_semantic"] = gen[[m + "_norm" for m in sem]].mean(axis=1) if sem else np.nan
    gen = gen.merge(trials_df[["trial", "context_lengths"]], on="trial", how="left")
    if "context_length" not in gen.columns: gen["context_length"] = gen["context_lengths"]
    if not pq.empty:
        th = pq.groupby(["trial", "filename"]).agg(think_char_share=("think_char_share", "mean"), has_think=("has_think", "max")).reset_index()
        gen = gen.merge(th, on=["trial", "filename"], how="left")
    gen = gen.sort_values("composite_minmax", ascending=False)
    gen.to_csv(os.path.join(args.out, "generator_configs.csv"), index=False)
    pq.to_csv(os.path.join(args.out, "generator_per_query.csv"), index=False)

    key = [c for c in ["model", "context_length"] if c in gen.columns]
    bym = gen.groupby(key).agg(n_configs=("filename", "size"), **{m + "_mean": (m, "mean") for m in gm}, **{m + "_sd": (m, "std") for m in gm},
                               composite_mean=("composite_minmax", "mean"), composite_best=("composite_minmax", "max"),
                               ngram_mean=("composite_ngram", "mean"), semantic_mean=("composite_semantic", "mean"),
                               bleu_best=("bleu", "max") if "bleu" in gm else ("composite_minmax", "max"),
                               out_tokens=("average_output_token", "mean") if "average_output_token" in gen.columns else ("composite_minmax", "size"),
                               think_share=("think_char_share", "mean") if "think_char_share" in gen.columns else ("composite_minmax", "size")).reset_index()
    bym = bym.sort_values("composite_mean", ascending=False); bym["rank_by_mean"] = range(1, len(bym) + 1)
    bym.to_csv(os.path.join(args.out, "generator_by_model.csv"), index=False)

    # model-level per-query paired comparison (median over a model's configurations, per query)
    pw = pd.DataFrame()
    if not pq.empty and "model" in gen.columns:
        pqm = pq.merge(gen[["trial", "filename", "model", "context_length"]], on=["trial", "filename"], how="left")
        pqm["model_ctx"] = pqm["model"].astype(str) + " @" + pqm["context_length"].astype(str)
        med = pqm.groupby(["model_ctx", "query_idx"])[gm].median().reset_index()
        rows = []
        models = sorted(med.model_ctx.unique())
        for a in models:
            for b in models:
                if a >= b: continue
                A = med[med.model_ctx == a].set_index("query_idx"); B = med[med.model_ctx == b].set_index("query_idx")
                idx = A.index.intersection(B.index)
                r = dict(model_a=a, model_b=b, n_queries=len(idx))
                for m in gm:
                    r[m + "_a_wins"] = int((A.loc[idx, m] > B.loc[idx, m]).sum()); r[m + "_b_wins"] = int((A.loc[idx, m] < B.loc[idx, m]).sum())
                rows.append(r)
        pw = pd.DataFrame(rows); pw.to_csv(os.path.join(args.out, "model_pairwise_wins.csv"), index=False)

    # variance decomposition: between model_ctx vs within (sampling grid)
    vd = []
    if "model" in gen.columns:
        gen["model_ctx"] = gen["model"].astype(str) + " @" + gen["context_length"].astype(str)
        for m in gm:
            grand = gen[m].mean(); groups = gen.groupby("model_ctx")[m]
            ssb = sum(len(g) * (g.mean() - grand) ** 2 for _, g in groups); ssw = sum(((g - g.mean()) ** 2).sum() for _, g in groups)
            vd.append(dict(metric=m, between_share=ssb / (ssb + ssw) if ssb + ssw > 0 else np.nan, within_share=ssw / (ssb + ssw) if ssb + ssw > 0 else np.nan,
                           within_sd=np.sqrt(ssw / max(1, len(gen) - gen.model_ctx.nunique())), between_range=groups.mean().max() - groups.mean().min()))
    vd = pd.DataFrame(vd); vd.to_csv(os.path.join(args.out, "variance_decomposition.csv"), index=False)

    # ---------- retrieval / reranker / prompt ----------
    for df, nm in [(ret, "retrieval_configs.csv"), (rer, "reranker_configs.csv"), (pm, "prompt_configs.csv")]:
        if not df.empty: df.to_csv(os.path.join(args.out, nm), index=False)

    def uniq_profiles(df, cols):
        c = [x for x in cols if x in df.columns]
        return df.groupby(["module_name"] + c).size().reset_index(name="n_configs") if c else pd.DataFrame()

    sel = {}
    if not ret.empty:
        # use the most complete trial(s): those with the most retrieval metrics
        ret_full = ret[ret[[c for c in RET_PRIORITY if c in ret.columns]].notna().all(axis=1)]
        top, tie, cols = pick(ret_full, RET_PRIORITY)
        sel["retrieval"] = dict(module=top.module_name, params=top.module_params, trial=top.trial, filename=top.filename,
                                metrics={c: float(top[c]) for c in cols}, n_tied=int(len(tie)),
                                tied_modules=sorted(tie.module_name.unique().tolist()))
        ret_profiles = uniq_profiles(ret_full, RET_PRIORITY)
    if not rer.empty:
        top, tie, cols = pick(rer, RER_PRIORITY)
        sel["passage_reranker"] = dict(module=top.module_name, params=top.module_params, trial=top.trial, filename=top.filename,
                                       metrics={c: float(top[c]) for c in cols}, n_tied=int(len(tie)), tied_modules=sorted(tie.module_name.unique().tolist()))
        rer_profiles = uniq_profiles(rer, RER_PRIORITY)
    if not pm.empty:
        top, tie, cols = pick(pm, PM_PRIORITY)
        sel["prompt_maker"] = dict(module=top.module_name, params=top.module_params, trial=top.trial, filename=top.filename,
                                   metrics={c: float(top[c]) for c in cols}, n_tied=int(len(tie)), tied_prompts=int(tie.prompt_hash.nunique()) if "prompt_hash" in tie else None)
    if not gen.empty:
        g = gen.sort_values(["composite_minmax", "bert_score" if "bert_score" in gm else gm[0], "sem_score" if "sem_score" in gm else gm[0]], ascending=False)
        top = g.iloc[0]
        sel["generator"] = dict(module=top.module_name, params=top.module_params, trial=top.trial, filename=top.filename,
                                metrics={m: float(top[m]) for m in gm}, composite_minmax=float(top.composite_minmax), composite_rank=float(top.composite_rank),
                                autorag_is_best_in_its_trial=bool(top.is_best))
        # AutoRAG's own per-trial best, for comparison
        ab = gen[gen.is_best == True][["trial", "model", "context_length", "filename"] + gm + ["composite_minmax"]].sort_values("composite_minmax", ascending=False)
        sel["autorag_best_per_trial"] = ab.to_dict(orient="records")
        sel["best_by_single_metric"] = {m: dict(trial=gen.loc[gen[m].idxmax(), "trial"], model=str(gen.loc[gen[m].idxmax(), "model"]), filename=gen.loc[gen[m].idxmax(), "filename"], value=float(gen[m].max())) for m in gm}

    json.dump(sel, open(os.path.join(args.out, "optimal_pipeline.json"), "w"), indent=1, default=str)

    # ---------- AutoRAG-style YAML of the selected pipeline ----------
    def params_of(section):
        return safe_eval(sel[section]["params"]) if section in sel else {}
    rp, rr, pp, gp = params_of("retrieval"), params_of("passage_reranker"), params_of("prompt_maker"), params_of("generator")
    ret_mod = sel.get("retrieval", {}).get("module", "")
    ret_type = {"HybridCC": "hybrid_cc", "HybridRRF": "hybrid_rrf", "VectorDB": "vectordb", "BM25": "bm25"}.get(ret_mod, ret_mod.lower())
    ret_module = {"module_type": ret_type}
    if ret_type.startswith("hybrid"):
        ret_module["target_modules"] = "('bm25', 'vectordb')"
        if "weights" in rp: ret_module["weights"] = [str(tuple(rp["weights"])) if not isinstance(rp["weights"], str) else rp["weights"]]
        if "rrf_k" in rp: ret_module["rrf_k"] = rp["rrf_k"]
        ret_module["_vectordb_embedding_model"] = rp.get("embedding_model", flat_params(rp).get("embedding_model"))
    elif ret_type == "vectordb":
        ret_module["embedding_model"] = rp.get("embedding_model")
    rer_type = re.sub(r"(?<!^)(?=[A-Z])", "_", sel.get("passage_reranker", {}).get("module", "")).lower()
    y = {"_note": "Selected by autorag_parse.py. The prompt template below is the tied template AutoRAG scored; it hard-codes one question. "
                  "Replace it with a template that uses {query} before deployment.",
         "node_lines": [
        {"node_line_name": "retrieve_node_line", "nodes": [
            {"node_type": "retrieval", "strategy": {"metrics": [c for c in RET_PRIORITY if c in ret.columns]}, "top_k": rp.get("top_k"), "modules": [ret_module]},
            {"node_type": "passage_reranker", "strategy": {"metrics": [c.replace("passage_reranker_", "") for c in RER_PRIORITY if c in rer.columns]}, "top_k": rr.get("top_k"),
             "modules": [{"module_type": rer_type}]}]},
        {"node_line_name": "post_retrieve_node_line", "nodes": [
            {"node_type": "prompt_maker", "strategy": {"metrics": [c.replace("prompt_maker_", "") for c in PM_PRIORITY if c in pm.columns]},
             "modules": [{"module_type": "fstring", "prompt": [pp.get("prompt", ""),
                          "You are a US Naval Officer.\n\n{query}\n\nUsing the following information as context, please provide a detailed answer:\n\n{retrieved_contents}\n\nAnswer:"]}]},
            {"node_type": "generator", "strategy": {"metrics": gm},
             "modules": [{"module_type": "llama_index_llm", **{k: v for k, v in gp.items() if k not in ("batch_size",)}}]}]}]}
    yaml.safe_dump(y, open(os.path.join(args.out, "optimal_pipeline.yaml"), "w"), sort_keys=False, allow_unicode=True, width=200)

    # ---------- report ----------
    L = ["# AutoRAG results: parsed summary and selected pipeline\n", f"Root: `{root}`  \nDistinct trials: {len(trials)}  \n"]
    L.append(f"Experiment families (trials sharing prompt templates and metrics): {trials_df.groupby('family').trial.apply(lambda x: ', '.join(x)).to_dict()}. Selection is made within **{main_family}**; other families are listed for completeness but their metrics are not comparable (different prompt templates and, most likely, a different evaluation set).\n")
    L.append("## Trials\n"); L.append(trials_df[[c for c in ["trial", "family", "gen_models", "context_lengths", "temperatures", "top_ps", "n_eval_queries", "retrieval_top_k", "reranker_top_k", "n_prompts"] if c in trials_df]].to_markdown(index=False)); L.append("")
    if not ret.empty:
        L.append("## Retrieval (distinct metric profiles across configurations)\n"); L.append(ret_profiles.to_markdown(index=False)); L.append("")
        L.append(f"Selected: **{sel['retrieval']['module']}** `{sel['retrieval']['params']}` from trial {sel['retrieval']['trial']} ({sel['retrieval']['filename']}); metrics {sel['retrieval']['metrics']}; tied with {sel['retrieval']['n_tied'] - 1} other configuration(s) of {sel['retrieval']['tied_modules']}.\n")
    if not rer.empty:
        L.append("## Passage reranker (distinct metric profiles)\n"); L.append(rer_profiles.to_markdown(index=False)); L.append("")
        L.append(f"Selected: **{sel['passage_reranker']['module']}** (ties broken by execution time); {sel['passage_reranker']['n_tied']} configurations tie on all metrics: {sel['passage_reranker']['tied_modules']}.\n")
    if not pm.empty:
        L.append("## Prompt maker\n"); L.append(pm[[c for c in ["trial", "filename", "prompt_topic", "average_prompt_token"] + PM_PRIORITY if c in pm.columns]].drop_duplicates().head(12).to_markdown(index=False)); L.append("")
        L.append(f"Selected: prompt {sel['prompt_maker']['filename']} of trial {sel['prompt_maker']['trial']} ({sel['prompt_maker']['n_tied']} configurations tie).\n")
    if not gen.empty:
        L.append("## Generator: models ranked by mean composite over their sampling grid\n")
        show = ["model", "context_length", "n_configs"] + [m + "_mean" for m in gm] + ["composite_mean", "ngram_mean", "semantic_mean", "composite_best", "out_tokens", "think_share", "rank_by_mean"]
        L.append(bym[[c for c in show if c in bym.columns]].round(3).to_markdown(index=False)); L.append("")
        L.append("## Generator: top 15 configurations by composite (min-max normalized mean over metrics)\n")
        showg = ["trial", "model", "context_length", "temperature", "top_p", "top_k", "min_p"] + gm + ["composite_minmax", "composite_ngram", "composite_semantic", "composite_rank", "is_best"]
        L.append(gen[[c for c in showg if c in gen.columns]].head(15).round(3).to_markdown(index=False)); L.append("")
        other = gen_all[gen_all.family != main_family]
        if not other.empty:
            L.append("## Other families (not comparable): best generator row per trial\n")
            oc = [c for c in ["trial", "family", "model", "filename"] + gm if c in other.columns]
            L.append(other.sort_values("bleu" if "bleu" in other else gm[0], ascending=False).groupby("trial").head(1)[oc].round(3).to_markdown(index=False)); L.append("")
        L.append("## Best configuration by each single metric\n")
        L.append(pd.DataFrame(sel["best_by_single_metric"]).T.to_markdown()); L.append("")
        L.append("## AutoRAG's own per-trial best generator, re-scored on the common composite\n")
        L.append(pd.DataFrame(sel["autorag_best_per_trial"]).round(3).to_markdown(index=False)); L.append("")
        if not vd.empty:
            L.append("## Variance decomposition of generator metrics (between model/context vs within sampling grid)\n"); L.append(vd.round(3).to_markdown(index=False)); L.append("")
        if not pw.empty:
            # condensed win matrix: wins summed over metrics and queries (row beats column)
            models = sorted(set(pw.model_a) | set(pw.model_b)); W = pd.DataFrame(0, index=models, columns=models)
            for _, r in pw.iterrows():
                for m in gm:
                    W.loc[r.model_a, r.model_b] += r[m + "_a_wins"]; W.loc[r.model_b, r.model_a] += r[m + "_b_wins"]
            W["total_wins"] = W.sum(axis=1); W = W.sort_values("total_wins", ascending=False)
            L.append(f"## Paired wins between models (row beats column; summed over {len(gm)} metrics x {int(pw.n_queries.max())} queries, medians over each model's grid; full table in model_pairwise_wins.csv)\n")
            L.append(W.to_markdown()); L.append("")
            a, b = W.index[0], W.index[1]
            sub = pw[((pw.model_a == a) & (pw.model_b == b)) | ((pw.model_a == b) & (pw.model_b == a))].iloc[0]
            flip = sub.model_a != a
            L.append(f"Head to head, {a} vs {b}, per metric (queries won by {a} / by {b}): " + "; ".join(f"{m} {sub[m + ('_b_wins' if flip else '_a_wins')]}/{sub[m + ('_a_wins' if flip else '_b_wins')]}" for m in gm) + "\n")
        L.append("## Selected generator\n"); L.append(f"**{sel['generator']['params']}** (trial {sel['generator']['trial']}, {sel['generator']['filename']}); metrics {sel['generator']['metrics']}; composite {sel['generator']['composite_minmax']:.3f}; AutoRAG marked it best in its own trial: {sel['generator']['autorag_is_best_in_its_trial']}.\n")
    L.append("## Caveats the selection inherits from the experiment\n")
    nq = trials_df.n_eval_queries.dropna().unique().tolist()
    L.append(f"* Every metric is a mean over **{nq} evaluation queries**. Differences between configurations are not statistically distinguishable at this sample size; treat the ranking as a heuristic.")
    if "temperature" in gen.columns:
        L.append("* Generator configurations were sampled once each at temperatures up to 1.0; a single sample per query at non-zero temperature adds noise of the same order as the between-configuration differences (see variance decomposition).")
    if "think_char_share" in gen.columns and gen.think_char_share.fillna(0).max() > 0:
        L.append("* Reasoning models emit `<think>` blocks that were scored as part of the answer; their BLEU/ROUGE/METEOR are depressed relative to non-reasoning models for that reason alone. Strip the block and re-score before comparing across model families.")
    if not ret.empty and "embedding_model" in ret.columns:
        vdb = ret[ret.module_name == "VectorDB"]
        if vdb.embedding_model.nunique() > 1 and vdb.groupby("trial")[[c for c in RET_PRIORITY if c in vdb.columns]].nunique().max().max() == 1:
            L.append(f"* All {vdb.embedding_model.nunique()} embedding models declared for VectorDB return identical retrieval metrics within every trial (and identical retrieved ids where checked), so the embedding-model sweep did not take effect (one Chroma collection was reused). The effective search space is far smaller than the declared one; drop the embedding factor from any configuration count.")
    if not rer.empty and len(sel.get("passage_reranker", {}).get("tied_modules", [])) > 1:
        L.append("* All rerankers tie on every metric at top_k 3, so the choice of `pass_reranker` is by cost, not by quality.")
    if "prompt_topic" in pm.columns and pm.prompt_topic.nunique() > 1:
        L.append("* The prompt-maker templates hard-code a question and do not use `{query}`, so the same fixed question was posed for every evaluation query; prompt-maker metrics cannot be interpreted as prompt quality.")
    L.append("* `context_length` in the generator module is the Ollama context window, not the chunk size; chunking is fixed at the data stage and was not varied inside these trials.")
    open(os.path.join(args.out, "report.md"), "w").write("\n".join(L))
    print("\n".join(L))


if __name__ == "__main__":
    main()
