# -*- coding: utf-8 -*-
"""Shared loaders and helpers for the non-expert RAG perception analysis."""
import re, os, numpy as np, pandas as pd

DIMS = ["PU", "PP", "CS", "GT", "AC", "FW"]
DIM_NAMES = {"PU": "Perceived uncertainty", "PP": "Perceived plausibility", "CS": "Comprehension support",
             "GT": "Grounding transparency", "AC": "Actionability", "FW": "Follow-up worthiness"}
COND_ORDER = ["base_llm|noPersona|noSteps", "base_llm|noPersona|steps", "base_llm|persona|noSteps", "base_llm|persona|steps",
              "autorag|noPersona|noSteps", "autorag|noPersona|steps", "autorag|persona|noSteps", "autorag|persona|steps"]
SU_TAGS = ["No source or unclear grounding", "Vague or ambiguous language", "Unexpected answer (not what I anticipated)",
           "Multiple plausible interpretations", "Conflicting or contradictory information", "Other"]
SU_SHORT = {"No source or unclear grounding": "no_source", "Vague or ambiguous language": "vague",
            "Unexpected answer (not what I anticipated)": "unexpected", "Multiple plausible interpretations": "multi_interp",
            "Conflicting or contradictory information": "conflict", "Other": "other"}
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "out")
os.makedirs(OUT, exist_ok=True)

def load():
    here = os.path.dirname(os.path.abspath(__file__))
    ratings = pd.read_parquet(os.path.join(here, "ratings.parquet"))
    responses = pd.read_parquet(os.path.join(here, "responses.parquet"))
    rerate = pd.read_parquet(os.path.join(here, "rerate.parquet"))
    return ratings, responses, rerate

def add_codes(df):
    """Sum/contrast coding: +0.5 / -0.5 so main effects are marginal mean differences."""
    df = df.copy()
    df["rag"] = np.where(df["pipeline"] == "autorag", 0.5, -0.5)
    df["per"] = np.where(df["persona"], 0.5, -0.5)
    df["stp"] = np.where(df["steps"], 0.5, -0.5)
    df["rag_per"] = df["rag"] * df["per"]; df["rag_stp"] = df["rag"] * df["stp"]; df["per_stp"] = df["per"] * df["stp"]
    df["rag_per_stp"] = df["rag"] * df["per"] * df["stp"]
    return df

TERMS = ["rag", "per", "stp", "rag_per", "rag_stp", "per_stp", "rag_per_stp"]
TERM_LABELS = {"rag": "RAG (AutoRAG vs base LLM)", "per": "Persona prefix", "stp": "Steps suffix",
               "rag_per": "RAG x Persona", "rag_stp": "RAG x Steps", "per_stp": "Persona x Steps", "rag_per_stp": "RAG x Persona x Steps"}

def holm(p):
    """Holm step-down adjusted p-values (array-like)."""
    p = np.asarray(p, dtype=float); n = len(p); order = np.argsort(p); adj = np.empty(n)
    running = 0.0
    for rank, idx in enumerate(order):
        val = (n - rank) * p[idx]; running = max(running, val); adj[idx] = min(1.0, running)
    return adj

def boot_ci(values, stat=np.mean, n=5000, seed=0, alpha=0.05):
    rng = np.random.default_rng(seed); v = np.asarray(values, dtype=float); v = v[~np.isnan(v)]
    if len(v) == 0: return (np.nan, np.nan, np.nan)
    idx = rng.integers(0, len(v), size=(n, len(v)))
    bs = stat(v[idx], axis=1)
    return (stat(v), np.percentile(bs, 100 * alpha / 2), np.percentile(bs, 100 * (1 - alpha / 2)))

def cluster_boot_ci(df, value_col, cluster_col, n=5000, seed=0, alpha=0.05):
    """Bootstrap resampling clusters (e.g. base prompts) with replacement; statistic = mean of value_col."""
    rng = np.random.default_rng(seed)
    groups = {k: g[value_col].to_numpy(dtype=float) for k, g in df.groupby(cluster_col)}
    keys = list(groups.keys()); est = np.nanmean(df[value_col].to_numpy(dtype=float))
    bs = np.empty(n)
    for b in range(n):
        pick = rng.integers(0, len(keys), size=len(keys))
        bs[b] = np.nanmean(np.concatenate([groups[keys[i]] for i in pick]))
    return est, np.percentile(bs, 100 * alpha / 2), np.percentile(bs, 100 * (1 - alpha / 2))

def fmt_p(p):
    if p < 0.001: return "<.001"
    return f"{p:.3f}".lstrip("0")

def tex_escape(s):
    return str(s).replace("&", "\\&").replace("%", "\\%").replace("_", "\\_")

def parse_su(s):
    if pd.isna(s) or not str(s).strip(): return []
    s = str(s); out = []
    for t in SU_TAGS:
        if t in s: out.append(SU_SHORT[t])
    return out
