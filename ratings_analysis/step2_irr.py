# -*- coding: utf-8 -*-
"""Step 2. Inter-rater reliability: ordinal Krippendorff's alpha with bootstrap CIs (all raters; core raters),
alpha by pipeline and by rater block, ICC on complete rater blocks, pairwise weighted kappa/Spearman,
intra-rater test-retest (two double submissions), and rater severity on shared responses."""
import numpy as np, pandas as pd, krippendorff, pingouin as pg
from scipy.stats import spearmanr
from sklearn.metrics import cohen_kappa_score
from common import *
ratings, responses, rerate = load()
log = []
raters = sorted(ratings.LABELER_ID.unique())

def alpha_matrix(df, d, units="Prompt_ID"):
    """rows = raters, cols = units, NaN = missing."""
    return df.pivot_table(index="LABELER_ID", columns=units, values=d, aggfunc="first").to_numpy(dtype=float)

def kalpha(df, d, level="ordinal"):
    M = alpha_matrix(df, d)
    return krippendorff.alpha(reliability_data=M, level_of_measurement=level)

def kalpha_boot(df, d, n=2000, seed=0, level="ordinal"):
    """Bootstrap over units (responses)."""
    rng = np.random.default_rng(seed)
    piv = df.pivot_table(index="LABELER_ID", columns="Prompt_ID", values=d, aggfunc="first")
    M = piv.to_numpy(dtype=float); k = M.shape[1]; est = krippendorff.alpha(reliability_data=M, level_of_measurement=level)
    bs = []
    for _ in range(n):
        idx = rng.integers(0, k, size=k)
        try: bs.append(krippendorff.alpha(reliability_data=M[:, idx], level_of_measurement=level))
        except Exception: pass
    return est, np.percentile(bs, 2.5), np.percentile(bs, 97.5)

# ---------- Krippendorff alpha ----------
rows = []
core = ["L02", "L03", "L05", "L07", "L08"]
for d in DIMS:
    a, lo, hi = kalpha_boot(ratings, d, seed=1)
    a_int = kalpha(ratings, d, "interval"); a_nom = kalpha(ratings, d, "nominal")
    a5, lo5, hi5 = kalpha_boot(ratings[ratings.LABELER_ID.isin(core)], d, seed=2)
    rows.append(dict(dim=d, alpha_ordinal=a, lo=lo, hi=hi, alpha_interval=a_int, alpha_nominal=a_nom, alpha_core5=a5, lo5=lo5, hi5=hi5,
                     alpha_base=kalpha(ratings[ratings.pipeline == "base_llm"], d), alpha_autorag=kalpha(ratings[ratings.pipeline == "autorag"], d)))
ka = pd.DataFrame(rows).set_index("dim"); ka.to_csv(f"{OUT}/02_krippendorff.csv")
log.append("Krippendorff alpha (ordinal; 95% bootstrap CI over responses), all 8 raters vs core 5 raters (L02,L03,L05,L07,L08), and by pipeline:\n" + ka.round(3).to_string())

# alpha by condition (each condition has its own rater subset)
rows = []
for c in COND_ORDER:
    sub = ratings[ratings.condition == c]
    rows.append(dict(condition=c, n_raters=sub.LABELER_ID.nunique(), raters=",".join(sorted(sub.LABELER_ID.unique())),
                     **{d: kalpha(sub, d) for d in DIMS}))
kc = pd.DataFrame(rows).set_index("condition"); kc.to_csv(f"{OUT}/02_krippendorff_by_condition.csv")
log.append("Krippendorff alpha (ordinal) by condition:\n" + kc.round(2).to_string())

# ---------- ICC on complete rater blocks ----------
rows = []
for c in COND_ORDER:
    sub = ratings[ratings.condition == c]
    full = sub.groupby("LABELER_ID").Prompt_ID.nunique(); full = full[full == 30].index.tolist()
    sub = sub[sub.LABELER_ID.isin(full)]
    r = dict(condition=c, k=len(full))
    for d in DIMS:
        icc = pg.intraclass_corr(data=sub, targets="Prompt_ID", raters="LABELER_ID", ratings=d).set_index("Type")
        r[d + "_ICC2_1"] = icc.loc["ICC(A,1)", "ICC"]; r[d + "_ICC2_k"] = icc.loc["ICC(A,k)", "ICC"]
    rows.append(r)
icc = pd.DataFrame(rows).set_index("condition"); icc.to_csv(f"{OUT}/02_icc_by_condition.csv")
log.append("ICC(2,1) single rater / ICC(2,k) mean of k raters, complete blocks per condition:\n" +
           icc[[f"{d}_ICC2_1" for d in DIMS] + ["k"]].round(2).to_string() + "\n" + icc[[f"{d}_ICC2_k" for d in DIMS]].round(2).to_string())
# pooled ICC across conditions: average of ICC2_1 weighted equally (report range)
log.append("ICC(2,1) range across conditions: " + ", ".join(f"{d} {icc[d+'_ICC2_1'].min():.2f}-{icc[d+'_ICC2_1'].max():.2f}" for d in DIMS))

# ---------- pairwise weighted kappa and Spearman ----------
rows = []
for i, a in enumerate(raters):
    for b in raters[i + 1:]:
        A = ratings[ratings.LABELER_ID == a].set_index("Prompt_ID"); B = ratings[ratings.LABELER_ID == b].set_index("Prompt_ID")
        shared = A.index.intersection(B.index)
        if len(shared) < 30: continue
        r = dict(rater_a=a, rater_b=b, n_shared=len(shared))
        for d in DIMS:
            x, y = A.loc[shared, d].astype(int), B.loc[shared, d].astype(int)
            r[d + "_wkappa"] = cohen_kappa_score(x, y, weights="linear", labels=[1, 2, 3, 4, 5])
            r[d + "_rho"] = spearmanr(x, y).correlation
            r[d + "_exact"] = (x == y).mean()
        rows.append(r)
pw = pd.DataFrame(rows); pw.to_csv(f"{OUT}/02_pairwise.csv", index=False)
log.append("Pairwise rater agreement (pairs sharing >= 30 responses): mean linear-weighted kappa / Spearman rho / exact agreement:\n" +
           pd.DataFrame({d: [pw[d + "_wkappa"].mean(), pw[d + "_rho"].mean(), pw[d + "_exact"].mean()] for d in DIMS}, index=["wkappa", "rho", "exact"]).round(2).to_string() +
           f"\n(n pairs = {len(pw)}; kappa range across pairs: " + ", ".join(f"{d} {pw[d+'_wkappa'].min():.2f}-{pw[d+'_wkappa'].max():.2f}" for d in DIMS) + ")")

# ---------- intra-rater test-retest ----------
rows = []
for (rid, c), g in rerate.groupby(["LABELER_ID", "condition"]):
    s1 = g[g.submission == 1].set_index("Prompt_ID"); s2 = g[g.submission == 2].set_index("Prompt_ID")
    idx = s1.index.intersection(s2.index)
    gap = (s2.ts.min() - s1.ts.max()).days
    r = dict(rater=rid, condition=c, n=len(idx), interval_days=gap)
    for d in DIMS:
        x, y = s1.loc[idx, d].astype(int), s2.loc[idx, d].astype(int)
        r[d + "_wkappa"] = cohen_kappa_score(x, y, weights="linear", labels=[1, 2, 3, 4, 5])
        r[d + "_rho"] = spearmanr(x, y).correlation
        r[d + "_exact"] = (x == y).mean(); r[d + "_within1"] = ((x - y).abs() <= 1).mean(); r[d + "_mad"] = (x - y).abs().mean()
        r[d + "_shift"] = (y - x).mean()
    rows.append(r)
tr = pd.DataFrame(rows); tr.to_csv(f"{OUT}/02_test_retest.csv", index=False)
log.append("Intra-rater test-retest (second minus first submission):\n" + "\n".join(
    f"{r.rater} ({r.condition}, {r.n} responses, {r.interval_days} d): " + ", ".join(f"{d} wk={r[d+'_wkappa']:.2f} rho={r[d+'_rho']:.2f} exact={r[d+'_exact']:.2f} w1={r[d+'_within1']:.2f} shift={r[d+'_shift']:+.2f}" for d in DIMS) for _, r in tr.iterrows()))

# ---------- rater severity: mean rating per rater on responses shared with at least one other rater, centred on response mean ----------
resp_mean = ratings.groupby("Prompt_ID")[DIMS].transform("mean")
dev = ratings[DIMS].astype(float) - resp_mean
dev["LABELER_ID"] = ratings.LABELER_ID
sev = dev.groupby("LABELER_ID").mean(); sev["n"] = ratings.groupby("LABELER_ID").size()
sev.to_csv(f"{OUT}/02_rater_severity.csv")
log.append("Rater severity (mean deviation from the response mean across that rater's responses):\n" + sev.round(2).to_string())
# rating-level rater means and SDs
rm = ratings.groupby("LABELER_ID")[DIMS].agg(["mean", "std"]).round(2)
log.append("Rater means (SD):\n" + rm.to_string())

# ---------- alpha with re-rated (second) submissions swapped in: sensitivity ----------
swap = ratings.set_index(["Prompt_ID", "LABELER_ID"]).copy()
s2 = rerate[rerate.submission == 2].set_index(["Prompt_ID", "LABELER_ID"])
swap.loc[s2.index, DIMS] = s2[DIMS]
swap = swap.reset_index()
log.append("Krippendorff alpha using the second submission for the 60 re-rated pairs: " + ", ".join(f"{d}={kalpha(swap, d):.3f}" for d in DIMS))
open(f"{OUT}/02_irr.txt", "w").write("\n\n".join(log)); print("\n\n".join(log))
