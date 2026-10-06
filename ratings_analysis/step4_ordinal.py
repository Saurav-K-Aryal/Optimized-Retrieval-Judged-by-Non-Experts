# -*- coding: utf-8 -*-
"""Step 4. Bayesian cumulative-link (ordinal) mixed models, one per dimension:
 rating ~ rag*per*stp + (1|rater) + (1|base_prompt) + (1|response), logit link.
Reports posterior mean, 94% HDI and P(effect > 0) for the seven design terms, on the latent logit scale."""
import warnings, sys, numpy as np, pandas as pd, arviz as az, bambi as bmb
warnings.filterwarnings("ignore")
from common import *
ratings, responses, rerate = load(); ratings = add_codes(ratings)
ratings["rater"] = ratings.LABELER_ID.astype(str); ratings["bp"] = ratings.base_prompt_id.astype(str); ratings["resp"] = ratings.Prompt_ID.astype(str)
dims = sys.argv[1:] or DIMS
rows = []; log = []
for d in dims:
    df = ratings[["rater", "bp", "resp", "rag", "per", "stp", "rag_per", "rag_stp", "per_stp", "rag_per_stp", d]].copy()
    df[d] = pd.Categorical(df[d].astype(int), categories=[1, 2, 3, 4, 5], ordered=True)
    model = bmb.Model(f"{d} ~ rag + per + stp + rag_per + rag_stp + per_stp + rag_per_stp + (1|rater) + (1|bp) + (1|resp)", df, family="cumulative")
    idata = model.fit(draws=1000, tune=1000, chains=4, cores=2, target_accept=0.9, random_seed=11, progressbar=False)
    summ = az.summary(idata, var_names=TERMS, hdi_prob=0.94)
    post = idata.posterior
    for t in TERMS:
        s = post[t].values.ravel()
        rows.append(dict(dim=d, term=t, mean=float(s.mean()), sd=float(s.std()), hdi_lo=summ.loc[t, "hdi_3%"], hdi_hi=summ.loc[t, "hdi_97%"],
                         p_gt0=float((s > 0).mean()), rhat=summ.loc[t, "r_hat"], ess=summ.loc[t, "ess_bulk"]))
    sd = az.summary(idata, var_names=["1|rater_sigma", "1|bp_sigma", "1|resp_sigma"], hdi_prob=0.94)
    log.append(f"{d}: random-effect SDs (logit): " + ", ".join(f"{i}={sd.loc[i,'mean']:.2f}" for i in sd.index) +
               f"; max rhat {summ.r_hat.max():.3f}, min ess {summ.ess_bulk.min():.0f}")
    pd.DataFrame(rows).to_csv(f"{OUT}/04_ordinal_coefficients{'_' + '_'.join(dims) if len(dims) < 6 else ''}.csv", index=False)
    print(log[-1], flush=True)
tab = pd.DataFrame(rows)
tab["cell"] = tab.apply(lambda r: f"{r['mean']:+.2f} [{r.hdi_lo:+.2f},{r.hdi_hi:+.2f}] P>0={r.p_gt0:.2f}", axis=1)
log.append("Cumulative-link mixed models (logit scale; posterior mean, 94% HDI, posterior probability the effect is positive):\n" +
           tab.pivot(index="term", columns="dim", values="cell").reindex(TERMS).to_string())
open(f"{OUT}/04_ordinal{'_' + '_'.join(dims) if len(dims) < 6 else ''}.txt", "w").write("\n\n".join(log)); print("\n\n".join(log))
