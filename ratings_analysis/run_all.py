#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
run_all.py  --  Reproduce every analysis in the paper from the ratings workbook.

    python run_all.py "COMBINED LABEL DATA.xlsx" [--skip-ordinal] [--out out]

Steps (each is a standalone script in this folder; run_all sets RATINGS_XLSX and runs them in order):
  step00_forms.py       rebuild the long rating table from the eight raw form tabs (full timestamps)
  step0_prep.py         analysis tables: 900 unique ratings, 60 re-rated pairs, 240 responses with text features
  step1_descriptives.py descriptives, cell means with cluster-bootstrap CIs, marginal contrasts, correlations, PCA, Fig. 1 and 2
  step2_irr.py          Krippendorff alpha (bootstrap CIs), ICC by condition, pairwise kappa, test-retest, rater severity
  step3_models.py       primary crossed random-intercept models, response-level model, robustness refits, variance
                        decomposition, leave-one-rater-out, simple effects
  step4_ordinal.py      Bayesian cumulative-link mixed models (bambi/PyMC; slow, ~20 min on 2 cores; --skip-ordinal keeps the
                        previous 04_* outputs)
  step5_secondary.py    source-of-uncertainty tags, surface vs grounding, per-prompt heterogeneity, text-feature model,
                        refusals, free-text coding, Fig. 3
  step6_report.py       LaTeX tables (out/tables.tex) and Fig. 4 and 5
Writes out/MANIFEST.txt with the workbook checksum, package versions, and a checksum of every output file.
"""
import argparse, hashlib, os, subprocess, sys, time, glob, platform

HERE = os.path.dirname(os.path.abspath(__file__))
STEPS = ["step00_forms.py", "step0_prep.py", "step1_descriptives.py", "step2_irr.py", "step3_models.py", "step4_ordinal.py", "step5_secondary.py", "step6_report.py"]

def md5(p):
    h = hashlib.md5()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""): h.update(chunk)
    return h.hexdigest()

def main():
    ap = argparse.ArgumentParser(); ap.add_argument("xlsx"); ap.add_argument("--skip-ordinal", action="store_true"); ap.add_argument("--out", default="out")
    a = ap.parse_args()
    xlsx = os.path.abspath(a.xlsx); assert os.path.exists(xlsx), xlsx
    env = dict(os.environ, RATINGS_XLSX=xlsx, MPLBACKEND="Agg")
    os.chdir(HERE); os.makedirs(a.out, exist_ok=True)
    log = [f"run_all.py started {time.strftime('%Y-%m-%d %H:%M:%S')}", f"workbook: {xlsx}", f"workbook md5: {md5(xlsx)}", f"python {platform.python_version()}"]
    try:
        import pandas, numpy, statsmodels, scipy, krippendorff, pingouin, matplotlib
        log.append(f"pandas {pandas.__version__}, numpy {numpy.__version__}, statsmodels {statsmodels.__version__}, scipy {scipy.__version__}, pingouin {pingouin.__version__}, matplotlib {matplotlib.__version__}")
        try:
            import bambi, pymc; log.append(f"bambi {bambi.__version__}, pymc {pymc.__version__}")
        except Exception: log.append("bambi/pymc not installed (ordinal step will fail unless skipped)")
    except Exception as e:
        log.append(f"package check failed: {e}")
    for step in STEPS:
        if step == "step4_ordinal.py" and a.skip_ordinal:
            log.append(f"[skip] {step}"); print(f"[skip] {step}"); continue
        t0 = time.time(); print(f"[run] {step} ...", flush=True)
        r = subprocess.run([sys.executable, step], env=env, capture_output=True, text=True)
        open(os.path.join(a.out, f"log_{step.replace('.py', '')}.txt"), "w").write(r.stdout + "\n--- stderr ---\n" + r.stderr)
        status = "ok" if r.returncode == 0 else f"FAILED rc={r.returncode}"
        log.append(f"[{status}] {step} ({time.time() - t0:.0f} s)"); print(f"[{status}] {step} ({time.time() - t0:.0f} s)")
        if r.returncode != 0:
            print(r.stderr[-3000:]); break
    log.append("\noutput checksums:")
    for f in sorted(glob.glob(os.path.join(a.out, "*"))):
        if os.path.isfile(f) and not os.path.basename(f).startswith("log_") and os.path.basename(f) != "MANIFEST.txt":
            log.append(f"  {md5(f)}  {os.path.basename(f)}")
    open(os.path.join(a.out, "MANIFEST.txt"), "w").write("\n".join(log)); print("\n".join(log[:6]))

if __name__ == "__main__":
    main()
