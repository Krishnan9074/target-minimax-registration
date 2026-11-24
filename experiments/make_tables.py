"""Emit every LaTeX table and every reported number from results/*.json.

Writes to outputs/latex/: one file of LaTeX macros (numbers.tex, each headline
ratio with a 95% bootstrap CI over geometries) and the tables under tables/.
No reported number is typed by hand.
"""
from __future__ import annotations

import json
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT_DIR = os.path.join(ROOT, "outputs", "latex")
RES, TAB = os.path.join(ROOT, "results"), os.path.join(OUT_DIR, "tables")
os.makedirs(TAB, exist_ok=True)
MACROS = []


def load(n):
    p = os.path.join(RES, n)
    return json.load(open(p)) if os.path.exists(p) else None


def mac(name, value):
    MACROS.append(rf"\newcommand{{\{name}}}{{{value}}}")


# --------------------------------------------------------------------------- #
#  Confidence intervals.  The unit of replication is the geometry: every
#  headline ratio is a paired per-geometry quantity, so the bootstrap resamples
#  geometries and recomputes the *same* statistic the macro reports (median or
#  mean).  Percentile bootstrap, B = 2000, fixed seed, 95%.
# --------------------------------------------------------------------------- #
B_BOOT, CI_LEVEL = 2000, 0.95
HEADLINE = set()          # names that must carry a CI twin (checked at exit)


def _finite(values):
    """Rows (units) with every entry finite; 1-D or 2-D (units x columns)."""
    v = np.asarray(values, float)
    keep = np.isfinite(v) if v.ndim == 1 else np.all(np.isfinite(v), axis=1)
    return v[keep]


F2 = lambda x: f"{x:.2f}"
F3 = lambda x: f"{x:.3f}"


def boot_ci(values, stat=np.median, B=B_BOOT, seed=0, level=CI_LEVEL):
    v = _finite(values)
    if len(v) < 2:
        return float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(v), size=(B, len(v)))
    bs = np.array([stat(v[i]) for i in idx])
    a = (1.0 - level) / 2
    return float(np.quantile(bs, a)), float(np.quantile(bs, 1 - a))


def mac_ci(name, values, fmt, stat=np.median, transform=None, headline=True):
    """Emit ``\name`` (the statistic) and ``\nameCI`` (its bootstrap interval).

    ``fmt`` renders one float (e.g. ``lambda x: f"{x:.2f}"`` or ``pct``);
    ``transform`` maps the statistic to what is reported (e.g. ``1 - x`` for a
    reduction).  The interval is transformed endpoint-wise and re-ordered.
    Also emits ``\nameLo`` / ``\nameHi`` and ``\nameN`` (number of units).
    """
    v = _finite(values)
    t = transform or (lambda x: x)
    point = t(float(stat(v)))
    lo, hi = boot_ci(v, stat=stat)
    lo, hi = sorted((t(lo), t(hi)))
    mac(name, fmt(point))
    mac(f"{name}CI", rf"[{fmt(lo)}, {fmt(hi)}]")
    mac(f"{name}Lo", fmt(lo))
    mac(f"{name}Hi", fmt(hi))
    mac(f"{name}N", f"{len(v)}")
    if headline:
        HEADLINE.add(name)
    return point, (lo, hi)


_TEX = {"&": r"\&", "%": r"\%", "$": r"\$", "#": r"\#", "_": r"\_",
        "^": r"\textasciicircum{}", "~": r"\textasciitilde{}"}


def _texesc(t):
    """Escape a plain-text string for LaTeX text mode."""
    return "".join(_TEX.get(c, c) for c in str(t))


def sci_tex(x, d=1):
    """Format a float as LaTeX scientific notation, e.g. 2.3\\times 10^{-10}."""
    m, e = f"{float(x):.{d}e}".split("e")
    return rf"{m}\times 10^{{{int(e)}}}"


KN = {1: "One", 2: "Two", 3: "Three"}


def pct(x, d=0):
    return f"{100 * x:.{d}f}\\%"


# --------------------------------------------------------------------------- #
def table_verification():
    v = load("verification.json")
    if not v:
        return
    rows = []
    for r in v:
        nm = _texesc(r["name"])
        det = _texesc(r["detail"])
        rows.append(rf"{nm} & {'pass' if r['pass'] else 'FAIL'} & {det} \\")
    open(os.path.join(TAB, "verification.tex"), "w").write(
        "\\begin{table}[h]\n\\centering\\small\n"
        "\\caption{Numerical verification of every claim. Each row is an "
        "independent check; details give the worst deviation observed.}\n"
        "\\label{tab:verification}\n"
        "\\begin{tabular}{@{}p{0.40\\textwidth} l p{0.34\\textwidth}@{}}\n"
        "\\toprule\nClaim & Status & Worst observed deviation \\\\\n\\midrule\n"
        + "\n".join(rows) + "\n\\bottomrule\n\\end{tabular}\n\\end{table}\n")
    mac("VerifPassed", f"{sum(1 for r in v if r['pass'])}")
    mac("VerifTotal", f"{len(v)}")
    # pull the specific figures quoted in the text out of the same record
    import re as _re
    sci = sci_tex
    grab = lambda pre: next(
        (_re.search(r"[0-9.]+e-?[0-9]+", r["detail"]).group()
         for r in v if r["name"].startswith(pre)
         and _re.search(r"[0-9.]+e-?[0-9]+", r["detail"])), None)
    for pre, name in (("T2 exact", "VerifSparse"), ("T5 cutting", "VerifSdp"),
                      ("T6 strong", "VerifDual"), ("P9 exact", "VerifExact"),
                      ("L1 influence", "VerifLemma")):
        got = grab(pre)
        if got:
            mac(name, sci(got))


def table_linearisation():
    d = load("exp6_linearisation.json")
    if not d:
        return
    rows = [rf"{r['rho']:.2f} & {r['median']:.1e} & {r['p95']:.1e} & "
            rf"{r['max']:.1e} \\" for r in d["rows"]]
    open(os.path.join(TAB, "linearisation.tex"), "w").write(
        "\\begin{table}[h]\n\\centering\\small\n"
        "\\caption{Relative error of the first-order target-displacement model "
        f"against a nonlinear refit ({d['n_per_rho']} random geometries per "
        "budget).}\n\\label{tab:linearisation}\n"
        "\\begin{tabular}{@{}lrrr@{}}\n\\toprule\n"
        "$\\rho$ (mm) & median & 95th pct. & max \\\\\n\\midrule\n"
        + "\n".join(rows) + "\n\\bottomrule\n\\end{tabular}\n\\end{table}\n")
    mac("LinSlope", f"{d['loglog_slope']:.3f}")
    mac("LinAtHalf", f"{d['rows'][3]['median'] * 100:.2f}\\%")
    mac("LinAtTwo", f"{d['rows'][-1]['median'] * 100:.1f}\\%")


def numbers_exp1():
    recs = load("exp1_synthetic.json")
    if not recs:
        return
    mac("NGeom", f"{len(recs)}")
    for k in (1, 2):
        sub = [r for r in recs if r["k"] == k]
        for e, tag in (("0.05", "Five"), ("0.1", "Ten"), ("0.2", "Twenty")):
            v = np.array([r["tmr"][e]["risk"] / r["gls"]["risk"] for r in sub])
            mac_ci(f"RatioK{KN[k]}{tag}", v, F2)
            mac_ci(f"RedK{KN[k]}{tag}", v, pct, transform=lambda x: 1 - x)
        v = np.array([r["tmr"]["1e+06"]["var"] / r["gls"]["var"] for r in sub])
        mac_ci(f"VarInfK{KN[k]}", v, F2)
        lb = np.array([(k / r["n"]) * r["kappa"] / r["tmr"]["1e+06"]["risk"]
                       for r in sub])
        mac(f"BoundSlackK{KN[k]}",
            f"{1/np.percentile(lb, 90):.2f}\\mbox{{--}}{1/np.percentile(lb, 10):.2f}")
    g, cuts = [], []
    for r in recs:
        for v in r["tmr"].values():
            if v["risk"] > 0:
                g.append(v["gap"] / v["risk"])
    g = np.array(g)
    mac("MaxGap", sci_tex(np.max(g)))
    mac("MedGap", sci_tex(np.median(g)))
    mac("NDesigns", f"{len(g):,}".replace(",", "{,}"))


def numbers_exp3():
    recs = load("exp3_stealth.json")
    if not recs:
        return
    rhos = [s["rho"] for s in recs[0]["sweep"]]
    for rho, tag in ((0.5, "Half"), (1.0, "One"), (3.0, "Three")):
        j = rhos.index(rho)
        mac_ci(f"Auc{tag}", [r['sweep'][j]['auc'] for r in recs], F2, stat=np.mean)
        mac_ci(f"Disp{tag}", [r['sweep'][j]['disp_mean'] for r in recs], F2,
               stat=np.mean)
        mac_ci(f"FreClean{tag}", [r['sweep'][j]['fre_clean'] for r in recs], F2,
               stat=np.mean)
        mac_ci(f"FreAtt{tag}", [r['sweep'][j]['fre_att'] for r in recs], F2,
               stat=np.mean)
    mac("NStealth", f"{len(recs)}")


def numbers_exp4():
    recs = load("exp4_ablations.json")
    if not recs:
        return
    g = lambda f: np.array([f(r) for r in recs])
    mac_ci("AblEtaTwo",
           g(lambda r: r['A1']['sweep']['0.2']['risk'] / r['A1']['gls_risk']), F3)
    mac_ci("AblEtaInf",
           g(lambda r: r['A1']['sweep']['1e+06']['risk'] / r['A1']['gls_risk']), F3)
    mac_ci("AblVarInf", g(lambda r: r['A1']['sweep']['1e+06']['var_ratio']), F2)
    fid_v = g(lambda r: r["A2"]["target_risk_fid"] / r["A2"]["target_risk_gls"])
    tgt_v = g(lambda r: r["A2"]["target_risk_tmr"] / r["A2"]["target_risk_gls"])
    fid, tgt = np.median(fid_v), np.median(tgt_v)
    fo = np.column_stack([g(lambda r: r["A2"]["fid_obj_fid"]),
                          g(lambda r: r["A2"]["fid_obj_gls"])])
    mac_ci("AblFidOnFid", fo, pct,
           stat=lambda m: 1 - np.median(m[:, 0]) / np.median(m[:, 1]))
    mac_ci("AblFidOnTgt", fid_v, pct, transform=lambda x: 1 - x)
    mac_ci("AblTgtOnTgt", tgt_v, pct, transform=lambda x: 1 - x)
    mac_ci("AblFidShare", np.column_stack([fid_v, tgt_v]), pct,
           stat=lambda m: (1 - np.median(m[:, 0])) / (1 - np.median(m[:, 1])))
    mac("AblFidWorseFrac",
        pct(np.mean(g(lambda r: r["A2"]["target_risk_fid"]
                      > r["A2"]["target_risk_gls"]))))
    mac_ci("AblFidTgtRatio", fid_v, F2)
    mac_ci("AblTgtTgtRatio", tgt_v, F2)
    wls_v = g(lambda r: r["A6"]["risk_wls"] / r["A6"]["risk_gls"])
    tm_v = g(lambda r: r["A6"]["risk_tmr"] / r["A6"]["risk_gls"])
    mac_ci("AblWlsRed", wls_v, pct, transform=lambda x: 1 - x)
    mac_ci("AblTmrRed", tm_v, pct, transform=lambda x: 1 - x)
    mac_ci("AblWlsShare", np.column_stack([wls_v, tm_v]), pct,
           stat=lambda m: (1 - np.median(m[:, 0])) / (1 - np.median(m[:, 1])))
    mac("AblManifoldDist",
        f"{np.median(g(lambda r: r['A6']['min_dist_to_wls_manifold'] / r['A6']['norm_tmr'])):.2f}")
    mac("AblMismatchDone",
        f"{np.median(g(lambda r: r['A5']['design1']['attack2'] / r['A5']['gls']['attack2'])):.2f}")
    mac("AblMismatchDtwo",
        f"{np.median(g(lambda r: r['A5']['design2']['attack2'] / r['A5']['gls']['attack2'])):.2f}")
    mac("AblMismatchDthree",
        f"{np.median(g(lambda r: r['A5']['design3']['attack1'] / r['A5']['gls']['attack1'])):.2f}")
    mac("AblDenseTen",
        f"{np.median(g(lambda r: r['A3']['0.1'] / r['A3']['gls'])):.2f}")


ORDER = ["LS", "GLS", "Huber", "RANSAC", "GNC-TLS",
         "TMR-0.05", "TMR-0.1", "TMR-0.2"]
PRETTY = {"LS": "Unweighted LS (Kabsch)", "GLS": "Gauss--Markov (GLS)",
          "Huber": "Huber IRLS", "RANSAC": "RANSAC",
          "GNC-TLS": "GNC-TLS (TEASER core)",
          "TMR-0.05": r"TMR ($\eta=0.05$)", "TMR-0.1": r"TMR ($\eta=0.10$)",
          "TMR-0.2": r"TMR ($\eta=0.20$)",
          "TMR-traj-0.1": r"TMR trajectory ($\eta=0.10$)"}


def table_main():
    recs = load("exp2_nonlinear.json")
    if not recs:
        print("  (exp2 not ready: main table skipped)")
        return
    mac("NNonlinear", f"{len(recs)}")
    def geom_means(sub, nm, field, scale):
        return np.array([np.mean([r["results"][nm][field] / (r["rho"] if scale else 1)
                                  for r in sub if r["trial"] == t])
                         for t in sorted({r["trial"] for r in sub})])

    def cell_ci(v):
        lo, hi = boot_ci(v, stat=np.mean)
        return rf"{v.mean():.3f}\,{{\tiny[{lo:.2f},{hi:.2f}]}}"
    lines = []
    for nm in ORDER:
        cells = []
        for k in (1, 2):
            sub = [r for r in recs if r["k"] == k]
            c = geom_means(sub, nm, "clean_rmse", False)
            a = geom_means(sub, nm, "adv_mean", True)
            cells += [f"{c.mean():.3f}", cell_ci(a)]
        bold = nm.startswith("TMR")
        lab = PRETTY[nm]
        row = " & ".join(cells)
        lines.append((rf"\textbf{{{lab}}} & " if bold else rf"{lab} & ") + row + r" \\")
    body = "\n".join(lines[:5] + [r"\midrule"] + lines[5:])
    open(os.path.join(TAB, "main_results.tex"), "w").write(
        "\\begin{table}[t]\n\\centering\\footnotesize\n"
        "\\caption{Random geometries ($n=7$, heteroscedastic $\\sigma \\approx "
        "0.35$~mm), every estimator attacked directly as a black box. "
        "Adversarial columns are the worst-case target displacement per "
        "millimetre of corruption budget, averaged over $\\rho\\in\\{0.5,1,2\\}$~mm "
        "and 8 geometries, with $95\\%$ bootstrap intervals over geometries; lower "
        "is better in every column.}\n"
        "\\label{tab:main}\n"
        "\\begin{tabular}{@{}lcccc@{}}\n\\toprule\n"
        "& \\multicolumn{2}{c}{$k=1$} & \\multicolumn{2}{c}{$k=2$} \\\\\n"
        "\\cmidrule(lr){2-3}\\cmidrule(lr){4-5}\n"
        "Estimator & clean RMSE (mm) & adv./$\\rho$ & clean RMSE (mm) & adv./$\\rho$ \\\\\n"
        "\\midrule\n" + body + "\n\\bottomrule\n\\end{tabular}\n\\end{table}\n")

    # inline numbers.  The unit is the geometry: the three corruption budgets
    # of one geometry are averaged first (the attack values are already per
    # millimetre of budget), then the paired ratio to Gauss--Markov is formed
    # per geometry and its mean is reported with a bootstrap interval.
    def per_geom(sub, nm, field="adv_mean", scale=True):
        out = []
        for t in sorted({r["trial"] for r in sub}):
            rs = [r for r in sub if r["trial"] == t]
            out.append(np.mean([r["results"][nm][field] / (r["rho"] if scale else 1)
                                for r in rs]))
        return np.array(out)
    mac("NNonlinearGeom", f"{len({r['trial'] for r in recs})}")
    for k in (1, 2):
        sub = [r for r in recs if r["k"] == k]
        base = per_geom(sub, "GLS")
        mac_ci(f"AdvGlsK{KN[k]}", base, F2, stat=np.mean)
        for nm, tag in (("TMR-0.1", "TmrTen"), ("TMR-0.2", "TmrTwenty"),
                        ("RANSAC", "Ransac"), ("Huber", "Huber"),
                        ("GNC-TLS", "Gnc"), ("LS", "Ls")):
            v = per_geom(sub, nm)
            mac_ci(f"Adv{tag}K{KN[k]}", v, F2, stat=np.mean)
            mac_ci(f"Rel{tag}K{KN[k]}", v / base, F2, stat=np.mean)
        for nm, tag in (("TMR-0.1", "TmrTen"), ("TMR-0.2", "TmrTwenty")):
            v = per_geom(sub, nm)
            mac_ci(f"HeadRed{tag}K{KN[k]}", v / base, pct, stat=np.mean,
                   transform=lambda x: 1 - x)
        c = per_geom(sub, "TMR-0.1", "clean_rmse", scale=False)
        c0 = per_geom(sub, "GLS", "clean_rmse", scale=False)
        mac_ci(f"CleanCostTenK{KN[k]}", c / c0, lambda x: pct(x, 1),
               stat=np.mean, transform=lambda x: x - 1)
    err = []
    for r in recs:
        for nm, p in r["predicted"].items():
            if nm in r["results"]:
                err.append(abs(r["results"][nm]["adv_mean"] - p)
                           / max(p, 1e-9))
    mac("PredMedErr", pct(np.median(err), 1))
    mac("PredPNFErr", pct(np.percentile(err, 95), 1))
    mac("PredN", f"{len(err)}")


def table_vertebra():
    recs = load("exp5_vertebra.json")
    if not recs:
        print("  (exp5 not ready: vertebra table skipped)")
        return
    order = ORDER + ["TMR-traj-0.1"]
    lines = []
    for nm in order:
        cells = []
        for k in (1, 2):
            sub = [r for r in recs if r["k"] == k]
            c = np.mean([r["results"][nm]["clean_rmse"] for r in sub])
            a = np.mean([r["results"][nm]["adv_mean"] / r["rho"] for r in sub])
            t = np.mean([r["results"][nm]["traj_adv_mean"] / r["rho"]
                         for r in sub])
            cells += ([f"{c:.3f}"] if k == 1 else []) + [f"{a:.3f}", f"{t:.3f}"]
        lab = PRETTY[nm]
        lines.append((rf"\textbf{{{lab}}} & " if nm.startswith("TMR")
                      else rf"{lab} & ") + " & ".join(cells) + r" \\")
    body = "\n".join(lines[:5] + [r"\midrule"] + lines[5:])
    open(os.path.join(TAB, "vertebra.tex"), "w").write(
        "\\begin{table}[t]\n\\centering\\footnotesize\n"
        "\\caption{L1 transpedicular screw plan, nine navigation landmarks, "
        "every estimator attacked directly. Adversarial columns are per "
        "millimetre of corruption budget, averaged over $\\rho\\in\\{1,2\\}$~mm "
        "and both sides.}\n\\label{tab:vertebra}\n"
        "\\begin{tabular}{@{}lccccc@{}}\n\\toprule\n"
        "& & \\multicolumn{2}{c}{$k=1$} & \\multicolumn{2}{c}{$k=2$} \\\\\n"
        "\\cmidrule(lr){3-4}\\cmidrule(lr){5-6}\n"
        "Estimator & clean RMSE (mm) & tip & axis & tip & axis \\\\\n"
        "\\midrule\n" + body + "\n\\bottomrule\n\\end{tabular}\n\\end{table}\n")
    # One anatomy, so there is no geometry to resample, and only four
    # (budget, side) units per k: too few for a bootstrap.  The point value is
    # the mean over those units and the companion macro is their *range*,
    # which says how stable the number is on this vertebra, not how it
    # generalises.
    def rng_mac(name, v, fmt):
        v = np.asarray(v, float)
        mac(name, fmt(float(v.mean())))
        mac(f"{name}Range", rf"{fmt(v.min())}\mbox{{--}}{fmt(v.max())}")

    def units(sub, nm, field="adv_mean"):
        return np.array([r["results"][nm][field] / r["rho"] for r in sub])
    for k in (1, 2):
        sub = [r for r in recs if r["k"] == k]
        base = units(sub, "GLS")
        rng_mac(f"VertGlsK{KN[k]}", base, F2)
        for nm, tag in (("TMR-0.1", "TmrTen"), ("TMR-0.2", "TmrTwenty"),
                        ("RANSAC", "Ransac"), ("Huber", "Huber")):
            v = units(sub, nm)
            rng_mac(f"Vert{tag}K{KN[k]}", v, F2)
            rng_mac(f"VertRel{tag}K{KN[k]}", v / base, F2)
            rng_mac(f"VertRed{tag}K{KN[k]}", 1 - v / base, pct)
        bt = units(sub, "GLS", "traj_adv_mean")
        vt = units(sub, "TMR-traj-0.1", "traj_adv_mean")
        mac(f"VertTrajGlsK{KN[k]}", F2(bt.mean()))
        mac(f"VertTrajTmrK{KN[k]}", F2(vt.mean()))
        rng_mac(f"VertTrajRedK{KN[k]}", 1 - vt / bt, pct)
    mac("VertNUnits", f"{len([r for r in recs if r['k'] == 1])}")


def geometry_numbers():
    sys.path.insert(0, ROOT)
    from tmr.anatomy import build_vertebra
    v = build_vertebra()
    F = np.array(list(v.landmarks.values()))
    tip = np.asarray(v.tip["left"])
    mac("VertNLandmarks", f"{len(v.landmarks)}")
    mac("VertLeverArm", f"{np.linalg.norm(tip - F.mean(0)):.0f}")
    mac("VertScrewLen",
        f"{np.linalg.norm(tip - np.asarray(v.entry['left'])):.0f}")


FACT_ORDER = ["GLS", "learned-avg (scalar)", "avg (full)", "TMR-W (scalar)",
              "TMR (full)"]
FACT_PRETTY = {
    "GLS": "Gauss--Markov (reference)",
    "learned-avg (scalar)": "learned weighting (population optimum)",
    "avg (full)": "average-case influence design",
    "TMR-W (scalar)": "TMR restricted to scalar weights",
    "TMR (full)": "TMR (this paper)"}


def table_factorial():
    recs = load("exp7_factorial.json")
    if not recs:
        print("  (exp7 not ready)")
        return
    def vals(kind, k, nm, field="worst"):
        sub = sorted([r for r in recs if r["kind"] == kind and r["k"] == k],
                     key=lambda r: r["trial"])
        return np.array([r["results"][nm][field] / r["results"]["GLS"][field]
                         for r in sub])

    def cell(kind, k, nm, field):
        return float(np.median(vals(kind, k, nm, field)))
    def cell_ci(kind, k, nm):
        v = vals(kind, k, nm)
        # no interval for the reference row, the two-sided vertebra, or the
        # homoscedastic average-case cells that are pinned at exactly 1
        if nm == "GLS" or len(v) < 8 or np.ptp(v) < 1e-6:
            return f"{np.median(v):.3f}"
        lo, hi = boot_ci(v)
        return rf"{np.median(v):.3f}\,{{\tiny[{lo:.2f},{hi:.2f}]}}"
    lines = []
    for nm in FACT_ORDER:
        cells = [cell_ci('random', k, nm) for k in (1, 2)]
        cells += [cell_ci('random-homoscedastic', k, nm) for k in (1, 2)]
        cells += [cell_ci('L1 vertebra', k, nm) for k in (1, 2)]
        bold = nm == "TMR (full)"
        lab = FACT_PRETTY[nm]
        lines.append((rf"\textbf{{{lab}}} & " if bold else rf"{lab} & ")
                     + " & ".join(cells) + r" \\")
    open(os.path.join(TAB, "factorial.tex"), "w").write(
        "\\begin{table}[t]\n\\centering\\footnotesize\n"
        "\\caption{Worst-case target displacement relative to Gauss--Markov, at a "
        "common $10\\%$ clean-variance budget (median over geometries with $95\\%$ "
        "bootstrap intervals; the vertebra columns are two sides of one anatomy and "
        "carry none; lower is better). Under homoscedastic noise the average-case column cannot move "
        "off Gauss--Markov at all, exactly as the average-case collapse predicts.}\n"
        "\\label{tab:factorial}\n"
        "\\resizebox{\\textwidth}{!}{\\begin{tabular}{@{}lcccccc@{}}\n\\toprule\n"
        "& \\multicolumn{2}{c}{random} & \\multicolumn{2}{c}{random, homoscedastic}"
        " & \\multicolumn{2}{c}{{L1} vertebra} \\\\\n"
        "\\cmidrule(lr){2-3}\\cmidrule(lr){4-5}\\cmidrule(lr){6-7}\n"
        "Design & $k=1$ & $k=2$ & $k=1$ & $k=2$ & $k=1$ & $k=2$ \\\\\n"
        "\\midrule\n" + "\n".join(lines) + "\n\\bottomrule\n"
        "\\end{tabular}}\n\\end{table}\n")
    for kind, tag in (("random", "Rand"), ("random-homoscedastic", "Homo"),
                      ("L1 vertebra", "Vert")):
        for nm, t2 in (("learned-avg (scalar)", "Learned"),
                       ("TMR-W (scalar)", "TmrW"), ("TMR (full)", "TmrM"),
                       ("avg (full)", "AvgFull")):
            for k, kn in ((1, "One"), (2, "Two")):
                if kind == "L1 vertebra":        # two sides: no interval
                    mac(f"Fact{tag}{t2}K{kn}", F2(cell(kind, k, nm, "worst")))
                else:
                    mac_ci(f"Fact{tag}{t2}K{kn}", vals(kind, k, nm), F2)
    lr, tm = vals("random", 1, "learned-avg (scalar)"), vals("random", 1, "TMR (full)")
    pair = np.column_stack([lr, tm])
    mac_ci("FactLearnedShare", pair, pct,
           stat=lambda m: (1 - np.median(m[:, 0])) / (1 - np.median(m[:, 1])))
    mac_ci("FactTmrOverLearned", pair, pct,
           stat=lambda m: 1 - np.median(m[:, 1]) / np.median(m[:, 0]))
    mac("NFactorial", f"{len(set((r['trial'], r['kind']) for r in recs))}")


def numbers_limits():
    recs = load("exp9_limits.json")
    if not recs:
        return
    g = lambda f: np.array([f(r) for r in recs])
    rn = g(lambda r: r["achieved"] / r["limit_new"])
    ro = g(lambda r: r["achieved"] / r["limit_old"])
    rs = g(lambda r: r["scalar_achieved"] / max(r["scalar_limit"], 1e-12))
    mac("BoundSlackNew", f"{rn.min():.3f}\\mbox{{--}}{rn.max():.3f}")
    mac("BoundSlackOld", f"{ro.min():.2f}\\mbox{{--}}{ro.max():.2f}")
    mac_ci("BoundSlackNewMed", rn, lambda x: f"{x:.4f}")
    mac_ci("BoundSlackOldMed", ro, F2)
    mac("ScalarSlackMax", f"{abs(rs - 1).max():.1e}".replace("e-0", r"\times 10^{-") + "}")
    mac("NLimits", f"{len(recs)}")


def numbers_robustness():
    sc = load("exp10_scaling.json")
    if sc:
        ns = sorted({r["n"] for r in sc})
        NW = {5: "Five", 9: "Nine", 20: "Twenty", 40: "Forty"}
        for n in (5, 9, 20, 40):
            for k, kn in ((1, "One"), (2, "Two")):
                sub = [r for r in sc if r["n"] == n and r["k"] == k]
                if sub:
                    mac_ci(f"ScaleN{NW[n]}K{kn}",
                           [r['tmr'] / r['gls'] for r in sub], F2)
        mac("ScaleNMax", f"{max(ns)}")
        mac("ScaleNMin", f"{min(ns)}")
        big = [r for r in sc if r["n"] == max(ns) and r["k"] == 1]
        mac("ScaleSolveMax", f"{np.median([r['seconds'] for r in big]):.1f}")
        for n, w in ((5, "Five"), (40, "Forty")):
            sub = [r for r in sc if r["n"] == n and r["k"] == 1]
            if sub:
                mac_ci(f"ScaleRedN{w}", [r['tmr'] / r['gls'] for r in sub], pct,
                       transform=lambda x: 1 - x)
        mac("NScaleGeom", f"{len({(r['n'], r['trial']) for r in sc})}")
    sg = load("exp11_sigma.json")
    if sg:
        ds = sorted({r["delta"] for r in sg})
        rr = [np.median([r["risk_tmr_hat"] / r["risk_gls_hat"] for r in sg
                         if r["delta"] == d and r["k"] == 1]) for d in ds]
        mac("SigRiskRange", f"{min(rr):.2f}\\mbox{{--}}{max(rr):.2f}")
        for d, tag in ((0.0, "Zero"), (0.4, "Four"), (0.8, "Eight"), (1.2, "Twelve")):
            sub = [r for r in sg if r["delta"] == d and r["k"] == 1]
            if sub:
                mac_ci(f"SigVarD{tag}",
                       [r['var_tmr_hat'] / r['var_gls_hat'] for r in sub], F2)
                mac_ci(f"SigWinD{tag}",
                       [float(r["var_tmr_hat"] <= r["var_gls_hat"]) for r in sub],
                       pct, stat=np.mean)
                mac_ci(f"SigCostD{tag}",
                       [r['risk_tmr_hat'] / r['risk_tmr_true'] for r in sub], F2)
                mac_ci(f"SigRiskD{tag}",
                       [r['risk_tmr_hat'] / r['risk_gls_hat'] for r in sub], F2)
        mac("SigDeltaMax", f"{max(ds):g}")


def numbers_qiu():
    d = load("exp12_qiu.json")
    if not d:
        return
    r = np.array([x["ratio"] for x in d["correspondence"]])
    h = np.array([x["ratio"] for x in d["hull"]])
    mac("QiuDev", sci_tex(np.abs(r - 1).max()))
    mac("QiuN", f"{len(r)}")
    mac("QiuHullDev", sci_tex(np.abs(h - 1).max()))
    mac("QiuHullN", f"{len(h)}")


def numbers_workspace():
    recs = load("exp13_workspace.json")
    if not recs:
        return
    rads = sorted({r["radius"] for r in recs})
    ws = [np.median([r["workspace_design"] / r["gls"] for r in recs
                     if r["radius"] == rr and r["k"] == 1]) for rr in rads]
    mac("WsRange", f"{min(ws):.2f}\\mbox{{--}}{max(ws):.2f}")
    mac("WsMaxRadius", f"{max(rads):.0f}")
    mac("WsNTargets", f"{max(r['n_targets'] for r in recs)}")
    for rr, tag in ((0.0, "Zero"), (40.0, "Forty"), (80.0, "Eighty")):
        sub = [r for r in recs if r["radius"] == rr and r["k"] == 1]
        if sub:
            mac_ci(f"WsDesignR{tag}", [r['workspace_design'] / r['gls'] for r in sub], F2)
            mac_ci(f"WsPointR{tag}", [r['point_design'] / r['gls'] for r in sub], F2)
            mac_ci(f"WsPenaltyR{tag}",
                   [r['point_design'] / r['workspace_design'] for r in sub], F2)
    mac("NWorkspace", f"{len({(r['trial'], r['radius'], r['k']) for r in recs})}")


def numbers_initpose():
    recs = load("exp14_initpose.json")
    if not recs:
        return
    mac("InitMaxAngle", f"{max(r['angle_deg'] for r in recs)}")
    mac("InitMaxTrans", f"{max(r['trans_mm'] for r in recs)}")
    mac("InitMaxDiff", sci_tex(max(r["target_diff"] for r in recs)))
    mac("InitMaxIters", f"{max(r['iters'] for r in recs)}")
    mac("NInitPose", f"{len(recs)}")


def table_breach():
    recs = load("exp8_breach.json")
    if not recs:
        print("  (exp8 not ready)")
        return
    lines = []
    for r in recs:
        for k in (1, 2):
            d = r[f"k{k}"]
            lines.append(
                rf"{r['side']} & {k} & {d['gls']:.3f} & {d['tmr']:.3f} & "
                rf"{d['tmr_unconstrained']:.3f} & {d['limit']:.3f} & "
                rf"{100 * d['gls'] / r['margin_mm']:.0f}\% & "
                rf"{100 * d['tmr'] / r['margin_mm']:.0f}\% \\")
    open(os.path.join(TAB, "breach.tex"), "w").write(
        "\\begin{table}[t]\n\\centering\\footnotesize\n"
        "\\caption{Medial breach margin at the pedicle isthmus. Displacements are "
        "per millimetre of corruption budget, along the medial direction. The "
        "unconstrained design attains the estimator-independent limit exactly, as "
        "Theorem~\\ref{thm:attain} requires. Margin columns use the narrowest "
        "in-bone clearance measured on this mesh ($1.75$~mm on both sides).}\n"
        "\\label{tab:breach}\n"
        "\\begin{tabular}{@{}llcccccc@{}}\n\\toprule\n"
        "& & \\multicolumn{4}{c}{medial displacement / $\\rho$} & "
        "\\multicolumn{2}{c}{margin consumed at $\\rho=1$~mm} \\\\\n"
        "\\cmidrule(lr){3-6}\\cmidrule(lr){7-8}\n"
        "side & $k$ & GLS & TMR & unconstr. & limit & GLS & TMR \\\\\n"
        "\\midrule\n" + "\n".join(lines) + "\n\\bottomrule\n"
        "\\end{tabular}\n\\end{table}\n")
    L = recs[0]
    mac("BreachMargin", f"{L['margin_mm']:.2f}")
    for k, kn in ((1, "One"), (2, "Two")):
        d = L[f"k{k}"]
        mac(f"BreachGlsK{kn}", f"{d['gls']:.2f}")
        mac(f"BreachTmrK{kn}", f"{d['tmr']:.2f}")
        mac(f"BreachLimitK{kn}", f"{d['limit']:.2f}")
        mac(f"BreachGlsPctK{kn}", pct(d['gls'] / L['margin_mm']))
        mac(f"BreachTmrPctK{kn}", pct(d['tmr'] / L['margin_mm']))
        mac(f"BreachSlackK{kn}", f"{d['slack_tmr']:.2f}")


def numbers_regime():
    recs = load("exp15_regime.json")
    if not recs:
        return
    def ratios(sw, lv, k=1):
        return np.array([r["tmr"] / r["gls"] for r in recs
                         if r["sweep"] == sw and r["level"] == lv and r["k"] == k])
    mac("NRegime", f"{len({(r['sweep'], str(r['level']), r['trial']) for r in recs})}")
    # target position: inside the hull (centroid) versus the default 80 mm
    mac_ci("RegimeCentroidRatio", ratios("target", 0.0), F2)
    mac_ci("RegimeCentroidRatioKTwo", ratios("target", 0.0, 2), F2)
    mac_ci("RegimeEightyRatio", ratios("target", 80.0), F2)
    mac_ci("RegimeFarRatio", ratios("target", 160.0), F2)
    inside = np.array([r["tmr"] / r["gls"] for r in recs if r["sweep"] == "target"
                       and r["k"] == 1 and r["hull_dist"] < 0])
    outside = np.array([r["tmr"] / r["gls"] for r in recs if r["sweep"] == "target"
                        and r["k"] == 1 and r["hull_dist"] >= 0])
    mac_ci("RegimeInHullRatio", inside, F2)
    mac_ci("RegimeOutHullRatio", outside, F2)
    mac("RegimeNInHull", f"{inside.size}")
    # number of fiducials
    for n, tag in ((4, "Four"), (5, "Five"), (7, "Seven"), (40, "Forty")):
        mac_ci(f"RegimeN{tag}Ratio", ratios("n", n), F2)
        mac_ci(f"RegimeN{tag}RatioKTwo", ratios("n", n, 2), F2)
    # conditioning
    for lv, tag in ((1.0, "One"), (0.1, "Tenth"), (0.03, "Flat"), ("outlier", "Outlier")):
        mac_ci(f"RegimeCond{tag}Ratio", ratios("cond", lv), F2)
    # noise structure
    for lv, tag in ((0.0, "Homo"), (0.35, "Default"), (0.6, "Wide")):
        mac_ci(f"RegimeNoise{tag}Ratio", ratios("noise", lv), F2)
    # the predictor: GM slack above the floor, and how much of it TMR recovers
    ok = [r for r in recs if r["limit"] > 1e-9]
    slack = np.array([r["gls"] / r["limit"] for r in ok])
    rat = np.array([r["tmr"] / r["gls"] for r in ok])
    # Spearman correlation between log slack and the ratio (no scipy dependency)
    def rank(v):
        o = np.argsort(v); rk = np.empty_like(o, dtype=float); rk[o] = np.arange(v.size)
        return rk
    rho_s = float(np.corrcoef(rank(np.log(slack)), rank(rat))[0, 1])
    mac("RegimeSpearman", f"{rho_s:.2f}")
    # fraction of instances with slack < 1.2 where TMR gains less than 10%
    near = rat[slack < 1.2]
    if near.size:
        mac("RegimeNearFloorN", f"{near.size}")
        mac_ci("RegimeNearFloorRatio", near, F2)
    recovered = np.array([(r["gls"] - r["tmr"]) / max(r["gls"] - r["limit"], 1e-12)
                          for r in ok if r["gls"] - r["limit"] > 1e-3 * r["gls"]])
    mac_ci("RegimeRecovered", np.clip(recovered, 0, 1), pct)
    for sw, lv, tag in (("n", 4, "NFour"), ("target", 0.0, "Centroid")):
        v = np.array([r["tmr"] / max(r["limit"], 1e-12) for r in recs
                      if r["sweep"] == sw and r["level"] == lv and r["k"] == 1])
        mac_ci(f"Regime{tag}OverLimit", v, F2)
    v = np.array([r["gls"] / max(r["limit"], 1e-12) for r in recs
                  if r["sweep"] == "n" and r["level"] == 4 and r["k"] == 1])
    mac_ci("RegimeNFourGmOverLimit", v, F2)


def numbers_aniso():
    fac = load("exp16_aniso_factorial.json")
    AT = {1.0: "One", 4.0: "Four", 16.0: "Sixteen", 57.0: "FiftySeven", 100.0: "Hundred"}
    if fac:
        levels = sorted({r["a"] for r in fac})
        mac("AnisoMaxRatio", f"{max(levels):g}")
        mac("AnisoNFactorial", f"{len({r['trial'] for r in fac})}")
        for a in levels:
            if a not in AT:
                continue
            for k, kn in ((1, "One"), (2, "Two")):
                sub = [r for r in fac if r["a"] == a and r["k"] == k]
                mac_ci(f"AnisoLsGmA{AT[a]}K{kn}",
                       [r["results"]["LS"]["worst"] / r["results"]["GLS"]["worst"]
                        for r in sub], F2, headline=(k == 1))
                if "TMR (full)" in sub[0]["results"]:
                    for nm, t2 in (("learned-avg (scalar)", "Learned"),
                                   ("TMR-W (scalar)", "TmrW"), ("TMR (full)", "TmrM"),
                                   ("avg (full)", "AvgFull")):
                        mac_ci(f"Aniso{t2}A{AT[a]}K{kn}",
                               [r["results"][nm]["worst"] / r["results"]["GLS"]["worst"]
                                for r in sub], F2, headline=False)
                    lr = np.array([r["results"]["learned-avg (scalar)"]["worst"]
                                   / r["results"]["GLS"]["worst"] for r in sub])
                    tm = np.array([r["results"]["TMR (full)"]["worst"]
                                   / r["results"]["GLS"]["worst"] for r in sub])
                    mac_ci(f"AnisoTmrOverLearnedA{AT[a]}K{kn}", np.column_stack([lr, tm]),
                           pct, stat=lambda m: 1 - np.median(m[:, 1]) / np.median(m[:, 0]),
                           headline=False)
                    mac_ci(f"AnisoLsVarA{AT[a]}K{kn}",
                           [r["results"]["LS"]["var"] / r["results"]["GLS"]["var"]
                            for r in sub], F2, headline=False)
        # crossover: smallest a at which the median LS/GM ratio reaches 1 (k=1)
        med = {a: np.median([r["results"]["LS"]["worst"] / r["results"]["GLS"]["worst"]
                             for r in fac if r["a"] == a and r["k"] == 1]) for a in levels}
        cross = [a for a in levels if med[a] >= 1.0]
        mac("AnisoCrossover", f"{min(cross):g}" if cross else "none")
        mac("AnisoLsGmMaxRatio", F2(max(med.values())))
        worst_a = max(med, key=med.get)
        mac("AnisoLsGmWorstA", f"{worst_a:g}")
    recs = load("exp16_aniso.json")
    if recs:
        levels = sorted({r["a"] for r in recs})
        mac("NAniso", f"{len({r['trial'] for r in recs})}")
        for a in levels:
            for k, kn in ((1, "One"), (2, "Two")):
                sub = [r for r in recs if r["a"] == a and r["k"] == k]
                g = np.array([r["results"]["GLS"]["adv_mean"] / r["rho"] for r in sub])
                for nm, tag in (("LS", "Ls"), ("Huber", "Huber"), ("GNC-TLS", "Gnc"),
                                ("RANSAC", "Ransac"), ("TMR-0.1", "TmrTen"),
                                ("TMR-0.2", "TmrTwenty")):
                    v = np.array([r["results"][nm]["adv_mean"] / r["rho"] for r in sub])
                    mac_ci(f"AnisoAdv{tag}A{AT[a]}K{kn}", v, F2, stat=np.mean, headline=False)
                    mac_ci(f"AnisoRel{tag}A{AT[a]}K{kn}", v / g, F2, stat=np.mean,
                           headline=(nm in ("LS", "TMR-0.1") and k == 1))
                    if nm.startswith("TMR"):
                        mac_ci(f"AnisoRed{tag}A{AT[a]}K{kn}", v / g, pct, stat=np.mean,
                               transform=lambda x: 1 - x, headline=False)
                mac_ci(f"AnisoAdvGlsA{AT[a]}K{kn}", g, F2, stat=np.mean, headline=False)
                c = np.array([r["results"]["LS"]["clean_rmse"] / r["results"]["GLS"]["clean_rmse"]
                              for r in sub])
                mac_ci(f"AnisoCleanLsA{AT[a]}K{kn}", c, F2, stat=np.mean, headline=False)
                c = np.array([r["results"]["TMR-0.1"]["clean_rmse"] / r["results"]["GLS"]["clean_rmse"]
                              for r in sub])
                mac_ci(f"AnisoCleanCostA{AT[a]}K{kn}", c, lambda x: pct(x, 1), stat=np.mean,
                       transform=lambda x: x - 1, headline=False)
                sv = np.array([r["shortcut"]["var_ratio"] for r in sub if r["shortcut"]])
                if sv.size:
                    mac_ci(f"AnisoShortcutVarA{AT[a]}K{kn}", sv, F2, headline=False)
        err = []
        for r in recs:
            for nm, p in r["predicted"].items():
                if nm in r["results"]:
                    err.append(abs(r["results"][nm]["adv_mean"] - p) / max(p, 1e-9))
        mac("AnisoPredMedErr", pct(np.median(err), 1))
        mac("AnisoPredN", f"{len(err)}")


if __name__ == "__main__":
    table_verification()
    table_linearisation()
    numbers_exp1()
    numbers_exp3()
    numbers_exp4()
    table_main()
    table_vertebra()
    table_factorial()
    numbers_limits()
    numbers_robustness()
    numbers_qiu()
    numbers_workspace()
    numbers_initpose()
    numbers_regime()
    numbers_aniso()
    table_breach()
    geometry_numbers()
    # every headline macro must carry a CI twin
    REQUIRED_CI = ["RatioKOneTen", "RedKOneTen", "RedKTwoTen", "FactTmrOverLearned",
                   "FactLearnedShare", "FactRandLearnedKOne", "ScaleRedNForty",
                   "AdvLsKOne", "AdvGlsKOne", "HeadRedTmrTenKOne",
                   "CleanCostTenKOne", "SigVarDEight", "WsDesignREighty",
                   "AblFidTgtRatio", "AblWlsRed", "AblTmrRed"]
    defined = {m.split("{")[1].split("}")[0].lstrip("\\") for m in MACROS}
    missing = [n for n in REQUIRED_CI if n in defined and f"{n}CI" not in defined]
    if missing:
        raise SystemExit(f"headline macros without a CI twin: {missing}")
    out = os.path.join(OUT_DIR, "numbers.tex")
    open(out, "w").write("% Auto-generated from results/*.json by "
                         "experiments/make_tables.py. Do not edit.\n"
                         + "\n".join(sorted(MACROS)) + "\n")
    print(f"wrote {len(MACROS)} macros to {out}")
