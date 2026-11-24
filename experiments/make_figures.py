"""Publication figures.  Every number is read from results/*.json."""
from __future__ import annotations

import json
import os
import sys

import matplotlib as mpl
import numpy as np

mpl.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
RES, FIG = os.path.join(ROOT, "results"), os.path.join(ROOT, "figures")
os.makedirs(FIG, exist_ok=True)

# Okabe-Ito, colour-blind safe; every series also carries a marker or hatch
OI = dict(black="#000000", orange="#E69F00", skyblue="#56B4E9",
          green="#009E73", yellow="#F0E442", blue="#0072B2",
          vermillion="#D55E00", purple="#CC79A7", grey="#8C8C8C")

mpl.rcParams.update({
    "font.size": 8, "axes.labelsize": 8, "axes.titlesize": 8.5,
    "xtick.labelsize": 7, "ytick.labelsize": 7, "legend.fontsize": 7,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.linewidth": 0.7, "lines.linewidth": 1.3, "figure.dpi": 200,
    "savefig.dpi": 400, "legend.frameon": False, "axes.grid": True,
    "grid.alpha": 0.25, "grid.linewidth": 0.5,
    "font.family": "serif", "mathtext.fontset": "cm",
    # IEEE PDF eXpress rejects Type 3 fonts in vector figures.
    "pdf.fonttype": 42, "ps.fonttype": 42,
})
MM = 1 / 25.4
W1, W2 = 89 * MM, 183 * MM      # single- and double-column widths


def load(name):
    p = os.path.join(RES, name)
    return json.load(open(p)) if os.path.exists(p) else None


def save(fig, name):
    fig.savefig(os.path.join(FIG, name + ".pdf"), bbox_inches="tight",
                pad_inches=0.01)
    fig.savefig(os.path.join(FIG, name + ".png"), bbox_inches="tight",
                pad_inches=0.01)
    plt.close(fig)
    print("  wrote", name)


# --------------------------------------------------------------------------- #
def fig_anatomy():
    from tmr.anatomy import build_vertebra
    from tmr.geometry import stack_jacobians, target_jacobian
    from tmr.estimators import gls_influence, tmr_influence
    from experiments.exp5_vertebra import SIGMA_BY_LANDMARK

    v = build_vertebra()
    names = list(SIGMA_BY_LANDMARK)
    P = v.fiducials(names)
    sig = np.array([SIGMA_BY_LANDMARK[n] for n in names])
    S = np.diag(np.repeat(sig ** 2, 3))
    n = len(P)
    A = stack_jacobians(np.eye(3), P)
    G = target_jacobian(np.eye(3), v.tip["left"])
    M_gls = gls_influence(A, G, S)
    M_tmr = tmr_influence(A, G, S, np.ones(n), k=1, eta=0.10).M
    lev = lambda M: np.linalg.norm(M.reshape(3, n, 3).transpose(1, 0, 2),
                                   ord=2, axis=(1, 2))

    fig = plt.figure(figsize=(W2, 66 * MM), layout="constrained")
    gs = fig.add_gridspec(1, 3, width_ratios=[1, 1, 1.25])
    X = v.nodes
    for j, (a, b, la, lb, ttl) in enumerate([
            (0, 1, "left\u2013right $x$ (mm)", "posterior\u2013anterior $y$ (mm)",
             "axial view"),
            (1, 2, "posterior\u2013anterior $y$ (mm)", "caudal\u2013cranial $z$ (mm)",
             "sagittal view")]):
        ax = fig.add_subplot(gs[0, j])
        ax.scatter(X[::2, a], X[::2, b], s=0.6, c=OI["grey"], alpha=0.5,
                   lw=0, rasterized=True)
        for s_ in ("left", "right"):
            tr = v.trajectory(s_, 40)
            ax.plot(tr[:, a], tr[:, b], color=OI["vermillion"], lw=1.6,
                    solid_capstyle="round",
                    label="planned screw axis" if (j == 0 and s_ == "left")
                    else None)
            ax.plot(v.tip[s_][a], v.tip[s_][b], marker="*", ms=8,
                    color=OI["vermillion"], mec="k", mew=0.4,
                    label="target (screw tip)" if (j == 0 and s_ == "left")
                    else None)
        ax.scatter(P[:, a], P[:, b], s=26, marker="o", facecolor=OI["blue"],
                   edgecolor="k", lw=0.4, zorder=5,
                   label="fiducial landmark" if j == 0 else None)
        ax.set(xlabel=la, ylabel=lb, title=ttl)
        ax.set_aspect("equal")
        if j == 0:
            ax.legend(loc="lower left", bbox_to_anchor=(-0.02, -0.02),
                      fontsize=6.3, handlelength=1.4)

    ax = fig.add_subplot(gs[0, 2])
    idx = np.arange(n)
    wdt = 0.38
    ax.bar(idx - wdt / 2, lev(M_gls), wdt, color=OI["skyblue"],
           edgecolor="k", lw=0.4, label="GLS (Gauss\u2013Markov)")
    ax.bar(idx + wdt / 2, lev(M_tmr), wdt, color=OI["orange"], hatch="//",
           edgecolor="k", lw=0.4, label=r"TMR ($\eta=0.10$)")
    ax.set_xticks(idx)
    ax.set_xticklabels([nm.replace("_", " ") for nm in names], rotation=45,
                       ha="right")
    ax.set_ylabel(r"target leverage $\|M_i\|_2$  (mm / mm)")
    ax.set_title("per-landmark target leverage")
    ax.legend(loc="upper left")
    save(fig, "fig1_anatomy")


def fig_stealth():
    recs = load("exp3_stealth.json")
    if not recs:
        return
    rhos = [s["rho"] for s in recs[0]["sweep"]]
    get = lambda f: np.array([[f(s) for s in r["sweep"]] for r in recs])
    auc, disp = get(lambda s: s["auc"]), get(lambda s: s["disp_mean"])
    frer = get(lambda s: s["fre_ratio"])

    fig, axes = plt.subplots(1, 2, figsize=(W2 * 0.82, 60 * MM),
                             layout="constrained")
    ax = axes[0]
    ax.axhspan(0.5, 0.75, color=OI["yellow"], alpha=0.3, lw=0)
    ax.text(0.28, 0.60, "stealth window\n(detector near chance)", fontsize=6.5,
            color="#6b5900")
    m, lo, hi = (np.median(auc, 0), np.percentile(auc, 10, 0),
                 np.percentile(auc, 90, 0))
    ax.fill_between(rhos, lo, hi, color=OI["blue"], alpha=0.18, lw=0)
    l1, = ax.plot(rhos, m, "o-", color=OI["blue"], ms=3.2,
                  label="residual detector AUC (left)")
    ax.axhline(0.5, color="k", lw=0.6, ls=":")
    ax.set(xscale="log", xlabel=r"corruption budget $\rho$ (mm)",
           ylabel="detector AUC", ylim=(0.42, 1.04),
           title="how visible is the corruption?")
    ax2 = ax.twinx()
    ax2.spines["right"].set_visible(True)
    l2, = ax2.plot(rhos, np.median(frer, 0), "s--", color=OI["vermillion"],
                   ms=3.2, label="FRE inflation (right)")
    ax2.set_ylabel(r"FRE$_{\rm attacked}$ / FRE$_{\rm clean}$",
                   color=OI["vermillion"])
    ax2.tick_params(axis="y", colors=OI["vermillion"], labelsize=7)
    ax2.grid(False)
    ax.legend(handles=[l1, l2], loc="upper left")

    ax = axes[1]
    md, ma = np.median(disp, 0), np.median(auc, 0)
    ax.plot(ma, md, "o-", color=OI["green"], ms=3.6)
    for j, r in enumerate(rhos):
        off = {0.25: (6, 2), 0.5: (6, -8), 1.0: (6, -8), 2.0: (-4, 10),
               5.0: (-30, 2), 12.0: (-34, -2)}
        if r in off:
            ax.annotate(rf"$\rho={r:g}$", (ma[j], md[j]),
                        textcoords="offset points", xytext=off[r],
                        fontsize=6.2)
    ax.axvspan(0.5, 0.75, color=OI["yellow"], alpha=0.3, lw=0)
    ax.axhline(1.0, color=OI["grey"], lw=0.7, ls="--")
    ax.text(0.50, 1.12, "1 mm", fontsize=6.2, color=OI["grey"], ha="left")
    ax.set(xlabel="detectability of the corrupted fiducial (AUC)",
           ylabel="target displacement (mm)",
           title="damage achievable at each detectability")
    save(fig, "fig2_stealth")


def fig_pareto():
    recs = load("exp1_synthetic.json")
    if not recs:
        return
    etas = ["0", "0.02", "0.05", "0.1", "0.2", "0.5", "1", "1e+06"]
    fig, axes = plt.subplots(1, 2, figsize=(W2 * 0.82, 62 * MM),
                             layout="constrained", sharey=True)
    for ax, k in zip(axes, (1, 2)):
        sub = [r for r in recs if r["k"] == k]
        xs, ys, lo, hi = [], [], [], []
        for e in etas:
            v = np.array([r["tmr"][e]["var"] / r["gls"]["var"] for r in sub])
            rr = np.array([r["tmr"][e]["risk"] / r["gls"]["risk"] for r in sub])
            xs.append(np.median(v)); ys.append(np.median(rr))
            lo.append(np.percentile(rr, 25)); hi.append(np.percentile(rr, 75))
        ax.fill_between(xs, lo, hi, color=OI["orange"], alpha=0.2, lw=0)
        ax.plot(xs, ys, "o-", color=OI["orange"], ms=3.4,
                label="TMR frontier (median, IQR)")
        for j, e in enumerate(etas):
            if e in ("0", "0.05", "0.1", "0.2", "1e+06"):
                lab = r"$\eta\!\to\!\infty$" if e == "1e+06" else rf"$\eta={float(e):g}$"
                ax.annotate(lab, (xs[j], ys[j]), textcoords="offset points",
                            xytext=(4, 5), fontsize=6.2)
        ls = np.array([r["ls"]["risk"] / r["gls"]["risk"] for r in sub])
        lv = np.array([r["ls"]["var"] / r["gls"]["var"] for r in sub])
        ax.plot(np.median(lv), np.median(ls), marker="s", ms=5,
                color=OI["skyblue"], mec="k", mew=0.4, ls="none",
                label="unweighted LS")
        lb = np.array([(k / r["n"]) * r["kappa"] / r["gls"]["risk"]
                       for r in sub])
        ax.axhline(np.median(lb), color=OI["vermillion"], ls="--", lw=1.0,
                   label=r"lower bound $\frac{k}{n}\rho\,\kappa_x$")
        ax.axhline(1.0, color="k", lw=0.6, ls=":")
        ax.set(xlabel=r"clean target variance / GLS",
               title=rf"$k={k}$ corrupted fiducial" + ("s" if k > 1 else ""))
        if k == 1:
            ax.set_ylabel("worst-case target displacement / GLS")
            ax.legend(loc="lower left", bbox_to_anchor=(0.02, 0.02))
    save(fig, "fig3_pareto")


def fig_headtohead():
    recs = load("exp2_nonlinear.json")
    if not recs:
        print("  (exp2 not ready)")
        return
    order = ["LS", "GLS", "Huber", "RANSAC", "GNC-TLS",
             "TMR-0.05", "TMR-0.1", "TMR-0.2"]
    style = {"LS": (OI["skyblue"], "s"), "GLS": (OI["blue"], "D"),
             "Huber": (OI["green"], "^"), "RANSAC": (OI["purple"], "v"),
             "GNC-TLS": (OI["yellow"], "P"), "TMR-0.05": (OI["orange"], "o"),
             "TMR-0.1": (OI["vermillion"], "o"), "TMR-0.2": (OI["black"], "o")}
    fig, axes = plt.subplots(1, 2, figsize=(W2 * 0.86, 64 * MM),
                             layout="constrained")
    ax = axes[0]
    sub = [r for r in recs if r["k"] == 1]
    for nm in order:
        c = np.array([r["results"][nm]["clean_rmse"] for r in sub])
        a = np.array([r["results"][nm]["adv_mean"] / r["rho"] for r in sub])
        col, mk = style[nm]
        ax.errorbar(c.mean(), a.mean(),
                    xerr=c.std() / np.sqrt(len(c)),
                    yerr=a.std() / np.sqrt(len(a)),
                    marker=mk, ms=6, color=col, mec="k", mew=0.5,
                    ecolor=col, elinewidth=0.8, capsize=1.6, ls="none",
                    label=nm)
        ax.annotate(nm, (c.mean(), a.mean()), textcoords="offset points",
                    xytext=(7, -1), fontsize=6.3)
    ax.set(xlabel="clean target RMSE (mm)",
           ylabel=r"worst-case target displacement / $\rho$",
           title=r"$k=1$: robustness vs. clean accuracy")
    ax.annotate("better", xy=(0.07, 0.07), xytext=(0.30, 0.28),
                xycoords="axes fraction", textcoords="axes fraction",
                fontsize=7, ha="center",
                arrowprops=dict(arrowstyle="->", lw=0.9))

    ax = axes[1]
    pv, mv, cols = [], [], []
    for r in recs:
        for nm, p in r["predicted"].items():
            if nm in r["results"]:
                pv.append(p / r["rho"])
                mv.append(r["results"][nm]["adv_mean"] / r["rho"])
                cols.append(style.get(nm, (OI["grey"], "o"))[0])
    pv, mv = np.array(pv), np.array(mv)
    lim = [0, max(pv.max(), mv.max()) * 1.05]
    ax.plot(lim, lim, "k--", lw=0.8, label="exact agreement")
    ax.scatter(pv, mv, s=9, c=cols, edgecolor="k", lw=0.2, alpha=0.85)
    rel = np.abs(mv - pv) / np.maximum(pv, 1e-9)
    ax.set(xlabel=r"first-order prediction $R_k(M)/\rho$",
           ylabel=r"measured nonlinear attack / $\rho$",
           title="theory predicts the realised attack")
    ax.text(0.04, 0.93, f"median relative error {np.median(rel):.2%}\n"
                        f"95th percentile {np.percentile(rel, 95):.2%}\n"
                        f"$n={len(pv)}$ estimator\u2013instance pairs",
            transform=ax.transAxes, fontsize=6.4, va="top")
    ax.legend(loc="lower right")
    save(fig, "fig4_headtohead")


def fig_ablations():
    recs = load("exp4_ablations.json")
    if not recs:
        return
    g = lambda f: np.array([f(r) for r in recs])
    fig, axes = plt.subplots(1, 4, figsize=(W2, 55 * MM),
                             layout="constrained")

    ax = axes[0]
    etas = ["0.05", "0.1", "0.2", "1", "1e+06"]
    xv = [0, 1, 2, 3, 4]
    y = [np.median(g(lambda r: r["A1"]["sweep"][e]["risk"] / r["A1"]["gls_risk"]))
         for e in etas]
    q1 = [np.percentile(g(lambda r: r["A1"]["sweep"][e]["risk"] / r["A1"]["gls_risk"]), 25) for e in etas]
    q3 = [np.percentile(g(lambda r: r["A1"]["sweep"][e]["risk"] / r["A1"]["gls_risk"]), 75) for e in etas]
    ax.fill_between(xv, q1, q3, color=OI["orange"], alpha=0.2, lw=0)
    ax.plot(xv, y, "o-", color=OI["orange"], ms=3.4)
    ax.set_xticks(xv)
    ax.set_xticklabels(["0.05", "0.1", "0.2", "1", r"$\infty$"])
    ax.set(xlabel=r"clean-efficiency budget $\eta$",
           ylabel="risk / GLS", title="A1  budget saturates")

    ax = axes[1]
    vals = [np.median(g(lambda r: r["A2"]["target_risk_gls"])),
            np.median(g(lambda r: r["A2"]["target_risk_fid"])),
            np.median(g(lambda r: r["A2"]["target_risk_tmr"]))]
    base = np.median(g(lambda r: r["A2"]["target_risk_gls"]))
    ax.bar(range(3), np.array(vals) / base,
           color=[OI["blue"], OI["purple"], OI["orange"]],
           hatch=["", "\\\\", "//"], edgecolor="k", lw=0.4)
    ax.set_xticks(range(3))
    ax.set_xticklabels(["GLS", "fiducial\nminimax", "target\nminimax"],
                       fontsize=5.8, rotation=0, ha="center")
    ax.set(ylabel="target risk / GLS", title="A2  which loss to robustify")

    ax = axes[2]
    K = np.array([[np.median(g(lambda r: r["A5"][f"design{kd}"][f"attack{ka}"]
                               / r["A5"]["gls"][f"attack{ka}"]))
                   for ka in (1, 2, 3)] for kd in (1, 2, 3)])
    im = ax.imshow(K, cmap="viridis_r", vmin=K.min(), vmax=1.0)
    for i in range(3):
        for j in range(3):
            ax.text(j, i, f"{K[i, j]:.2f}", ha="center", va="center",
                    fontsize=6.4,
                    color="w" if K[i, j] < 0.87 else "k")
    ax.set_xticks(range(3), [1, 2, 3]); ax.set_yticks(range(3), [1, 2, 3])
    ax.set(xlabel="attack $k$", ylabel="design $k$",
           title="A5  threat mismatch")
    ax.grid(False)
    fig.colorbar(im, ax=ax, shrink=0.82, label="risk / GLS")

    ax = axes[3]
    v = [1.0,
         np.median(g(lambda r: r["A6"]["risk_wls"] / r["A6"]["risk_gls"])),
         np.median(g(lambda r: r["A6"]["risk_tmr"] / r["A6"]["risk_gls"]))]
    ax.bar(range(3), v, color=[OI["blue"], OI["green"], OI["orange"]],
           hatch=["", "..", "//"], edgecolor="k", lw=0.4)
    ax.set_xticks(range(3))
    ax.set_xticklabels(["GLS", "scalar\nweights", "influence\ndesign"],
                       fontsize=5.8, rotation=0, ha="center")
    ax.set(ylabel="risk / GLS", title="A6  reweighting is not enough")
    ax.set_ylim(0, 1.1)
    save(fig, "fig5_ablations")


def fig_vertebra():
    recs = load("exp5_vertebra.json")
    if not recs:
        print("  (exp5 not ready)")
        return
    order = ["LS", "GLS", "Huber", "RANSAC", "GNC-TLS", "TMR-0.05",
             "TMR-0.1", "TMR-0.2", "TMR-traj-0.1"]
    cols = [OI["skyblue"], OI["blue"], OI["green"], OI["purple"], OI["yellow"],
            OI["orange"], OI["vermillion"], OI["black"], OI["grey"]]
    hat = ["", "", "", "", "", "//", "//", "//", "xx"]
    fig, axes = plt.subplots(1, 3, figsize=(W2, 56 * MM), layout="constrained")
    for ax, key, ttl, ylab in [
            (axes[0], "clean_rmse", "clean accuracy",
             "clean target RMSE (mm)"),
            (axes[1], "adv_mean", "worst case: screw tip",
             r"tip displacement / $\rho$"),
            (axes[2], "traj_adv_mean", "worst case: whole screw axis",
             r"max axis displacement / $\rho$")]:
        sub = [r for r in recs if r["k"] == 1]
        vals, errs = [], []
        for nm in order:
            v = np.array([r["results"][nm][key] / (1.0 if key == "clean_rmse"
                                                   else r["rho"]) for r in sub])
            vals.append(v.mean()); errs.append(v.std() / np.sqrt(len(v)))
        ax.bar(range(len(order)), vals, yerr=errs, color=cols, hatch=hat,
               edgecolor="k", lw=0.4, capsize=1.8,
               error_kw=dict(elinewidth=0.7))
        ax.set_xticks(range(len(order)))
        ax.set_xticklabels(order, rotation=45, ha="right", fontsize=6.2)
        ax.set(ylabel=ylab, title=ttl)
    save(fig, "fig6_vertebra")


def fig_robustness():
    sc, sg = load("exp10_scaling.json"), load("exp11_sigma.json")
    if not sc or not sg:
        print("  (exp10/exp11 not ready)")
        return
    fig, axes = plt.subplots(1, 2, figsize=(W2 * 0.84, 58 * MM),
                             layout="constrained")
    ax = axes[0]
    ns = sorted({r["n"] for r in sc})
    for k, col, mk in ((1, OI["vermillion"], "o"), (2, OI["blue"], "s")):
        med, lo, hi = [], [], []
        for n in ns:
            v = np.array([r["tmr"] / r["gls"] for r in sc
                          if r["n"] == n and r["k"] == k])
            med.append(np.median(v)); lo.append(np.percentile(v, 25))
            hi.append(np.percentile(v, 75))
        ax.fill_between(ns, lo, hi, color=col, alpha=0.15, lw=0)
        ax.plot(ns, med, marker=mk, ms=3.6, color=col, label=f"$k={k}$")
    ax.axhline(1.0, color="k", lw=0.6, ls=":")
    ax.set(xscale="log", xlabel="number of fiducials $n$",
           ylabel="worst-case risk / GLS", ylim=(0.35, 1.03),
           title="the advantage grows with $n$")
    ax.set_xticks(ns); ax.set_xticklabels([str(n) for n in ns], fontsize=6.2)
    ax.minorticks_off(); ax.legend(loc="upper right")

    ax = axes[1]
    ds = sorted({r["delta"] for r in sg})
    rr = [np.median([r["risk_tmr_hat"] / r["risk_gls_hat"] for r in sg
                     if r["delta"] == d and r["k"] == 1]) for d in ds]
    vr = [np.median([r["var_tmr_hat"] / r["var_gls_hat"] for r in sg
                     if r["delta"] == d and r["k"] == 1]) for d in ds]
    ax.plot(ds, rr, "o-", color=OI["vermillion"], ms=3.6,
            label="worst-case risk / GLS")
    ax.plot(ds, vr, "s--", color=OI["blue"], ms=3.6,
            label="clean variance / GLS")
    ax.axhline(1.0, color="k", lw=0.6, ls=":")
    ax.annotate("TMR becomes\nmore efficient", xy=(0.8, 0.86),
                xytext=(0.42, 0.62), fontsize=6.3,
                arrowprops=dict(arrowstyle="->", lw=0.8))
    ax.set(xlabel=r"covariance misspecification $\delta$",
           ylabel=r"ratio to GLS (both under true $\Sigma$)",
           title=r"misspecifying $\Sigma$ costs GLS more")
    ax.legend(loc="upper right", fontsize=6.3)
    save(fig, "fig7_robustness")


def fig_regime():
    """Where the method does not help: TMR/GM against the slack of GM above the
    estimator-independent floor, and against the four swept variables."""
    recs = load("exp15_regime.json")
    if not recs:
        print("  (exp15 not ready)")
        return
    fig, axes = plt.subplots(1, 5, figsize=(W2, 50 * MM), layout="constrained",
                             gridspec_kw=dict(width_ratios=[1.35, 1, 1, 1, 1]))
    cols = {1: OI["vermillion"], 2: OI["blue"]}
    mks = {1: "o", 2: "s"}
    ax = axes[0]
    for k in (1, 2):
        sub = [r for r in recs if r["k"] == k and r["limit"] > 1e-9]
        sl = np.array([r["gls"] / r["limit"] for r in sub])
        rt = np.array([r["tmr"] / r["gls"] for r in sub])
        ax.scatter(sl, rt, s=4, alpha=0.35, color=cols[k], marker=mks[k],
                   lw=0, label=f"$k={k}$", rasterized=True)
    x = np.linspace(1, max(r["gls"] / max(r["limit"], 1e-9) for r in recs), 200)
    ax.plot(x, 1 / x, color="k", lw=0.8, ls="--", label="floor $1/\\mathrm{slack}$")
    ax.axhline(1.0, color="k", lw=0.6, ls=":")
    ax.set(xscale="log", xlabel="GM slack $R_k(\\mathrm{GM})/\\rho\\kappa^{(k)}$",
           ylabel="worst-case risk, TMR / GM", ylim=(0.2, 1.05),
           title="gain $\\leq$ slack above the floor")
    ax.set_xticks([1, 1.5, 2, 3, 4, 6])
    ax.set_xticklabels(["1", "1.5", "2", "3", "4", "6"])
    ax.minorticks_off()
    ax.legend(loc="upper right", fontsize=6.3)

    panels = [("target", "target distance (mm)", "target position", False),
              ("n", "fiducials $n$", "fiducial count", True),
              ("cond", "aspect ratio (1 = cube)", "flatness of the cloud", True),
              ("noise", "hetero. spread $s$", "noise structure", False)]
    for ax, (sw, xl, ttl, logx) in zip(axes[1:], panels):
        sub = [r for r in recs if r["sweep"] == sw]
        levels = [lv for lv in dict.fromkeys(r["level"] for r in sub)
                  if lv != "outlier"]
        levels = sorted(levels, key=float)
        for k in (1, 2):
            med, lo, hi, xs = [], [], [], []
            for lv in levels:
                v = np.array([r["tmr"] / r["gls"] for r in sub
                              if r["level"] == lv and r["k"] == k])
                if v.size == 0:
                    continue
                xs.append(float(lv) if sw != "target" else float(lv))
                med.append(np.median(v)); lo.append(np.percentile(v, 25))
                hi.append(np.percentile(v, 75))
            xs = np.array(xs)
            if sw == "target":
                xs = np.where(xs == 0, 5.0, xs)      # centroid drawn at 5 mm on the log axis
            ax.fill_between(xs, lo, hi, color=cols[k], alpha=0.15, lw=0)
            ax.plot(xs, med, marker=mks[k], ms=3.2, color=cols[k], label=f"$k={k}$")
        if sw == "target":
            hull = np.median([r["hull_dist"] + r["target_dist"] for r in sub
                              if r["level"] == 80.0])
            # typical hull radius along the target ray: where hull_dist crosses 0
            ins = [r["target_dist"] for r in sub if r["hull_dist"] < 0]
            outs = [r["target_dist"] for r in sub if r["hull_dist"] >= 0]
            if ins and outs:
                ax.axvspan(5.0, max(ins), color=OI["grey"], alpha=0.12, lw=0)
                ax.text(5.5, 0.27, "inside\nhull", fontsize=6, color=OI["grey"])
            ax.set(xscale="log")
            ax.set_xticks([5, 10, 20, 40, 80, 160])
            ax.set_xticklabels(["0", "10", "20", "40", "80", "160"], fontsize=6.2)
            ax.minorticks_off()
        elif logx:
            ax.set(xscale="log")
            ax.set_xticks(xs); ax.set_xticklabels([f"{v:g}" for v in xs], fontsize=6.2)
            ax.minorticks_off()
        ax.axhline(1.0, color="k", lw=0.6, ls=":")
        ax.set(xlabel=xl, ylim=(0.2, 1.05), title=ttl)
        if sw == "n":
            ax.legend(loc="upper right", fontsize=6.3)
    for ax in axes[1:]:
        ax.set_yticklabels([])
    save(fig, "fig8_regime")


def fig_aniso():
    """Direct attack under camera-axis anisotropic noise, variance ratio a."""
    recs = load("exp16_aniso.json")
    fac = load("exp16_aniso_factorial.json")
    if not recs or not fac:
        print("  (exp16 not ready)")
        return
    fig, axes = plt.subplots(1, 3, figsize=(W2 * 0.9, 50 * MM), layout="constrained")
    levels = sorted({r["a"] for r in recs})
    style = {"LS": (OI["black"], "o", "unweighted LS"),
             "GLS": (OI["grey"], "s", "Gauss--Markov"),
             "Huber": (OI["orange"], "^", "Huber"),
             "GNC-TLS": (OI["purple"], "v", "GNC-TLS"),
             "TMR-0.1": (OI["vermillion"], "D", "TMR $\\eta{=}0.10$")}
    for ax, field, ttl, yl in ((axes[0], "clean_rmse", "clean accuracy",
                                "clean target RMSE (mm)"),
                               (axes[1], "adv_mean", "worst case under attack, $k=1$",
                                "displacement / $\\rho$ (mm)")):
        for nm, (col, mk, lab) in style.items():
            med, lo, hi = [], [], []
            for a in levels:
                v = np.array([r["results"][nm][field] / (r["rho"] if field == "adv_mean" else 1)
                              for r in recs if r["a"] == a and r["k"] == 1])
                med.append(np.mean(v)); lo.append(np.percentile(v, 25))
                hi.append(np.percentile(v, 75))
            ax.fill_between(levels, lo, hi, color=col, alpha=0.12, lw=0)
            ax.plot(levels, med, marker=mk, ms=3.4, color=col, label=lab)
        ax.set(xscale="log", xlabel="variance ratio $a$ (depth : lateral)",
               ylabel=yl, title=ttl)
        ax.set_xticks(levels); ax.set_xticklabels([f"{a:g}" for a in levels])
        ax.minorticks_off()
    fig.legend(*axes[0].get_legend_handles_labels(), loc="lower center", ncol=5,
               fontsize=6.3, bbox_to_anchor=(0.36, -0.06))
    ax = axes[2]
    sw = sorted({r["a"] for r in fac})
    for k, col, mk in ((1, OI["vermillion"], "o"), (2, OI["blue"], "s")):
        med, lo, hi = [], [], []
        for a in sw:
            v = np.array([r["results"]["LS"]["worst"] / r["results"]["GLS"]["worst"]
                          for r in fac if r["a"] == a and r["k"] == k])
            med.append(np.median(v)); lo.append(np.percentile(v, 25))
            hi.append(np.percentile(v, 75))
        ax.fill_between(sw, lo, hi, color=col, alpha=0.15, lw=0)
        ax.plot(sw, med, marker=mk, ms=3.4, color=col, label=f"LS / GM, $k={k}$")
    for k, col, mk in ((1, OI["green"], "D"),):
        med = []
        for a in sw:
            v = [r["results"]["TMR (full)"]["worst"] / r["results"]["GLS"]["worst"]
                 for r in fac if r["a"] == a and r["k"] == k
                 and "TMR (full)" in r["results"]]
            med.append(np.median(v) if v else np.nan)
        ax.plot(sw, med, marker=mk, ms=3.4, color=col, ls="--", label="TMR / GM, $k=1$")
    ax.axhline(1.0, color="k", lw=0.6, ls=":")
    ax.set(xscale="log", xlabel="variance ratio $a$ (depth : lateral)",
           ylabel="risk ratio (closed form)",
           title="is LS still safer than GM?")
    ax.set_xticks(sw); ax.set_xticklabels([f"{a:g}" for a in sw], fontsize=6.2)
    ax.minorticks_off(); ax.legend(loc="upper right", fontsize=6.3)
    save(fig, "fig9_aniso")


if __name__ == "__main__":
    which = sys.argv[1:] or ["anatomy", "stealth", "pareto", "headtohead",
                             "ablations", "vertebra", "robustness", "regime",
                             "aniso"]
    for w in which:
        print("figure:", w)
        globals()["fig_" + w]()
