"""
revision_figures.py
Figures for the scope and robustness studies (reads revision_results.json
written by revision_studies.py):
  fig8_scope.pdf       main-text figure: limits of the window law
  figS8_extensions.pdf supplementary figure: area ratio, ferroelectric-side
                       charge, switched polarization, back gate
"""
import json
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker
import params as P

FIGDIR = os.path.join(os.path.dirname(__file__), "..", "figures")
os.makedirs(FIGDIR, exist_ok=True)
R = json.load(open("revision_results.json"))
OI = {"blue": "#0072B2", "orange": "#E69F00", "green": "#009E73",
      "red": "#D55E00", "purple": "#CC79A7", "sky": "#56B4E9",
      "black": "#000000", "grey": "#777777"}
plt.rcParams.update({
    "font.size": 9, "axes.labelsize": 9.5, "legend.fontsize": 7.2,
    "xtick.labelsize": 8.5, "ytick.labelsize": 8.5, "lines.linewidth": 1.3,
    "axes.spines.top": False, "axes.spines.right": False,
    "legend.framealpha": 0.9, "legend.edgecolor": "0.85",
    "legend.fancybox": False, "font.family": "serif",
    "font.serif": ["Times New Roman", "Liberation Serif", "STIXGeneral",
                   "DejaVu Serif"], "mathtext.fontset": "stix",
    "pdf.fonttype": 42, "ps.fonttype": 42})


def save(fig, name):
    fig.savefig(os.path.join(FIGDIR, name + ".pdf"), bbox_inches="tight")
    fig.savefig(os.path.join(FIGDIR, name + ".png"), bbox_inches="tight",
                dpi=200)
    plt.close(fig)
    print("saved", name)


def label(ax, s, dx=-0.17, dy=1.10):
    ax.text(dx, dy, s, transform=ax.transAxes, fontweight="bold",
            fontsize=10, va="top")


# ---------------------------------------------------------------- Fig. 8
fig, axs = plt.subplots(2, 2, figsize=(7.0, 5.3))

ax = axs[0, 0]
rows = R["S1_grid_refinement"]
cases = []
for r in rows:
    if r["case"] not in cases:
        cases.append(r["case"])
nice = {"eps = -1.0%": r"$\epsilon=-1\%$", "eps = 0": r"$\epsilon=0$",
        "eps = +0.90%": r"$\epsilon=+0.9\%$",
        "t_FE = 86.5 nm": r"$t_{\rm FE}=86.5$ nm",
        "Dit = 1e12": r"$D_{\rm it}=10^{12}$", "Dit = 1e13": r"$D_{\rm it}=10^{13}$"}
cols = [OI["blue"], OI["black"], OI["sky"], OI["green"], OI["orange"],
        OI["red"]]
mk = ["o", "s", "^", "D", "v", "P"]
for c, col, m in zip(cases, cols, mk):
    rr = [r for r in rows if r["case"] == c]
    ax.loglog([abs(r["density_mismatch_pct"]) for r in rr],
              [max(abs(r["err_law_pct"]), 1e-5) for r in rr],
              ls="none", marker=m, color=col, ms=4.5,
              label=nice.get(c.strip(), c.strip()))
xx = np.logspace(-3.3, 1.0, 20)
ax.loglog(xx, 0.012 * xx, color="0.7", lw=0.8, zorder=0)
ax.text(1.2, 0.006, "slope 1", fontsize=6.5, color="0.45")
ax.set_xlabel(r"Density mismatch $|p_{\rm f}/p_{\rm b}-1|$ (%)")
ax.set_ylabel("|Eq. (7) $-$ simulation| (%)")
ax.set_ylim(5e-5, 1)
ax.legend(ncol=2, fontsize=6.3, loc="upper left", handlelength=1.2,
          columnspacing=0.8)
label(ax, "(a)")

ax = axs[0, 1]
s2 = R["S2_branch_mobility_imposed"]
rat = np.array([r["mu_ratio"] for r in s2])
dmw = np.array([1e3 * (r["mw_sim"] - r["law"]) for r in s2])
cf = np.array([1e3 * r["corr_closed_form_V"] for r in s2])
rr = np.logspace(np.log10(0.45), np.log10(2.2), 80)
Cs = R["C_series_uFcm2"] * 1e-2
pf = np.mean([r["p_f"] for r in s2 if r["mu_ratio"] == 1.0])
cf_line = 1e3 * (P.kT_eV * np.log(rr) + P.q * pf * (1 - 1 / rr) / Cs)
ax.semilogx(rr, cf_line, color=OI["grey"], lw=1.0,
            label="closed form, Eq. (9)")
ax.semilogx(rat, dmw, "o", color=OI["red"], ms=4.5,
            label="simulation $-$ Eq. (7)")
ax.axhline(0, color="0.8", lw=0.6)
ax.set_xlabel(r"Mobility ratio $\mu_{\rm b}/\mu_{\rm f}$")
ax.set_ylabel("Window correction (mV)")
ax.set_xticks([0.5, 1, 2])
ax.set_xticklabels(["0.5", "1", "2"])
ax.xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
ax.legend(loc="upper left")
label(ax, "(b)")

ax = axs[1, 0]
s3 = R["S3_dos_scale"]
cqp = np.array([r["CQ_prog_uFcm2"] for r in s3])
cins = s3[0]["Cins_uFcm2"]
mw = np.array([r["mw_mean"] for r in s3])
mw1 = [r["mw_mean"] for r in s3 if r["dos_scale"] == 1.0][0]
ax.semilogx(cqp / cins, mw / mw1, "o-", color=OI["purple"], ms=4.5,
            label="channel DOS scaled")
for r, x, y in zip(s3, cqp / cins, mw / mw1):
    ax.annotate(f"$\\times${r['dos_scale']:g}", (x, y),
                textcoords="offset points", xytext=(4, -9), fontsize=6.5)
ax.axvline(10, color="0.7", lw=0.7, ls=":")
ax.set_xlabel(r"$C_{\rm Q}$ at program extreme / $C_{\rm hBN}$")
ax.set_ylabel("Window / WSe$_2$ value")
ax.set_ylim(0.45, 1.08)
label(ax, "(c)")

ax = axs[1, 1]
s3c = R["S3_read_level"]
for key, col, lab in [("eps-1.0", OI["blue"], r"$\epsilon=-1\%$"),
                      ("eps+0.0", OI["black"], r"$\epsilon=0$"),
                      ("eps+0.9", OI["red"], r"$\epsilon=+0.9\%$")]:
    xs = [r["I_crit_A_per_um"] for r in s3c if key in r]
    ys = [r[key]["mw_mean"] for r in s3c if key in r]
    ax.semilogx(np.array(xs) * 1e9, ys, "o-", color=col, ms=4, label=lab)
ax.axvline(100, color="0.7", lw=0.7, ls=":")
ax.text(110, 1.06, "read level\nused here", fontsize=6.5, color="0.4")
ax.set_xlabel(r"Read current (nA/$\mu$m)")
ax.set_ylabel("Memory window (V)")
ax.set_ylim(1.0, 1.55)
ax.legend(loc="lower left")
label(ax, "(d)")
fig.tight_layout()
save(fig, "fig8_scope")

# ------------------------------------------------------------ Fig. S8
fig, axs = plt.subplots(2, 2, figsize=(7.0, 5.2))

ax = axs[0, 0]
s4 = R["S4_area_ratio"]
ar = np.array([r["A_R"] for r in s4])
ax.semilogx(ar, [r["mw_mean"] for r in s4], "o-", color=OI["blue"], ms=4.5,
            label="clean interface")
ax.semilogx(ar, [r["with_Dit1e12"]["mw_mean"] for r in s4], "s--",
            color=OI["orange"], ms=4.5,
            label=r"$D_{\rm it}=10^{12}$ cm$^{-2}$eV$^{-1}$")
ax.set_xlabel(r"Area ratio $A_{\rm FE}/A_{\rm MIS}$")
ax.set_ylabel("Memory window (V)")
ax.set_xticks([0.25, 0.5, 1, 2, 4])
ax.set_xticklabels(["0.25", "0.5", "1", "2", "4"])
ax.xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
ax.legend(loc="lower left")
label(ax, "(a)")

ax = axs[0, 1]
s5 = R["S5_FE_side_charge"]
xlab = [f"{r['sigma_FE0_uCcm2']:.2g}\n{r['d_frac']:.1f}" for r in s5]
x = np.arange(len(s5))
ax.bar(x - 0.18, [abs(r["err_without_FE_term_pct"]) for r in s5], 0.36,
       color=OI["red"], label="Eq. (7) without the term")
ax.bar(x + 0.18, [max(abs(r["err_generalized_pct"]), 1e-3) for r in s5],
       0.36, color=OI["green"], label="with the term, Eq. (8)")
ax.set_yscale("log")
ax.set_xticks(x)
ax.set_xticklabels(xlab, fontsize=7)
ax.set_xlabel(r"$\sigma_{\rm FE,0}$ ($\mu$C/cm$^2$) / $d/t_{\rm FE}$")
ax.set_ylabel("|error| (%)")
ax.set_ylim(5e-4, 3000)
ax.legend(loc="upper left", fontsize=6.5, ncol=2)
label(ax, "(b)")

ax = axs[1, 0]
s7 = R["S7_switched_polarization"]
names = {"Pr_CIPS": r"$P_{\rm r}$", "Ec_CIPS": r"$E_{\rm c}$",
         "sigma_Ec": r"$\sigma_{E_{\rm c}}$", "eps_FE_b": r"$\varepsilon_{\rm b}$"}
facs = {"Pr_CIPS": P.Pr_CIPS, "Ec_CIPS": P.Ec_CIPS, "sigma_Ec": P.sigma_Ec,
        "eps_FE_b": P.eps_FE_b}
mw_ref = [r["mw_mean"] for r in s7 if r["param"] == "Pr_CIPS"
          and abs(r["value"] - P.Pr_CIPS) < 1e-9][0]
for (k, lab), col, m in zip(names.items(),
                            [OI["blue"], OI["red"], OI["green"],
                             OI["purple"]], ["o", "s", "^", "D"]):
    rr_ = [r for r in s7 if r["param"] == k]
    ax.semilogx([r["value"] / facs[k] for r in rr_],
                [r["mw_mean"] / mw_ref for r in rr_], marker=m, color=col,
                ms=4.5, label=lab)
ax.axhline(1, color="0.8", lw=0.6)
ax.set_xlabel("Parameter / reference value")
ax.set_ylabel("Window / reference")
ax.set_xticks([0.5, 1, 2])
ax.set_xticklabels(["0.5", "1", "2"])
ax.xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
ax.legend(loc="upper left", ncol=2)
label(ax, "(c)")

ax = axs[1, 1]
s9 = R["S9_back_gate"]
xb = [r["x_BG"] for r in s9]
ax.plot(xb, [r["mw_mean"] for r in s9], "o-", color=OI["blue"], ms=4.5)
ax.set_xlabel(r"Back-gate drive $x_{\rm BG}$ (V)")
ax.set_ylabel("Memory window (V)", color=OI["blue"])
ax.tick_params(axis="y", colors=OI["blue"])
ax2 = ax.twinx()
ax2.semilogy(xb, [r["onoff_x0"] for r in s9], "s--", color=OI["orange"],
             ms=4)
ax2.set_ylabel(r"On/off at $V_{\rm G}=0$", color=OI["orange"])
ax2.tick_params(axis="y", colors=OI["orange"])
ax2.spines["right"].set_visible(True)
ax.set_ylim(1.35, 1.50)
label(ax, "(d)")
fig.tight_layout()
save(fig, "figS8_extensions")
