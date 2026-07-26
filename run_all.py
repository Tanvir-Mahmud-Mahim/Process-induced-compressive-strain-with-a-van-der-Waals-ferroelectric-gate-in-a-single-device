"""
run_all.py
Master driver: runs every study and produces all main-text and
supplementary figures plus results.json with the headline numbers.

Usage:  python3 run_all.py
Output: ../figures/*.pdf, results.json
"""
import json
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, FancyArrow

import params as P
import mobility as M
import ferro
import fefet
import circuit as C

FIGDIR = os.path.join(os.path.dirname(__file__), "..", "figures")
os.makedirs(FIGDIR, exist_ok=True)
R = {}  # results dictionary

# Okabe-Ito colorblind-safe palette
OI = {"blue": "#0072B2", "orange": "#E69F00", "green": "#009E73",
      "red": "#D55E00", "purple": "#CC79A7", "sky": "#56B4E9",
      "yellow": "#F0E442", "black": "#000000"}

plt.rcParams.update({
    "font.size": 9, "axes.labelsize": 9.5, "axes.titlesize": 9.5,
    "legend.fontsize": 8, "xtick.labelsize": 8.5, "ytick.labelsize": 8.5,
    "lines.linewidth": 1.4, "figure.dpi": 200,
    "axes.spines.top": False, "axes.spines.right": False,
    "legend.framealpha": 0.9, "legend.edgecolor": "0.85",
    "legend.fancybox": False,
    "font.family": "DejaVu Sans"})


def save(fig, name):
    fig.savefig(os.path.join(FIGDIR, name + ".pdf"), bbox_inches="tight")
    fig.savefig(os.path.join(FIGDIR, name + ".png"), bbox_inches="tight")
    plt.close(fig)
    print(f"  saved {name}")


def panel_label(ax, s, dx=-0.155, dy=1.13):
    ax.text(dx, dy, s, transform=ax.transAxes, fontweight="bold",
            fontsize=10, va="top")


# ======================================================================
print("[1/6] Strain-dependent transport ...")
eps_grid = np.linspace(-1.0, 1.0, 41)
mu_grid = np.array([M.hole_mobility(e) for e in eps_grid]) * 1e4
mu0 = M.hole_mobility(0.0) * 1e4
R["mu0_cm2Vs"] = float(mu0)
R["mu_ratio_m1pct"] = float(M.hole_mobility(-1.0) * 1e4 / mu0)
R["mu_ratio_m05pct"] = float(M.hole_mobility(-0.5) * 1e4 / mu0)
R["mu_ratio_m025pct"] = float(M.hole_mobility(-0.25) * 1e4 / mu0)

# valley populations
pops = []
for e in eps_grid:
    _, _, _, pK, pG, _ = M.hole_mobility(e, return_parts=True)
    pops.append(pG / (pK + pG))
pops = np.array(pops)

# published full-band targets (Afrid 2026, extrinsic, WSe2)
tgt_eps = np.array([-1.0, -0.75, -0.5, -0.25, 0.0, 0.25, 0.5, 1.0])
tgt_mu = np.array([2.37, 2.05, 1.75, 1.40, 1.00, 0.72, 0.55, 0.33])

fig, axs = plt.subplots(2, 2, figsize=(7.0, 5.2))
ax = axs[0, 0]
dE = P.dE_GK0 - P.dE_GK_gauge * eps_grid
ax.plot(eps_grid, -dE * 1e3, color=OI["blue"])
ax.axvspan(-1.05, 0, color=OI["sky"], alpha=0.12)
ax.set_xlabel("Biaxial strain (%)")
ax.set_ylabel(r"$\Delta E_{\Gamma-K}$ (meV)")
ax.axhline(0, color="gray", lw=0.6, ls=":")
ax.annotate("compressive", (-0.95, -80), fontsize=7.5, color=OI["blue"])
panel_label(ax, "(a)")

ax = axs[0, 1]
for e, col in zip([0.5, 0.0, -0.5, -1.0],
                  [OI["red"], OI["black"], OI["sky"], OI["blue"]]):
    Ek, r_tot, r_intra, r_iv = M.scattering_spectrum(e)
    ax.semilogy(Ek * 1e3, r_tot, color=col,
                label=rf"$\epsilon$={e:+.1f}%")
ax.set_xlabel("Hole energy (meV)")
ax.set_ylabel(r"K-valley scattering rate (s$^{-1}$)")
ax.legend(frameon=False, ncol=2, loc="lower center",
          bbox_to_anchor=(0.5, 1.0), columnspacing=1.2, handlelength=1.4)
panel_label(ax, "(b)")

ax = axs[1, 0]
ax.plot(eps_grid, mu_grid / mu0, color=OI["green"], label="this work (two-valley)")
ax.plot(tgt_eps, tgt_mu, "o", ms=4.5, color=OI["black"], mfc="none",
        label="full-band (Afrid 2026)")
ax.errorbar([-0.22], [1.70], yerr=[0.21], fmt="s", ms=4.5,
            color=OI["red"], label="FET expt. (Zhao 2026)", capsize=2.5)
ax.axhline(1, color="gray", lw=0.6, ls=":")
ax.set_xlabel("Biaxial strain (%)")
ax.set_ylabel(r"$\mu_{\rm h}/\mu_{\rm h0}$")
ax.set_ylim(0, 2.75)
ax.legend(frameon=True, loc="lower left", fontsize=7.2,
          borderpad=0.5, handlelength=1.6)
panel_label(ax, "(c)")

ax = axs[1, 1]
ax.plot(eps_grid, mu_grid, color=OI["purple"])
ax2 = ax.twinx()
ax2.plot(eps_grid, 100 * pops, color=OI["orange"], ls="--")
ax2.set_ylabel(r"$\Gamma$-valley occupancy (%)", color=OI["orange"])
ax2.tick_params(axis="y", colors=OI["orange"])
ax2.spines["right"].set_visible(True)
ax.set_xlabel("Biaxial strain (%)")
ax.set_ylabel(r"$\mu_{\rm h}$ (cm$^2$/Vs)", color=OI["purple"])
ax.tick_params(axis="y", colors=OI["purple"])
panel_label(ax, "(d)")
fig.tight_layout()
save(fig, "fig2_transport")

# ======================================================================
print("[2/6] Ferroelectric response ...")
E_up, P_up, E_dn, P_dn = ferro.pe_loop(E_max=1.2e8)
fields = np.array([1.2, 1.5, 2.0, 3.0, 5.0]) * P.Ec_CIPS
tsw = np.array([ferro.switching_time(f) for f in fields])
R["t_switch_us_at_1p5Ec"] = float(ferro.switching_time(1.5 * P.Ec_CIPS) * 1e6)

fig, axs = plt.subplots(1, 3, figsize=(7.0, 2.3))
ax = axs[0]
ax.plot(E_up * 1e-8, P_up * 1e2, color=OI["blue"])
ax.plot(E_dn * 1e-8, P_dn * 1e2, color=OI["blue"])
ax.axhline(P.Pr_CIPS * 1e2, color=OI["red"], lw=0.7, ls="--")
ax.axhline(-P.Pr_CIPS * 1e2, color=OI["red"], lw=0.7, ls="--")
ax.text(-1.15, P.Pr_CIPS * 1e2 + 0.9, r"$P_{\rm r}$ = 5.5 $\mu$C/cm$^2$",
        fontsize=7.5, color=OI["red"])
ax.set_ylim(-9.5, 9.5)
ax.set_xlabel("E (MV/cm)")
ax.set_ylabel(r"P ($\mu$C/cm$^2$)")
panel_label(ax, "(a)")

ax = axs[1]
for f, col in zip([1.5, 2.0, 3.0], [OI["blue"], OI["green"], OI["orange"]]):
    t, Pt = ferro.lk_switch(f * P.Ec_CIPS, t_end=1e-3)
    ax.semilogx(t * 1e6, Pt / P.Pr_CIPS, color=col,
                label=rf"$E={f:.1f}E_{{\rm c}}$")
ax.axhline(0.9, color="gray", lw=0.6, ls=":")
ax.set_xlabel(r"Time ($\mu$s)")
ax.set_ylabel(r"$P/P_{\rm r}$")
ax.set_xlim(1e-2, 1e3)
ax.legend(frameon=True, loc="upper left", borderpad=0.5)
panel_label(ax, "(b)")

ax = axs[2]
ax.loglog(fields / P.Ec_CIPS, tsw * 1e6, "o-", color=OI["purple"], ms=4)
ax.set_xticks([1.2, 1.5, 2, 3, 5])
ax.set_xticklabels(["1.2", "1.5", "2", "3", "5"])
ax.minorticks_off()
ax.axhline(60, color="gray", lw=0.6, ls=":")
ax.text(2.45, 120, "60 $\\mu$s programming\n(Lee 2026)", fontsize=7,
        ha="left")
ax.set_xlabel(r"$E/E_{\rm c}$")
ax.set_ylabel(r"$t_{\rm sw}$ ($\mu$s)")
panel_label(ax, "(c)")
fig.tight_layout()
save(fig, "fig3_ferro")

# ======================================================================
print("[3/6] FeFET transfer characteristics ...")
sweeps = {}
for e in [0.0, -0.5]:
    d = fefet.FeFET(eps_pct=e)
    sweeps[e] = d.sweep(x_max=6.0, n=401)
mw0, vtf0, vtb0 = fefet.memory_window(*sweeps[0.0][:4])
mw5, vtf5, vtb5 = fefet.memory_window(*sweeps[-0.5][:4])
R["MW_V_eps0"] = float(mw0)
R["MW_V_eps-0.5"] = float(mw5)

# retained (nonvolatile) read currents at hold x = 0
def retained_reads(e):
    d = fefet.FeFET(eps_pct=e, n_dom=1600)
    d.fe.reset(-1)
    _, p_off = d.program(-6.0, 0.0)   # erase then hold
    I_off = d.drain_current(p_off)
    _, p_on = d.program(+6.0, 0.0)    # program then hold
    I_on = d.drain_current(p_on)
    return I_on, I_off, p_on, p_off

eps_dev = np.linspace(-1.0, 0.25, 26)
Ion_arr, Ioff_arr = [], []
for e in eps_dev:
    Ion, Ioff, p_on, p_off = retained_reads(e)
    Ion_arr.append(Ion)
    Ioff_arr.append(Ioff)
Ion_arr = np.array(Ion_arr)
Ioff_arr = np.array(Ioff_arr)
i0 = np.argmin(np.abs(eps_dev - 0.0))
R["Ion_uA_eps0"] = float(Ion_arr[i0] * 1e6)
R["Ion_uA_eps-0.5"] = float(np.interp(-0.5, eps_dev, Ion_arr) * 1e6)
R["Ion_uA_eps-1.0"] = float(Ion_arr[0] * 1e6)
R["NV_onoff_eps0"] = float(Ion_arr[i0] / Ioff_arr[i0])

# memory window vs CIPS thickness
tFE_grid = np.array([15, 20, 30, 45, 60, 87]) * 1e-9
mw_t, vprog_t = [], []
for tf in tFE_grid:
    d = fefet.FeFET(t_FE=tf)
    s = d.sweep(x_max=max(4.0, 2.2e8 * tf), n=301)
    mw, _, _ = fefet.memory_window(*s[:4])
    mw_t.append(mw)
    vprog_t.append(max(4.0, 2.2e8 * tf))
mw_t = np.array(mw_t)
R["MW_V_87nm"] = float(mw_t[-1])

fig, axs = plt.subplots(2, 2, figsize=(7.0, 5.2))
ax = axs[0, 0]
for e, col in zip([0.0, -0.5], [OI["black"], OI["blue"]]):
    xs_f, I_f, xs_b, I_b, _, _ = sweeps[e]
    # present in p-FET gate-voltage convention: VG = -x
    ax.semilogy(-xs_f, I_f * 1e6, color=col, label=rf"$\epsilon$ = {e:+.1f}%")
    ax.semilogy(-xs_b, I_b * 1e6, color=col)
ax.annotate("", xy=(-vtf0, 1e-2), xytext=(-vtb0, 1e-2),
            arrowprops=dict(arrowstyle="<->", color="gray", lw=0.8))
ax.text(-0.5 * (vtf0 + vtb0), 2.2e-2, f"MW = {mw0:.2f} V", ha="center", fontsize=7)
ax.set_xlabel(r"$V_{\rm G}$ (V)")
ax.set_ylabel(r"$|I_{\rm D}|$ ($\mu$A)")
ax.legend(frameon=True, loc="lower left", borderpad=0.5)
panel_label(ax, "(a)")

ax = axs[0, 1]
ax.plot(tFE_grid * 1e9, mw_t, "o-", color=OI["green"], ms=4)
ax.axvline(86.5, color="gray", lw=0.7, ls=":")
ax.text(64, 0.4, "expt. stack\n(Lee 2026)", fontsize=6.5)
ax.set_xlabel(r"CIPS thickness $t_{\rm FE}$ (nm)")
ax.set_ylabel("Memory window (V)")
panel_label(ax, "(b)")

ax = axs[1, 0]
ax.plot(eps_dev, Ion_arr * 1e6, "o-", color=OI["blue"], ms=3.5,
        label=r"$I_{\rm on}$ (retained)")
ax.set_xlabel("Biaxial strain (%)")
ax.set_ylabel(r"Retained $I_{\rm on}$ ($\mu$A)")
ax.axvline(0, color="gray", lw=0.6, ls=":")
ax2 = ax.twinx()
ax2.semilogy(eps_dev, Ion_arr / Ioff_arr, "s--", ms=3,
             color=OI["orange"])
ax2.set_ylabel(r"Nonvolatile $I_{\rm on}/I_{\rm off}$", color=OI["orange"])
ax2.tick_params(axis="y", colors=OI["orange"])
ax2.spines["right"].set_visible(True)
panel_label(ax, "(c)")

ax = axs[1, 1]
mwlist = []
for e in [0.0, -0.25, -0.5, -0.75, -1.0]:
    d = fefet.FeFET(eps_pct=e)
    s = d.sweep(x_max=6.0, n=301)
    mw, _, _ = fefet.memory_window(*s[:4])
    mwlist.append(mw)
epsl = [0.0, -0.25, -0.5, -0.75, -1.0]
ax.plot(epsl, mwlist, "o-", color=OI["purple"], ms=4, label="memory window")
ax.set_ylim(0, 2.0)
ax.set_xlabel("Biaxial strain (%)")
ax.set_ylabel("Memory window (V)")
ax.axhline(np.mean(mwlist), color="gray", lw=0.6, ls=":")
ratio = [float(np.interp(e, eps_dev, Ion_arr) / Ion_arr[i0]) for e in epsl]
ax2 = ax.twinx()
ax2.plot(epsl, ratio, "s--", ms=3.5, color=OI["red"])
ax2.set_ylabel(r"$I_{\rm on}(\epsilon)/I_{\rm on}(0)$", color=OI["red"])
ax2.tick_params(axis="y", colors=OI["red"])
ax2.spines["right"].set_visible(True)
panel_label(ax, "(d)")
R["MW_flatness_pct"] = float(100 * (max(mwlist) - min(mwlist)) / np.mean(mwlist))
R["Ion_gain_m1pct"] = float(ratio[-1])
fig.tight_layout()
save(fig, "fig4_fefet")

# ======================================================================
print("[4/6] Nonvolatile logic circuits ...")
VTs = (float(vtf0), float(vtb0))  # (erased high-VT, programmed low-VT)
R["VT_states"] = VTs

inv1 = C.NVInverter(eps_pct=0.0, VT_p_states=VTs, state=1)
inv0 = C.NVInverter(eps_pct=0.0, VT_p_states=VTs, state=0)
Vin1, Vo1 = inv1.vtc()
Vin0, Vo0 = inv0.vtc()
Vr, VoR = C.resistor_inverter_vtc(VT_state=0, VT_p_states=VTs)

# butterfly for cross-coupled programmed pair
snm = C.butterfly_snm(C.NVInverter(VT_p_states=VTs, state=1),
                      C.NVInverter(VT_p_states=VTs, state=1))
R["SNM_V"] = float(snm)

# power-up restore transient of the cross-coupled latch
def latch_restore(state_a=1, state_b=0, t_end=2e-9, n=6000, eps_pct=0.0):
    invA = C.NVInverter(eps_pct=eps_pct, VT_p_states=VTs, state=state_a)
    invB = C.NVInverter(eps_pct=eps_pct, VT_p_states=VTs, state=state_b)
    t = np.linspace(0, t_end, n)
    dt = t[1] - t[0]
    VQ = np.zeros_like(t)
    VQb = np.zeros_like(t)
    vdd_t = np.clip(t / (0.4 * t_end), 0, 1) * P.VDD  # supply ramp
    for i in range(1, n):
        vdd = vdd_t[i]
        invA.VDD = vdd
        invB.VDD = vdd
        IpA, InA = invA.currents(VQb[i - 1], VQ[i - 1])
        IpB, InB = invB.currents(VQ[i - 1], VQb[i - 1])
        VQ[i] = np.clip(VQ[i - 1] + dt * (IpA - InA) / P.C_load, 0, vdd)
        VQb[i] = np.clip(VQb[i - 1] + dt * (IpB - InB) / P.C_load, 0, vdd)
    return t, VQ, VQb, vdd_t

t_r, VQ_r, VQb_r, vdd_r = latch_restore()
R["restore_Q_final"] = float(VQ_r[-1])
R["restore_Qb_final"] = float(VQb_r[-1])

# delay / energy vs strain
eps_c = np.linspace(-1.0, 0.0, 11)
tpl, edp = [], []
for e in eps_c:
    inv = C.NVInverter(eps_pct=e, VT_p_states=VTs, state=1)
    tp, Ed = inv.delay_energy()
    tpl.append(tp)
    edp.append(tp * Ed)
tpl = np.array(tpl)
edp = np.array(edp)
R["tpLH_ps_eps0"] = float(tpl[-1] * 1e12)
R["tpLH_ps_m1pct"] = float(tpl[0] * 1e12)
R["EDP_gain_m1pct"] = float(edp[-1] / edp[0])

Pn_off, Pp_off = inv1.static_power()
P_stat_cmos = max(Pn_off, Pp_off)
P_stat_res = P.VDD ** 2 / 10e6
R["P_static_cmos_pW"] = float(P_stat_cmos * 1e12)
R["P_static_res_uW"] = float(P_stat_res * 1e6)

fig, axs = plt.subplots(2, 2, figsize=(7.0, 5.2))
ax = axs[0, 0]
ax.plot(Vin1, Vo1, color=OI["blue"], label="programmed")
ax.plot(Vin0, Vo0, color=OI["red"], label="erased")
ax.plot(Vr, VoR, color="gray", ls="--", lw=1.0,
        label=r"10 M$\Omega$ resistor")
ax.set_xlabel(r"$V_{\rm in}$ (V)")
ax.set_ylabel(r"$V_{\rm out}$ (V)")
ax.legend(frameon=True, fontsize=7.5, loc="lower left", borderpad=0.5)
panel_label(ax, "(a)")

ax = axs[0, 1]
ax.plot(Vin1, Vo1, color=OI["blue"])
ax.plot(Vo1, Vin1, color=OI["orange"])
s = snm / np.sqrt(2)
ax.set_xlabel(r"$V_{Q}$ (V)")
ax.set_ylabel(r"$V_{\bar Q}$ (V)")
ax.text(2.92, 2.78, f"SNM = {snm*1e3:.0f} mV", fontsize=8, ha="right")
panel_label(ax, "(b)")

ax = axs[1, 0]
ax.plot(t_r * 1e9, vdd_r, color="gray", ls=":", label=r"$V_{\rm DD}$ ramp")
ax.plot(t_r * 1e9, VQ_r, color=OI["blue"], label=r"$Q$ (stored 1)")
ax.plot(t_r * 1e9, VQb_r, color=OI["red"], label=r"$\bar{Q}$ (stored 0)")
ax.set_xlabel("Time (ns)")
ax.set_ylabel("Node voltage (V)")
ax.legend(frameon=True, loc="center right", fontsize=7.5, borderpad=0.5)
panel_label(ax, "(c)")

ax = axs[1, 1]
ax.plot(eps_c, tpl * 1e12, "o-", ms=4, color=OI["green"])
ax.set_xlabel("Biaxial strain (%)")
ax.set_ylabel(r"$t_{\rm pLH}$ (ps)", color=OI["green"])
ax.tick_params(axis="y", colors=OI["green"])
ax2 = ax.twinx()
gain_read = [float(np.interp(e, eps_dev, Ion_arr) / Ion_arr[i0]) for e in eps_c]
ax2.plot(eps_c, gain_read, "s--", ms=3.5, color=OI["purple"])
ax2.set_ylabel(r"Retained read gain $I_{\rm on}(\epsilon)/I_{\rm on}(0)$",
               color=OI["purple"])
ax2.tick_params(axis="y", colors=OI["purple"])
ax2.spines["right"].set_visible(True)
panel_label(ax, "(d)")
fig.tight_layout()
save(fig, "fig5_circuit")

# ======================================================================
print("[5/6] Concept figure ...")
fig, axs = plt.subplots(1, 3, figsize=(7.0, 2.5),
                        gridspec_kw={"width_ratios": [1.15, 1, 1]})
ax = axs[0]
ax.set_xlim(0, 10)
ax.set_ylim(0, 10)
ax.axis("off")
layers = [
    (1.2, 1.0, "p++ Si / SiO$_2$", "#c8c8c8"),
    (2.2, 0.7, "Al$_2$O$_3$", "#e0d6f5"),
    (2.9, 0.45, "WSe$_2$ (1L, compressed)", OI["sky"]),
    (3.35, 0.9, "h-BN", "#d3f0e0"),
    (4.25, 0.55, "floating gate (M)", "#f5d9a8"),
    (4.8, 1.6, "CIPS (FE)", "#f2b8b8"),
    (6.4, 0.6, "top gate (M)", "#f5d9a8"),
]
for y0, h, lab, colr in layers:
    ax.add_patch(Rectangle((1.6, y0), 6.8, h, facecolor=colr,
                           edgecolor="k", lw=0.5))
    ax.text(5.0, y0 + h / 2, lab, ha="center", va="center", fontsize=6.6)
# stressor arrows
ax.annotate("", xy=(1.8, 3.12), xytext=(0.4, 3.12),
            arrowprops=dict(arrowstyle="-|>", color=OI["blue"], lw=1.6))
ax.annotate("", xy=(8.2, 3.12), xytext=(9.6, 3.12),
            arrowprops=dict(arrowstyle="-|>", color=OI["blue"], lw=1.6))
ax.text(0.25, 2.35, r"$\epsilon < 0$", color=OI["blue"], fontsize=8)
ax.add_patch(Rectangle((0.9, 2.7), 0.7, 0.85, facecolor="#ffe28a",
                       edgecolor="k", lw=0.5))
ax.add_patch(Rectangle((8.4, 2.7), 0.7, 0.85, facecolor="#ffe28a",
                       edgecolor="k", lw=0.5))
ax.text(1.25, 2.05, "S", ha="center", va="top", fontsize=7.5)
ax.text(8.75, 2.05, "D", ha="center", va="top", fontsize=7.5)
ax.set_title("Strained p-type MFMIS FeFET", fontsize=8)
panel_label(ax, "(a)", dx=-0.02)

ax = axs[1]
kk = np.linspace(-1, 1, 200)
E_K = -2.2 * (kk - 0.55) ** 2
E_G = -0.157 - 1.0 * (kk + 0.45) ** 2
E_Gc = -0.157 - 0.341 * 0.7 - 1.0 * (kk + 0.45) ** 2
ax.plot(kk, E_K, color=OI["black"], label="K valley")
ax.plot(kk, E_G, color="gray", ls="--", label=r"$\Gamma$, $\epsilon=0$")
ax.plot(kk, E_Gc, color=OI["blue"], label=r"$\Gamma$, compressed")
ax.annotate("", xy=(-0.45, -0.52), xytext=(-0.45, -0.22),
            arrowprops=dict(arrowstyle="-|>", color=OI["blue"], lw=1.4))
ax.text(-0.40, -0.76, "IV scattering\nsuppressed", fontsize=6.8,
        color=OI["blue"], ha="center")
ax.set_ylim(-1.05, 0.42)
ax.set_xticks([])
ax.set_yticks([])
ax.set_xlabel(r"$\Gamma\;\longleftarrow\;k\;\longrightarrow\;$K")
ax.set_ylabel(r"$E_{\rm VB}$")
ax.legend(frameon=False, fontsize=6.8, loc="upper left", ncol=1,
          bbox_to_anchor=(0.0, 1.02), handlelength=1.4, labelspacing=0.25)
ax.spines["left"].set_visible(True)
panel_label(ax, "(b)", dx=-0.06)

ax = axs[2]
ax.axis("off")
steps = ["Two-valley strained\ntransport (calibrated\nto full-band data)",
         "Multidomain CIPS\nLandau-Khalatnikov +\nPreisach model",
         "Self-consistent\nMFMIS electrostatics",
         "Nonvolatile logic:\nVTC, SNM, restore,\ndelay and energy"]
cols = [OI["sky"], "#f2b8b8", "#d3f0e0", "#f5d9a8"]
for i, (s_, colr) in enumerate(zip(steps, cols)):
    y = 8.6 - 2.35 * i
    ax.add_patch(Rectangle((0.6, y - 1.55), 8.6, 1.9,
                           facecolor=colr, edgecolor="k", lw=0.5))
    ax.text(4.9, y - 0.6, s_, ha="center", va="center", fontsize=6.2)
    if i < 3:
        ax.annotate("", xy=(4.9, y - 1.95), xytext=(4.9, y - 1.6),
                    arrowprops=dict(arrowstyle="-|>", color="k", lw=1.0))
ax.set_xlim(0, 10)
ax.set_ylim(0, 10)
ax.set_title("Multiscale workflow", fontsize=8)
panel_label(ax, "(c)", dx=-0.02)
fig.tight_layout()
save(fig, "fig1_concept")

# ======================================================================
print("[6/6] Supplementary figures ...")
# S1: valley populations & Fermi level vs strain
fig, axs = plt.subplots(1, 2, figsize=(6.6, 2.4))
ax = axs[0]
muK_l, muG_l = [], []
for e in eps_grid:
    _, muK, muG, pK, pG, Ef = M.hole_mobility(e, return_parts=True)
    muK_l.append(muK * 1e4)
    muG_l.append(muG * 1e4)
ax.plot(eps_grid, muK_l, color=OI["blue"], label=r"$\mu_{\rm K}$")
ax.plot(eps_grid, muG_l, color=OI["orange"], label=r"$\mu_{\Gamma}$")
ax.plot(eps_grid, mu_grid, color=OI["black"], ls="--", label=r"$\mu_{\rm eff}$")
ax.set_xlabel("Biaxial strain (%)")
ax.set_ylabel(r"$\mu$ (cm$^2$/Vs)")
ax.legend(frameon=False)
panel_label(ax, "(a)")
ax = axs[1]
for pdens, col in zip([1e16, 1e17, 5e17], [OI["sky"], OI["blue"], OI["purple"]]):
    mus = [M.hole_mobility(e, p_sheet=pdens) * 1e4 for e in
           np.linspace(-1, 0.5, 16)]
    mus = np.array(mus)
    ax.plot(np.linspace(-1, 0.5, 16), mus / mus[np.argmin(np.abs(np.linspace(-1, 0.5, 16)))],
            "o-", ms=3, color=col, label=rf"$p$ = {pdens/1e16:.0f}e12 cm$^{{-2}}$")
ax.axhline(1, color="gray", lw=0.6, ls=":")
ax.set_xlabel("Biaxial strain (%)")
ax.set_ylabel(r"$\mu/\mu_0$")
ax.legend(frameon=False, fontsize=6.5)
panel_label(ax, "(b)")
fig.tight_layout()
save(fig, "figS1_valley")

# S2: sensitivity of mobility calibration
fig, axs = plt.subplots(1, 2, figsize=(6.6, 2.4))
ax = axs[0]
for div, col in zip([1.2e11, 1.5e11, 1.8e11], [OI["sky"], OI["blue"], OI["purple"]]):
    mus = np.array([M.hole_mobility(e, D_iv=div * P.eV) for e in
                    np.linspace(-1, 0.5, 16)])
    mu00 = M.hole_mobility(0.0, D_iv=div * P.eV)
    ax.plot(np.linspace(-1, 0.5, 16), mus / mu00, "o-", ms=3, color=col,
            label=rf"$D_{{\rm iv}}$ = {div/1e11:.1f}e11 eV/m")
ax.plot(tgt_eps, tgt_mu, "k s", ms=4, mfc="none", label="full-band")
ax.set_xlim(-1.05, 0.55)
ax.set_xlabel("Biaxial strain (%)")
ax.set_ylabel(r"$\mu/\mu_0$")
ax.legend(frameon=False, fontsize=6.5)
panel_label(ax, "(a)")
ax = axs[1]
sig_save = M.SIGMA_IV
for sig, col in zip([70e-3, 90e-3, 110e-3], [OI["sky"], OI["blue"], OI["purple"]]):
    M.SIGMA_IV = sig * P.eV
    mus = np.array([M.hole_mobility(e) for e in np.linspace(-1, 0.5, 16)])
    mu00 = M.hole_mobility(0.0)
    ax.plot(np.linspace(-1, 0.5, 16), mus / mu00, "o-", ms=3, color=col,
            label=rf"$\sigma_{{\rm iv}}$ = {sig*1e3:.0f} meV")
M.SIGMA_IV = sig_save
ax.plot(tgt_eps, tgt_mu, "k s", ms=4, mfc="none", label="full-band")
ax.set_xlim(-1.05, 0.55)
ax.set_xlabel("Biaxial strain (%)")
ax.set_ylabel(r"$\mu/\mu_0$")
ax.legend(frameon=False, fontsize=6.5)
panel_label(ax, "(b)")
fig.tight_layout()
save(fig, "figS2_calibration")

# S3: coercive-field spread effect on the P-E loop and FeFET window
fig, axs = plt.subplots(1, 2, figsize=(6.6, 2.4))
ax = axs[0]
for s_, col in zip([0.1, 0.22, 0.4], [OI["sky"], OI["blue"], OI["purple"]]):
    Eu, Pu, Ed_, Pd = ferro.pe_loop(E_max=1.2e8, sigma_rel=s_)
    ax.plot(Eu * 1e-8, Pu * 1e2, color=col, label=rf"$\sigma_{{Ec}}$ = {s_:.2f}")
    ax.plot(Ed_ * 1e-8, Pd * 1e2, color=col)
ax.set_xlabel("E (MV/cm)")
ax.set_ylabel(r"P ($\mu$C/cm$^2$)")
ax.legend(frameon=False, fontsize=6.5)
panel_label(ax, "(a)")
ax = axs[1]
xs_f, I_f, xs_b, I_b, p_f, p_b = sweeps[0.0]
ax.plot(-xs_f, p_f / 1e16, color=OI["blue"])
ax.plot(-xs_b, p_b / 1e16, color=OI["blue"])
ax.set_xlabel(r"$V_{\rm G}$ (V)")
ax.set_ylabel(r"$p$ (10$^{12}$ cm$^{-2}$)")
panel_label(ax, "(b)")
fig.tight_layout()
save(fig, "figS3_ferro_extra")

# S4: depolarization / retention proxy and thickness design
fig, axs = plt.subplots(1, 2, figsize=(6.6, 2.4))
ax = axs[0]
Edep_list = []
for tf in tFE_grid:
    d = fefet.FeFET(t_FE=tf)
    d.fe.reset(-1)
    d.program(+max(4.0, 2.2e8 * tf), 0.0)
    E_hold, _ = d.solve_bias(0.0)
    Edep_list.append(abs(E_hold) / P.Ec_CIPS)
ax.plot(tFE_grid * 1e9, Edep_list, "o-", ms=4, color=OI["red"])
ax.axhline(1.0, color="gray", ls=":", lw=0.8)
ax.text(20, 1.03, r"$|E_{\rm dep}| = E_{\rm c}$", fontsize=7)
ax.set_xlabel(r"$t_{\rm FE}$ (nm)")
ax.set_ylabel(r"$|E_{\rm dep}|/E_{\rm c}$ (retained state)")
panel_label(ax, "(a)")
R["Edep_over_Ec_30nm"] = float(Edep_list[2])
ax = axs[1]
ax.plot(tFE_grid * 1e9, np.array(vprog_t), "s-", ms=4, color=OI["green"])
ax.set_xlabel(r"$t_{\rm FE}$ (nm)")
ax.set_ylabel("Required program drive (V)")
panel_label(ax, "(b)")
fig.tight_layout()
save(fig, "figS4_retention")

# S5: strained latch restore + resistor comparison transient
fig, axs = plt.subplots(1, 2, figsize=(6.6, 2.4))
ax = axs[0]
for e, col in zip([0.0, -1.0], [OI["black"], OI["blue"]]):
    t_r2, VQ_r2, VQb_r2, vdd2 = latch_restore(eps_pct=e, t_end=2e-9)
    ax.plot(t_r2 * 1e9, VQ_r2, color=col, label=rf"$Q$, $\epsilon$={e:+.0f}%")
    ax.plot(t_r2 * 1e9, VQb_r2, color=col, ls="--")
ax.plot(t_r * 1e9, vdd_r, color="gray", ls=":", lw=0.8)
ax.set_xlabel("Time (ns)")
ax.set_ylabel("Node voltage (V)")
ax.legend(frameon=False, fontsize=7)
panel_label(ax, "(a)")
ax = axs[1]
labels = ["resistor load\n(Lee 2026 style)", "complementary\n(this work)"]
vals = [P_stat_res * 1e6, P_stat_cmos * 1e6]
bars = ax.bar(labels, vals, color=[OI["orange"], OI["green"]], width=0.55)
ax.set_yscale("log")
ax.set_ylabel(r"Worst-case static power ($\mu$W)")
for b, v in zip(bars, vals):
    ax.text(b.get_x() + b.get_width() / 2, v * 1.6,
            f"{v:.2g}", ha="center", fontsize=7)
panel_label(ax, "(b)")
fig.tight_layout()
save(fig, "figS5_circuit_extra")


# ======================================================================
print("[7/7] Graphical abstract ...")
fig = plt.figure(figsize=(7.2, 2.15))
gs = fig.add_gridspec(1, 3, width_ratios=[1.05, 1.15, 1.0],
                      left=0.01, right=0.99, top=0.88, bottom=0.05,
                      wspace=0.28)

# --- left: strained FeFET stack ---
axA = fig.add_subplot(gs[0])
axA.set_xlim(0, 10); axA.set_ylim(0, 10); axA.axis("off")
ga_layers = [
    (1.0, 0.9, "substrate", "#d9d9d9"),
    (1.9, 0.75, "WSe$_2$ channel", OI["sky"]),
    (2.65, 1.1, "h-BN", "#d3f0e0"),
    (3.75, 0.7, "floating gate", "#f5d9a8"),
    (4.45, 1.9, "CIPS ferroelectric", "#f2b8b8"),
    (6.35, 0.75, "top gate", "#f5d9a8"),
]
for y0, h, lab, colr in ga_layers:
    axA.add_patch(Rectangle((1.7, y0), 6.6, h, facecolor=colr,
                            edgecolor="k", lw=0.5))
    axA.text(5.0, y0 + h / 2, lab, ha="center", va="center", fontsize=7)
axA.annotate("", xy=(1.65, 2.28), xytext=(0.15, 2.28),
             arrowprops=dict(arrowstyle="-|>", color=OI["blue"], lw=2.0))
axA.annotate("", xy=(8.35, 2.28), xytext=(9.85, 2.28),
             arrowprops=dict(arrowstyle="-|>", color=OI["blue"], lw=2.0))
axA.text(5.0, 0.25, r"biaxial compression $\epsilon<0$", color=OI["blue"],
         fontsize=7.5, ha="center")
axA.text(5.0, 8.6, "Strained p-type FeFET", ha="center", fontsize=8.5,
         fontweight="bold")
axA.text(5.0, 7.55, "polarization stores the bit", ha="center", fontsize=7,
         color="0.35")

# --- middle: two orthogonal knobs ---
axB = fig.add_subplot(gs[1])
pos = axB.get_position()
axB.axis("off")
axB.set_xlim(0, 10); axB.set_ylim(0, 10)
axB.text(5.0, 9.6, "Two independent knobs", ha="center", fontsize=8.5,
         fontweight="bold")
# small mobility inset
axm = fig.add_axes([pos.x0 + 0.005, 0.16, 0.135, 0.50])
eg = np.linspace(-1, 0, 30)
mm = np.array([M.hole_mobility(e) for e in eg])
axm.plot(eg, mm / mm[-1], color=OI["green"], lw=1.6)
axm.set_xticks([-1, 0]); axm.set_yticks([1, 2])
axm.tick_params(labelsize=6, pad=1.5)
axm.set_xlabel("strain (%)", fontsize=6.5, labelpad=1)
axm.set_title(r"$\mu_{\rm h}\times 2.3$", fontsize=7.5, pad=2,
              color=OI["green"])
# small memory-window inset
axp = fig.add_axes([pos.x0 + 0.175, 0.16, 0.135, 0.50])
axp.plot([-1, 0], [1.24, 1.24], "o-", ms=3, color=OI["purple"], lw=1.6)
axp.set_ylim(0, 2); axp.set_xticks([-1, 0]); axp.set_yticks([0, 1, 2])
axp.tick_params(labelsize=6, pad=1.5)
axp.set_xlabel("strain (%)", fontsize=6.5, labelpad=1)
axp.set_title("window unchanged", fontsize=7.5, pad=2, color=OI["purple"])

# --- right: outcome ---
axC = fig.add_subplot(gs[2])
axC.set_xlim(0, 10); axC.set_ylim(0, 10); axC.axis("off")
axC.text(5.0, 9.6, "Nonvolatile logic", ha="center", fontsize=8.5,
         fontweight="bold")
msgs = ["restores state after\npower loss (< 2 ns)",
        "10$^{8}\\times$ lower\nstatic power",
        "2.3$\\times$ faster reads\nunder strain"]
mc = [OI["blue"], OI["green"], OI["red"]]
for i, (m_, c_) in enumerate(zip(msgs, mc)):
    y = 7.3 - 2.9 * i
    axC.add_patch(Rectangle((0.4, y - 1.15), 9.2, 2.3, facecolor="white",
                            edgecolor=c_, lw=1.2))
    axC.text(5.0, y, m_, ha="center", va="center", fontsize=7.5, color=c_)

# arrows between panels, placed from actual axes positions
posA = axA.get_position()
posC = axC.get_position()
x_arr1 = 0.5 * (posA.x1 + (pos.x0 + 0.005))
x_arr2 = 0.5 * ((pos.x0 + 0.175 + 0.135) + posC.x0)
for xarr in [x_arr1, x_arr2]:
    fig.text(xarr, 0.45, r"$\Rightarrow$", fontsize=17, ha="center",
             va="center", color="0.3")
save(fig, "fig0_abstract")

# ======================================================================
with open(os.path.join(os.path.dirname(__file__), "results.json"), "w") as f:
    json.dump(R, f, indent=2)
print(json.dumps(R, indent=2))
print("done.")
