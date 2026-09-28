"""
revision_studies.py
Robustness and scope studies of the constant-current memory-window law,
added in the revision. Writes revision_results.json; the figures are made
by revision_figures.py.

Studies
  S0  the generalized solver (fefet_ext.FeFETX) reproduces fefet.FeFET
  S1  origin of the residual: sweep-grid refinement
  S2  branch-dependent mobility: imposed ratio, trap-coupled scattering,
      and density-dependent mobility
  S3  channel density of states, interlayer capacitance and read level
  S4  area ratio of the ferroelectric and channel capacitors
  S5  ferroelectric-side charge (injection, ion migration, leakage)
  S6  ambipolar channel that can screen the depolarization field
  S7  what sets the switched polarization: Pr, Ec, spread, eps_b
  S8  parameter sensitivity of the headline numbers

Usage: python3 revision_studies.py   (about 15 minutes on one core)
"""
import json
import time
import numpy as np
import params as P
import mobility as M
import fefet
from fefet_ext import FeFETX, probe, law_terms

OUT = {}
T0 = time.time()


def log(s):
    print(f"[{time.time() - T0:7.1f} s] {s}", flush=True)


def check(n=401, x_max=6.0, I_crit=None, seed=3, **kw):
    d = FeFETX(seed=seed, **kw)
    f, b, fw, bw = probe(d, x_max=x_max, n=n, I_crit=I_crit)
    if f is None or b is None:
        return None, d, fw, bw
    r = law_terms(d, f, b)
    return r, d, fw, bw


def window_stats(seeds=(1, 2, 3, 4), **kw):
    """Mean and spread of the window and of the law error over draws."""
    rows = []
    for s in seeds:
        r, d, fw, bw = check(seed=s, **kw)
        if r is not None:
            rows.append(r)
    if not rows:
        return None
    mw = np.array([r["mw_sim"] for r in rows])
    return dict(
        mw_mean=float(mw.mean()), mw_sd=float(mw.std()),
        dP_mean=float(np.mean([r["dP"] for r in rows])),
        dQ_mean=float(np.mean([r["dQ"] for r in rows])),
        max_err_law_pct=float(max(abs(r["err_law_pct"]) for r in rows)),
        max_err_corr_pct=float(max(abs(r["err_corr_pct"]) for r in rows)),
        E_f_over_Ec=float(np.mean([r["E_f"] for r in rows]) / P.Ec_CIPS),
        E_b_over_Ec=float(np.mean([r["E_b"] for r in rows]) / P.Ec_CIPS),
        p_f=float(np.mean([r["p_f"] for r in rows])),
        n=len(rows))


# ----------------------------------------------------------------------
log("S0 generalized solver against the original solver")
s0 = []
for kw in [dict(eps_pct=0.0), dict(eps_pct=-1.0), dict(eps_pct=0.9),
           dict(Dit_cm2=1e12), dict(Dit_cm2=3e12), dict(Dit_cm2=1e13),
           dict(t_FE=86.5e-9)]:
    d0 = fefet.FeFET(seed=3, **kw)
    sw = d0.sweep(x_max=6.0, n=401)
    mw0 = fefet.memory_window(*sw[:4])[0]
    r, _, _, _ = check(**kw)
    s0.append(dict(case=str(kw), mw_original=float(mw0),
                   mw_generalized=r["mw_sim"],
                   diff_V=float(r["mw_sim"] - mw0)))
OUT["S0_solver_equivalence"] = s0
OUT["S0_max_abs_diff_V"] = float(max(abs(r["diff_V"]) for r in s0))
log(f"  max |difference| = {OUT['S0_max_abs_diff_V']:.2e} V")

# ----------------------------------------------------------------------
log("S1 grid refinement of the residual")
s1 = []
cases = [("eps = -1.0%", dict(eps_pct=-1.0)),
         ("eps = 0", dict(eps_pct=0.0)),
         ("eps = +0.90%", dict(eps_pct=0.90)),
         ("t_FE = 86.5 nm", dict(t_FE=86.5e-9)),
         ("Dit = 1e12", dict(Dit_cm2=1e12)),
         ("Dit = 1e13", dict(Dit_cm2=1e13))]
for name, kw in cases:
    for n in [401, 1601, 6401]:
        r, d, _, _ = check(n=n, **kw)
        s1.append(dict(case=name, n=n, step_mV=1e3 * 12.0 / (n - 1),
                       mw_sim=r["mw_sim"], law=r["law"],
                       err_law_pct=r["err_law_pct"],
                       err_corr_pct=r["err_corr_pct"],
                       density_mismatch_pct=100.0 * (r["p_f"] / r["p_b"] - 1.0)))
        log(f"  {name:>15} n={n:5d}: err {r['err_law_pct']:+.4f} %, "
            f"p_f/p_b-1 {s1[-1]['density_mismatch_pct']:+.3f} %, "
            f"after correction {r['err_corr_pct']:+.2e} %")
OUT["S1_grid_refinement"] = s1

# ----------------------------------------------------------------------
log("S2 branch-dependent mobility")
N_FINE = 1601
s2 = []
kT = P.kT_eV
for ratio in [0.5, 0.8, 0.9, 1.0, 1.1, 1.25, 2.0]:
    r, d, _, _ = check(n=N_FINE, eps_pct=0.0, mu_bwd_ratio=ratio)
    # closed form in the nondegenerate threshold regime with R_c << R_ch:
    # p_f/p_b = mu_b/mu_f = ratio, delta = (kT/q) ln(ratio) + q p_f (1 - 1/ratio)/C_s
    cf = kT * np.log(ratio) + P.q * r["p_f"] * (1.0 - 1.0 / ratio) / d.C_series()
    s2.append(dict(mu_ratio=ratio, mw_sim=r["mw_sim"], law=r["law"],
                   err_law_pct=r["err_law_pct"],
                   corr_from_crossings_V=r["corr"],
                   corr_closed_form_V=float(cf),
                   err_after_closed_form_pct=float(
                       100.0 * (r["law"] + cf - r["mw_sim"]) / r["mw_sim"]),
                   p_f=r["p_f"], p_b=r["p_b"]))
    log(f"  mu_b/mu_f = {ratio:4.2f}: sim {r['mw_sim']:.4f} law {r['law']:.4f} "
        f"err {r['err_law_pct']:+.3f} %, closed form {1e3 * cf:+.2f} mV, "
        f"residual {s2[-1]['err_after_closed_form_pct']:+.3f} %")
OUT["S2_branch_mobility_imposed"] = s2
OUT["C_series_uFcm2"] = float(FeFETX().C_series() * 1e2)

s2b = []
for Dit in [3e11, 1e12, 3e12]:
    for mode in ["worst"]:
        kwd = dict(eps_pct=0.0, Dit_cm2=Dit, trap_mode=mode, mu_trap=True)
        r, d, fw, bw = check(n=N_FINE, **kwd)
        s2b.append(dict(Dit=Dit, trap_mode=mode, mw_sim=r["mw_sim"],
                        law=r["law"], err_law_pct=r["err_law_pct"],
                        err_corr_pct=r["err_corr_pct"],
                        density_mismatch_pct=100.0 * (r["p_f"] / r["p_b"] - 1.0)))
        log(f"  trap-coupled mobility Dit={Dit:.0e} {mode}: err "
            f"{r['err_law_pct']:+.3f} %, p_f/p_b-1 "
            f"{s2b[-1]['density_mismatch_pct']:+.2f} %")
OUT["S2_branch_mobility_trap_coupled"] = s2b

s2c = []
for e in [-1.0, 0.0, 0.5, 0.85]:
    r, d, _, _ = check(n=N_FINE, eps_pct=e, mu_of_p=True)
    r0, _, _, _ = check(n=N_FINE, eps_pct=e)
    s2c.append(dict(eps=e, mw_sim=r["mw_sim"], mw_const_mu=r0["mw_sim"],
                    err_law_pct=r["err_law_pct"],
                    mu_at_read_cm2=float(d.mobility(r["p_f"]) * 1e4),
                    mu_const_cm2=float(d.mu * 1e4)))
    log(f"  mu(p) eps={e:+.2f}: sim {r['mw_sim']:.4f} (const mu "
        f"{r0['mw_sim']:.4f}), err {r['err_law_pct']:+.3f} %")
OUT["S2_density_dependent_mobility"] = s2c

# ----------------------------------------------------------------------
log("S3 density of states, interlayer capacitance, read level")


def cq_nondeg(p):
    return P.q ** 2 * p / P.kT


s3 = []
for ds in [1e-3, 1e-2, 1e-1, 1.0, 10.0]:
    st = window_stats(eps_pct=0.0, dos_scale=ds)
    r, d, fw, bw = check(eps_pct=0.0, dos_scale=ds)
    pmax = float(np.max(fw[:, 4]))
    # quantum capacitance at threshold and at the program extreme
    h = 1e-4
    psi_top = float(fw[np.argmax(fw[:, 4]), 5])
    cq_top = P.q * (d.p_hole(psi_top + h) - d.p_hole(psi_top - h)) / (2 * h)
    from scipy.optimize import brentq as _bq
    psi_f = _bq(lambda s_: d.p_hole(s_) - st["p_f"], -5.0, 5.0, xtol=1e-12)
    cq_th = P.q * (d.p_hole(psi_f + h) - d.p_hole(psi_f - h)) / (2 * h)
    st.update(dict(dos_scale=ds, CQ_th_uFcm2=float(cq_th * 1e2),
                   CQ_prog_uFcm2=float(cq_top * 1e2),
                   Cs_uFcm2=float(d.C_series() * 1e2),
                   Cins_uFcm2=float(d.C_hBN * 1e2),
                   p_max_cm2=pmax * 1e-4))
    s3.append(st)
    log(f"  DOS x{ds:g}: MW {st['mw_mean']:.4f} +- {st['mw_sd']:.4f}, dP "
        f"{st['dP_mean'] * 1e2:.3f} uC/cm2, CQ_th {cq_th * 1e2:.3g}, CQ_prog "
        f"{cq_top * 1e2:.3g} uF/cm2, law err {st['max_err_law_pct']:.3f} %")
OUT["S3_dos_scale"] = s3

s3b = []
for th in [2e-9, 5e-9, 10e-9, 20e-9, 40e-9]:
    st = window_stats(eps_pct=0.0, t_hBN=th)
    d = FeFETX(t_hBN=th)
    st.update(dict(t_hBN_nm=th * 1e9, Cins_uFcm2=float(d.C_hBN * 1e2)))
    s3b.append(st)
    log(f"  t_hBN {th * 1e9:4.0f} nm: MW {st['mw_mean']:.4f}, dP "
        f"{st['dP_mean'] * 1e2:.3f}, law err {st['max_err_law_pct']:.3f} %")
OUT["S3_t_hBN"] = s3b

s3c = []
Wum = P.W_ch / 1e-6
for Ic_um in [1e-10, 1e-9, 1e-8, 1e-7, 1e-6, 3e-6]:
    row = dict(I_crit_A_per_um=Ic_um)
    for e in [-1.0, 0.0, 0.9]:
        st = window_stats(eps_pct=e, I_crit=Ic_um * Wum)
        if st is None:
            continue
        row[f"eps{e:+.1f}"] = st
    s3c.append(row)
    msg = ", ".join(f"{k}: {v['mw_mean']:.4f}" for k, v in row.items()
                    if k.startswith("eps"))
    log(f"  I_crit {Ic_um:.0e} A/um: {msg}")
OUT["S3_read_level"] = s3c

# ----------------------------------------------------------------------
log("S4 area ratio")
s4 = []
for ar in [0.25, 0.5, 1.0, 2.0, 4.0]:
    st = window_stats(eps_pct=0.0, A_R=ar)
    st_t = window_stats(eps_pct=0.0, A_R=ar, Dit_cm2=1e12)
    st["A_R"] = ar
    st["with_Dit1e12"] = st_t
    s4.append(st)
    log(f"  A_R {ar:4.2f}: MW {st['mw_mean']:.4f}, law err "
        f"{st['max_err_law_pct']:.3f} %; with Dit 1e12 MW {st_t['mw_mean']:.4f},"
        f" err {st_t['max_err_law_pct']:.3f} %")
OUT["S4_area_ratio"] = s4

# ----------------------------------------------------------------------
log("S5 ferroelectric-side charge")
s5 = []
denom = P.eps0 * P.eps_FE_b
for sig in [0.02e-2, 0.1e-2, 0.3e-2]:           # C/m^2 (0.02 to 0.3 uC/cm^2)
    for dfr in [0.0, 0.5]:
        r, d, _, _ = check(n=N_FINE, eps_pct=0.0, sigma_FE0=sig, d_frac=dfr)
        law_old = d.t_FE * (r["dQ"] - r["dP"]) / denom + r["dQ"] / d.C_hBN
        s5.append(dict(sigma_FE0_uCcm2=sig * 1e2, d_frac=dfr,
                       mw_sim=r["mw_sim"], law_generalized=r["law"],
                       err_generalized_pct=r["err_law_pct"],
                       law_without_FE_term=float(law_old),
                       err_without_FE_term_pct=float(
                           100.0 * (law_old - r["mw_sim"]) / r["mw_sim"]),
                       dSigma_uCcm2=r["dS"] * 1e2, dP_uCcm2=r["dP"] * 1e2))
        log(f"  sigma0 {sig * 1e2:.2f} uC/cm2, d/t {dfr}: MW {r['mw_sim']:.4f},"
            f" generalized err {r['err_law_pct']:+.3f} %, Eq.7 alone err "
            f"{s5[-1]['err_without_FE_term_pct']:+.2f} %")
OUT["S5_FE_side_charge"] = s5

# ----------------------------------------------------------------------
log("S6 ambipolar channel (depolarization screening by inversion)")
s6 = []
base = window_stats(eps_pct=0.0)
for Eg in [2.0, 1.6, 1.2, 0.8, 0.5]:
    st = window_stats(eps_pct=0.0, ambipolar=True, Eg=Eg)
    r, d, fw, bw = check(eps_pct=0.0, ambipolar=True, Eg=Eg)
    nmin = float(max(d.n_elec(np.min(fw[:, 5])), 0.0))
    st.update(dict(Eg=Eg, n_max_cm2=nmin * 1e-4))
    s6.append(st)
    log(f"  Eg {Eg:.1f} eV: MW {st['mw_mean']:.4f} (unipolar "
        f"{base['mw_mean']:.4f}), dP {st['dP_mean'] * 1e2:.3f}, electrons "
        f"at erase {nmin * 1e-4:.2e} cm^-2, law err {st['max_err_law_pct']:.3f} %")
OUT["S6_ambipolar"] = s6
OUT["S6_unipolar_reference"] = base

# ----------------------------------------------------------------------
log("S7 what sets the switched polarization")


def with_params(fn, **pv):
    old = {k: getattr(P, k) for k in pv}
    try:
        for k, v in pv.items():
            setattr(P, k, v)
        fefet._PSI_TAB.clear()
        return fn()
    finally:
        for k, v in old.items():
            setattr(P, k, v)
        fefet._PSI_TAB.clear()


s7 = []
grid = ([("Pr_CIPS", v) for v in [2.75e-2, 5.5e-2, 11e-2]]
        + [("Ec_CIPS", v) for v in [2.0e7, 3.0e7, 4.0e7]]
        + [("sigma_Ec", v) for v in [0.10, 0.22, 0.40]]
        + [("eps_FE_b", v) for v in [12.0, 25.0, 40.0]])
for name, val in grid:
    st = with_params(lambda: window_stats(eps_pct=0.0), **{name: val})
    Ec = val if name == "Ec_CIPS" else P.Ec_CIPS
    eb = val if name == "eps_FE_b" else P.eps_FE_b
    st.update(dict(param=name, value=val,
                   kappa=float(st["mw_mean"] / (2.0 * Ec * P.t_FE_default)),
                   E_f_over_Ec=float(st["E_f_over_Ec"] * P.Ec_CIPS / Ec),
                   E_b_over_Ec=float(st["E_b_over_Ec"] * P.Ec_CIPS / Ec),
                   dP_over_2eps0epsbEc=float(-st["dP_mean"] / (2 * P.eps0 * eb * Ec))))
    s7.append(st)
    log(f"  {name}={val:g}: MW {st['mw_mean']:.4f}, kappa {st['kappa']:.3f}, "
        f"dP {st['dP_mean'] * 1e2:.3f} uC/cm2, E_f/Ec {st['E_f_over_Ec']:.3f}, "
        f"E_b/Ec {st['E_b_over_Ec']:.3f}")
OUT["S7_switched_polarization"] = s7

# ----------------------------------------------------------------------
log("S9 back gate below the channel (300 nm SiO2)")
C_BG = P.eps0 * 3.9 / 300e-9
s9 = []
for xbg in [-10.0, -5.0, 0.0, 5.0, 10.0]:
    st = window_stats(eps_pct=0.0, C_BG=C_BG, x_BG=xbg)
    r, d, fw, bw = check(eps_pct=0.0, C_BG=C_BG, x_BG=xbg)
    i0 = int(np.argmin(np.abs(fw[:, 0])))
    I_prog, I_erase = float(bw[i0, 1]), float(fw[i0, 1])
    st.update(dict(x_BG=xbg, I_prog_x0_A=I_prog, I_erase_x0_A=I_erase,
                   onoff_x0=float(I_prog / I_erase)))
    s9.append(st)
    log(f"  x_BG {xbg:+5.1f} V: MW {st['mw_mean']:.4f}, dP "
        f"{st['dP_mean'] * 1e2:.3f} uC/cm2, law err "
        f"{st['max_err_law_pct']:.3f} %, on/off at x=0 {I_prog / I_erase:.2e}")
OUT["S9_back_gate"] = s9
OUT["S9_C_BG_uFcm2"] = float(C_BG * 1e2)

# ----------------------------------------------------------------------
log("S8 parameter sensitivity")


def headline():
    fefet._PSI_TAB.clear()
    mw = window_stats(eps_pct=0.0, seeds=(1, 2, 3, 4))["mw_mean"]
    mu0 = M.hole_mobility(0.0)
    mu1 = M.hole_mobility(-1.0)

    def ion(e):
        d = fefet.FeFET(eps_pct=e, n_dom=1600)
        d.fe.reset(-1)
        d.program(-6.0, 0.0)
        _, p_on = d.program(+6.0, 0.0)
        return d.drain_current(p_on)
    i0, i1 = ion(0.0), ion(-1.0)
    return dict(MW=float(mw), mu0_cm2=float(mu0 * 1e4),
                mu_gain=float(mu1 / mu0), Ion0_uA=float(i0 * 1e6),
                Ion_gain=float(i1 / i0))


ref = headline()
OUT["S8_reference"] = ref
log(f"  reference: {ref}")
pert = [("Pr_CIPS", 0.8, 1.2), ("Ec_CIPS", 0.8, 1.2), ("sigma_Ec", 0.8, 1.2),
        ("eps_FE_b", 0.8, 1.2), ("eps_hBN", 0.8, 1.2), ("Rc_W", 0.1, 7.4),
        ("mK_h", 0.9, 1.1), ("mG_h", 0.8, 1.2), ("dE_GK0", 0.8, 1.2),
        ("dE_GK_gauge", 0.8, 1.2), ("D_iv", 0.8, 1.2), ("n_imp", 0.5, 1.5)]
s8 = []
for name, lo, hi in pert:
    for fac in (lo, hi):
        val = getattr(P, name) * fac
        h = with_params(headline, **{name: val})
        row = dict(param=name, factor=fac)
        for k in ref:
            row[k] = h[k]
            row[k + "_change_pct"] = float(100.0 * (h[k] / ref[k] - 1.0))
        s8.append(row)
        log(f"  {name} x{fac}: " + ", ".join(
            f"{k} {row[k + '_change_pct']:+.1f}%" for k in ref))
# calibrated transport broadening and screening factor live in mobility.py
for name, lo, hi in [("SIGMA_IV", 0.8, 1.2), ("SCR_FACTOR", 0.8, 1.2)]:
    for fac in (lo, hi):
        old = getattr(M, name)
        setattr(M, name, old * fac)
        try:
            h = headline()
        finally:
            setattr(M, name, old)
        row = dict(param=name, factor=fac)
        for k in ref:
            row[k] = h[k]
            row[k + "_change_pct"] = float(100.0 * (h[k] / ref[k] - 1.0))
        s8.append(row)
        log(f"  {name} x{fac}: " + ", ".join(
            f"{k} {row[k + '_change_pct']:+.1f}%" for k in ref))
OUT["S8_sensitivity"] = s8

OUT["runtime_s"] = float(time.time() - T0)
with open("revision_results.json", "w") as fh:
    json.dump(OUT, fh, indent=1)
log("done")
