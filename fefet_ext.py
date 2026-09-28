"""
fefet_ext.py
Generalized MFMIS FeFET used for the robustness studies of the
decoupling law (revision study, run by revision_studies.py).

FeFETX extends fefet.FeFET with options that relax, one at a time, the
assumptions under which the constant-current memory-window law is
derived. With every option at its default the solver is the same MFMIS
model as fefet.FeFET (checked in revision_studies.py).

Options
-------
A_R          area ratio A_FE / A_MIS of the ferroelectric capacitor to the
             channel (MIS) area. Charge neutrality of the floating gate
             gives D_FE = D_MIS / A_R.
dos_scale    multiplies the density of states of both hole valleys
             (a low-density-of-states channel for dos_scale << 1).
ambipolar    if True, the channel also holds electrons, with a band gap
             Eg (eV) and a conduction-band density of states equal to
             the K-valley one times dos_scale; the channel can then
             screen the depolarization field when it is inverted.
mu_bwd_ratio mobility on the backward (erase-to-program return) branch
             divided by the mobility on the forward branch.
mu_of_p      if True, the mobility is evaluated at the instantaneous hole
             density from the Kubo-Greenwood model (tabulated).
mu_trap      if True, trapped interface charge adds Coulomb scatterers,
             n_imp -> n_imp + |Q_it|/q, at every bias point, which makes
             the mobility branch dependent whenever Q_it is.
C_BG, x_BG   capacitance per area (F/m^2) and drive (V, hole-accumulating
             positive) of an optional back gate below the channel, as in
             devices on SiO2/Si; it adds charge C_BG (x_BG - psi) to the
             channel-side balance.
sigma_FE0    amplitude (C/m^2) of a ferroelectric-side charge sheet
             (injected charge, migrated Cu ions, or floating-gate charge
             from leakage) that follows the ferroelectric field,
             sigma_FE = -sigma_FE0 tanh(E_FE / E_inj), i.e. it screens
             the polarization. d_frac is its distance from the floating
             gate as a fraction of t_FE (0: on the floating gate).

Stack equation solved for the surface potential psi (strictly monotonic):
    D      = Q_ch(psi) + Q_it - C_BG (x_BG - psi)   (MIS displacement)
    D_FE   = D / A_R
    E_1    = (D_FE - P_sw) / (eps0 eps_b)     (between floating gate and sheet)
    x      = t_FE E_1 + (1 - d_frac) t_FE sigma_FE / (eps0 eps_b)
             + D / C_ins + psi
"""
import numpy as np
from scipy.optimize import brentq
import params as P
import mobility as M
import fefet
from fefet import FeFET


class FeFETX(FeFET):
    def __init__(self, A_R=1.0, dos_scale=1.0, ambipolar=False, Eg=1.6,
                 mu_bwd_ratio=1.0, mu_of_p=False, mu_trap=False,
                 sigma_FE0=0.0, E_inj=None, d_frac=0.0, C_BG=0.0,
                 x_BG=0.0, **kw):
        super().__init__(**kw)
        self.A_R = A_R
        self.dos_scale = dos_scale
        self.ambipolar = ambipolar
        self.Eg = Eg
        self.mu_bwd_ratio = mu_bwd_ratio
        self.mu_of_p = mu_of_p
        self.mu_trap = mu_trap
        self.sigma_FE0 = sigma_FE0
        self.E_inj = P.Ec_CIPS if E_inj is None else E_inj
        self.d_frac = d_frac
        self.C_BG = C_BG
        self.x_BG = x_BG
        self.sigma_FE = 0.0
        self.branch = "f"
        self.last_psi = None
        self.last_E = 0.0
        if mu_of_p:
            ps = np.logspace(13, 18, 26)       # m^-2 (1e9 to 1e14 cm^-2)
            self._mu_tab = (np.log(ps), np.array(
                [M.hole_mobility(self.eps, p_sheet=pp) for pp in ps]))
        if mu_trap:
            n0 = P.n_imp
            extra = np.concatenate([[0.0], np.logspace(14, 17.5, 15)])
            vals = []
            for e_ in extra:
                P.n_imp = n0 + e_
                vals.append(M.hole_mobility(self.eps))
            P.n_imp = n0
            self._mut_tab = (extra, np.array(vals))

    # ------------------------------------------------------------------
    def p_hole(self, psi):
        """Both hole valleys, Fermi-Dirac, without the exponent clipping of
        fefet.p_of_psi (identical within 1e-26 relative in its range), so
        that very low density-of-states channels are handled correctly."""
        aK = (np.asarray(psi, dtype=float) - fefet.phi_F(self.eps)) / P.kT_eV
        pK = fefet.NK_dos() * P.kT * np.logaddexp(0.0, aK)
        pG = 0.0
        if fefet.TWO_VALLEY_ES:
            aG = aK - fefet.dE_GK(self.eps) / P.kT_eV
            pG = fefet.NG_dos() * P.kT * np.logaddexp(0.0, aG)
        return self.dos_scale * (pK + pG)

    def n_elec(self, psi):
        if not self.ambipolar:
            return 0.0
        NC = self.dos_scale * fefet.NK_dos()
        a = (-(psi - fefet.phi_F(self.eps)) - self.Eg) / P.kT_eV
        return NC * P.kT * np.log1p(np.exp(np.clip(a, -60, 60)))

    def Q_ch(self, psi):
        return P.q * (self.p_hole(psi) - self.n_elec(psi))

    def _x_of_psi(self, psi, Psw, Qs, sig):
        D = self.Q_ch(psi) + Qs - self.C_BG * (self.x_BG - psi)
        E1 = (D / self.A_R - Psw) / (P.eps0 * P.eps_FE_b)
        return (self.t_FE * E1
                + (1.0 - self.d_frac) * self.t_FE * sig / (P.eps0 * P.eps_FE_b)
                + D / self.C_hBN + psi), E1

    def _solve_field(self, x):
        Psw = self.fe.P_switch()
        Qs = self.Q_slow()
        sig = self.sigma_FE
        g = lambda s: self._x_of_psi(s, Psw, Qs, sig)[0] - x
        lo, hi = -80.0, 20.0
        psi = brentq(g, lo, hi, xtol=1e-12, maxiter=200)
        _, E1 = self._x_of_psi(psi, Psw, Qs, sig)
        # field seen by the hysterons: average over the ferroelectric
        E_avg = E1 + (1.0 - self.d_frac) * sig / (P.eps0 * P.eps_FE_b)
        self.last_psi = psi
        self.last_E = E_avg
        p = float(self.p_hole(psi))
        # same depletion criterion as fefet.FeFET: a hole density below the
        # one carried by a 1e-4 V/m field increment counts as fully
        # depleted (p = 0), which empties the bounding-model traps
        if p < P.eps0 * P.eps_FE_b * 1e-4 / P.q:
            p = 0.0
        return E_avg, p

    def solve_bias(self, x, **kw):
        E, p = super().solve_bias(x, **kw)
        if self.sigma_FE0 != 0.0:
            # quasi-static ferroelectric-side charge, iterated to consistency
            for _ in range(60):
                new = -self.sigma_FE0 * np.tanh(E / self.E_inj)
                if abs(new - self.sigma_FE) < 1e-9 * abs(self.sigma_FE0):
                    break
                self.sigma_FE = 0.5 * self.sigma_FE + 0.5 * new
                E, p = super().solve_bias(x, **kw)
        return E, p

    # ------------------------------------------------------------------
    def mobility(self, p):
        mu = self.mu
        if self.mu_of_p:
            lp, mt = self._mu_tab
            mu = float(np.interp(np.log(max(p, 1e13)), lp, mt))
        if self.mu_trap and self.Dit > 0.0:
            ex, mt = self._mut_tab
            mu *= float(np.interp(abs(self.Q_slow()) / P.q, ex, mt)) / mt[0]
        if self.branch == "b":
            mu *= self.mu_bwd_ratio
        return mu

    def drain_current(self, p, VDS=None):
        VDS = abs(P.VDS_read) if VDS is None else abs(VDS)
        if p <= 1e6:
            return 1e-13
        G_ch = (P.W_ch / P.L_ch) * P.q * p * self.mobility(p)
        R_c = 2.0 * P.Rc_W / P.W_ch
        return max(VDS / (1.0 / G_ch + R_c), 1e-13)

    def C_series(self):
        """Series capacitance of ferroelectric background and interlayer
        per unit channel area, the denominator of the density-mismatch
        correction."""
        return 1.0 / (self.t_FE / (self.A_R * P.eps0 * P.eps_FE_b)
                      + 1.0 / self.C_hBN)


def probe(dev, x_max=6.0, n=401, I_crit=None, x_start=None):
    """Double sweep with a record at every bias point, and the linearly
    interpolated state at the constant-current crossing of each branch.
    Record columns: x, I, P_sw, Q_it, p, psi, E_FE, sigma_FE."""
    if I_crit is None:
        I_crit = 1e-7 * (P.W_ch / 1e-6)
    fefet._PSI_TAB.clear()

    def run(xs, br):
        dev.branch = br
        rec = []
        for x in xs:
            _, p = dev.solve_bias(x)
            rec.append((x, dev.drain_current(p), dev.fe.P_switch(),
                        dev.Q_slow(), p, dev.last_psi, dev.last_E,
                        dev.sigma_FE))
        return np.array(rec)

    dev.fe.reset(-1)
    dev.reset_traps()
    run(np.linspace(0.0, -x_max, 50), "f")
    fwd = run(np.linspace(-x_max, x_max, n), "f")
    bwd = run(np.linspace(x_max, -x_max, n), "b")[::-1]

    def cross(rec):
        I = rec[:, 1]
        idx = np.where(I > I_crit)[0]
        if len(idx) == 0 or idx[0] == 0:
            return None
        i = idx[0]
        y0, y1 = np.log10(I[i - 1]), np.log10(I[i])
        if y1 == y0:
            return rec[i]
        w = (np.log10(I_crit) - y0) / (y1 - y0)
        return rec[i - 1] + w * (rec[i] - rec[i - 1])

    return cross(fwd), cross(bwd), fwd, bwd


def law_terms(dev, f, b):
    """Simulated window, the law (Eq. 7 generalized), and the
    density-mismatch correction evaluated from the crossing records."""
    denom = P.eps0 * P.eps_FE_b
    mw_sim = float(f[0] - b[0])
    dP = float(f[2] - b[2])
    dQ = float(f[3] - b[3])
    dS = float(f[7] - b[7])
    law = (dev.t_FE * (dQ / dev.A_R - dP) / denom
           + (1.0 - dev.d_frac) * dev.t_FE * dS / denom
           + dQ / dev.C_hBN)
    dp = float(f[4] - b[4])
    dpsi = float(f[5] - b[5])
    corr = P.q * dp / dev.C_series() + dpsi
    return dict(mw_sim=mw_sim, law=law, dP=dP, dQ=dQ, dS=dS,
                p_f=float(f[4]), p_b=float(b[4]), dpsi=dpsi, corr=corr,
                err_law_pct=100.0 * (law - mw_sim) / mw_sim,
                err_corr_pct=100.0 * (law + corr - mw_sim) / mw_sim,
                E_f=float(f[6]), E_b=float(b[6]))
