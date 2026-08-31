"""
fefet.py
Self-consistent electrostatics and transfer characteristics of the
MFMIS p-type FeFET: top gate / CIPS / floating metal gate / h-BN / WSe2.

Convention: the drive variable is the hole overdrive x = -(VG - VFB), so
x > 0 accumulates holes. With zero net charge on the floating gate and
ideal insulators, displacement continuity gives

    x = t_FE E_FE + D / C_hBN + psi_s(p),   D = eps0 epsb E_FE + P_sw,
    p = D / q  (hole sheet density, D >= 0; the h-BN blocks electrons).

P_sw is the switching polarization of the Preisach hysteron ensemble.
Because Pr of CIPS (5.5 uC/cm^2) far exceeds the available channel
charge, the stable states are partially switched minor-loop states: the
solver flips hysterons one small batch at a time until the depolarizing
field no longer exceeds any remaining coercive field. This reproduces
the self-limited, charge-matched polarization observed in real 2D
FeFETs and yields the retained memory window directly.

Drain current (read regime, linear, with contact resistance):
    ID = q p mu_h(eps) (W/L) VDS_eff.
"""
import numpy as np
from scipy.optimize import brentq
import params as P
import mobility as M
from ferro import PreisachFE


# Flat-band hole Fermi offset (E_VBM to E_F at psi_s = 0), set by the
# TOS-style p-doping of the channel (Zhao 2026); strain moves the K-VBM.
PHI_F0 = 0.15  # eV


def phi_F(eps_pct):
    """Effective flat-band offset with the strain-induced K-VBM shift."""
    return PHI_F0 + P.dEK_gauge * eps_pct


def NK_dos():
    return P.gK * P.mK_h * P.m0 / (2.0 * np.pi * P.hbar ** 2) * 2.0


def NG_dos():
    """Gamma-valley 2D density of states (heavy holes)."""
    return P.gG * P.mG_h * P.m0 / (2.0 * np.pi * P.hbar ** 2) * 2.0


def dE_GK(eps_pct):
    """Gamma-K separation (eV), positive when Gamma lies below the K-VBM.
    Same strain gauge used by the transport model (mobility.valley_edges),
    so the electrostatics and the transport see one consistent band
    structure."""
    return P.dE_GK0 - P.dE_GK_gauge * eps_pct


# Set False to recover the single-valley (K-only) electrostatics used in
# the first version of this model; kept for regression testing.
TWO_VALLEY_ES = True


def p_of_psi(psi, eps_pct):
    """Hole sheet density (m^-2) at surface potential psi (V).

    Both hole valleys are filled with Fermi-Dirac statistics. The Gamma
    valley sits dE_GK(eps) below the K-VBM and carries roughly three
    times the K-valley density of states (gG*mG = 2.20 against
    gK*mK = 0.72), so its strain-driven population changes the channel
    quantum capacitance substantially."""
    NK = NK_dos()
    aK = (np.asarray(psi, dtype=float) - phi_F(eps_pct)) / P.kT_eV
    pK = NK * P.kT * np.log1p(np.exp(np.clip(aK, -60, 60)))
    if not TWO_VALLEY_ES:
        return pK
    NG = NG_dos()
    aG = aK - dE_GK(eps_pct) / P.kT_eV
    pG = NG * P.kT * np.log1p(np.exp(np.clip(aG, -60, 60)))
    return pK + pG


_PSI_TAB = {}


def _psi_table(eps_pct):
    """Monotonic (log p, psi) table for inverting p_of_psi at this strain.
    Built once per strain value; the inversion is needed inside the
    self-consistent field solve, so an analytic-quality interpolation is
    much cheaper than a nested root find."""
    key = round(float(eps_pct), 9)
    tab = _PSI_TAB.get(key)
    if tab is None:
        psi_g = np.linspace(phi_F(eps_pct) - 2.5, phi_F(eps_pct) + 2.5, 24001)
        p_g = p_of_psi(psi_g, eps_pct)
        good = p_g > 0
        tab = (np.log(p_g[good]), psi_g[good])
        _PSI_TAB[key] = tab
    return tab


def psi_of_p(p, eps_pct):
    """Surface potential (V) needed to hold hole density p (m^-2)."""
    if p <= 0:
        return -10.0
    if not TWO_VALLEY_ES:
        NK = NK_dos()
        arg = p / (NK * P.kT)
        if arg < 30:
            return phi_F(eps_pct) + P.kT_eV * np.log(np.expm1(arg))
        return phi_F(eps_pct) + P.kT_eV * arg
    lp, ps = _psi_table(eps_pct)
    return float(np.interp(np.log(p), lp, ps))


def quantum_cap(psi, eps_pct, h=1e-4):
    """Channel quantum capacitance C_Q = q dp/dpsi (F/m^2)."""
    return P.q * (p_of_psi(psi + h, eps_pct)
                  - p_of_psi(psi - h, eps_pct)) / (2.0 * h)


T_CH = 0.65e-9  # monolayer WSe2 thickness (m), for 2D -> 3D densities


class FeFET:
    def __init__(self, t_FE=None, t_hBN=None, eps_pct=0.0, n_dom=6400,
                 Dit_cm2=0.0, trap_mode="worst", sigma_p_cm2=1e-15,
                 n_trap_bins=24, seed=7):
        """Interface traps at the WSe2/h-BN interface: uniform density
        Dit_cm2 (cm^-2 eV^-1) over a 0.6 eV band of gap states.

        trap_mode = "worst": bounding quasi-static model (instantaneous
        capture while the surface potential rises, no re-emission while
        the channel conducts, complete emission at full depletion).

        trap_mode = "dynamic": finite-rate Shockley-Read-Hall kinetics
        with energy-resolved occupancies. Capture rate c = sigma v_th
        p3d; emission follows detailed balance, e_i = sigma v_th
        p3d(psi_t,i), i.e. the free density that would coexist with the
        Fermi level at the trap level, so no extra parameter beyond the
        capture cross section sigma_p is introduced. Pass dwell times dt
        to solve_bias/sweep/program/hold to evolve the occupancies.

        Dit_cm2 = 0 reproduces the trap-free device bit-for-bit."""
        self.t_FE = P.t_FE_default if t_FE is None else t_FE
        self.t_hBN = P.t_hBN if t_hBN is None else t_hBN
        self.C_hBN = P.eps0 * P.eps_hBN / self.t_hBN
        self.eps = eps_pct
        self.fe = PreisachFE(n_dom=n_dom, seed=seed)
        self.mu = M.hole_mobility(eps_pct)  # m^2/Vs
        # --- interface trap band ---
        self.Dit = Dit_cm2 * 1e4          # states / (m^2 eV)
        self.trap_band = 0.60             # eV of gap states that can charge
        self.trap_lo = phi_F(eps_pct) - 0.45
        self.trap_mode = trap_mode
        self.psi_hist = -1e9              # worst mode: highest psi_s seen
        # dynamic mode: SRH kinetics
        self.sigma_p = sigma_p_cm2 * 1e-4          # m^2
        m_eff = P.mK_h * P.m0
        self.v_th = np.sqrt(3.0 * P.kT / m_eff)    # m/s
        self.n_bins = n_trap_bins
        edges = np.linspace(self.trap_lo, self.trap_lo + self.trap_band,
                            n_trap_bins + 1)
        self.trap_lev = 0.5 * (edges[:-1] + edges[1:])  # bin centers (V)
        self.trap_f = np.zeros(n_trap_bins)             # hole occupancy
        # detailed-balance emission rates per bin (1/s), fixed by levels
        self._e_rate = self.sigma_p * self.v_th * \
            p_of_psi(self.trap_lev, eps_pct) / T_CH

    def Q_slow(self):
        """Trapped hole charge per area (C/m^2)."""
        if self.Dit <= 0.0:
            return 0.0
        if self.trap_mode == "dynamic":
            dN = self.Dit * self.trap_band / self.n_bins  # states/m^2/bin
            return P.q * dN * float(np.sum(self.trap_f))
        dE = min(max(self.psi_hist - self.trap_lo, 0.0), self.trap_band)
        return P.q * self.Dit * dE

    def reset_traps(self):
        self.psi_hist = -1e9
        self.trap_f[:] = 0.0

    def advance_traps(self, p, dt):
        """Evolve dynamic trap occupancies over dwell time dt (s) with
        the channel at hole density p (m^-2)."""
        if self.Dit <= 0.0 or self.trap_mode != "dynamic" or dt <= 0.0:
            return
        p3d = max(p, 0.0) / T_CH
        c = self.sigma_p * self.v_th * p3d          # capture rate (1/s)
        rate = c + self._e_rate
        f_eq = np.where(rate > 0, c / np.maximum(rate, 1e-300), 0.0)
        self.trap_f = f_eq + (self.trap_f - f_eq) * np.exp(-rate * dt)

    # ------------------------------------------------------------------
    def _solve_field(self, x):
        """Field E_FE for drive x with the hysteron state frozen."""
        Psw = self.fe.P_switch()
        Qs = self.Q_slow()
        E0 = -(Psw - Qs) / (P.eps0 * P.eps_FE_b)  # D = Qs (p = 0)

        def f(E_):
            D = P.eps0 * P.eps_FE_b * E_ + Psw   # displacement in the stack
            p = (D - Qs) / P.q
            return self.t_FE * E_ + D / self.C_hBN + psi_of_p(p, self.eps) - x

        lo = E0 + 1e-4
        hi = 5e8
        if f(lo) >= 0.0:
            # depletion branch: no free hole charge, D = Q_slow
            return E0, 0.0
        E = brentq(f, lo, hi, xtol=0.5)
        D = P.eps0 * P.eps_FE_b * E + Psw
        p = (D - Qs) / P.q
        return E, max(p, 0.0)

    def solve_bias(self, x, max_flips=4000, batch=4, dt=None):
        """Self-consistent bias point: alternate field solution and
        self-limited hysteron flipping (lowest-coercive-field first).
        In dynamic trap mode, dt is the dwell time at this bias."""
        E, p = self._solve_field(x)
        flips = 0
        while flips < max_flips:
            up = np.where((self.fe.state < 0) & (E > self.fe.Ecs))[0]
            dn = np.where((self.fe.state > 0) & (-E > self.fe.Ecs))[0]
            if len(up) == 0 and len(dn) == 0:
                break
            if len(up) > 0:
                order = up[np.argsort(self.fe.Ecs[up])][:batch]
                self.fe.state[order] = 1.0
            else:
                order = dn[np.argsort(self.fe.Ecs[dn])][:batch]
                self.fe.state[order] = -1.0
            flips += batch
            E, p = self._solve_field(x)
        if self.Dit > 0.0:
            if self.trap_mode == "dynamic":
                if dt is not None and dt > 0.0:
                    self.advance_traps(p, dt)
                    E, p = self._solve_field(x)
            else:
                # worst-case bound: instantaneous capture while the
                # surface potential rises, no re-emission while the
                # channel conducts, complete emission at full depletion
                if p > 0.0:
                    psi = psi_of_p(p, self.eps)
                    if psi > self.psi_hist:
                        self.psi_hist = psi
                        E, p = self._solve_field(x)
                else:
                    self.psi_hist = -1e9
        return E, p

    def hold(self, t_hold, x_hold=0.0, n_sub=30):
        """Evolve the device at fixed drive x_hold for t_hold seconds
        (dynamic trap mode); returns (times, hole densities)."""
        ts = np.logspace(-8, np.log10(max(t_hold, 1e-7)), n_sub)
        dts = np.diff(np.concatenate([[0.0], ts]))
        out_p = []
        for dt in dts:
            _, p = self.solve_bias(x_hold, dt=dt)
            out_p.append(p)
        return ts, np.array(out_p)

    # ------------------------------------------------------------------
    def drain_current(self, p, VDS=None):
        """Read-current magnitude (A), linear regime with Rc."""
        VDS = abs(P.VDS_read) if VDS is None else abs(VDS)
        if p <= 1e6:
            return 1e-13
        G_ch = (P.W_ch / P.L_ch) * P.q * p * self.mu
        R_c = 2.0 * P.Rc_W / P.W_ch
        I = VDS / (1.0 / G_ch + R_c)
        return max(I, 1e-13)

    def program(self, x_prog, x_hold=0.0, t_pulse=None):
        """Apply a program drive and return to hold, tracking states.
        In dynamic trap mode, t_pulse is the total pulse duration (s)."""
        dt = None if t_pulse is None else t_pulse / 80.0
        for x in np.linspace(x_hold, x_prog, 40):
            self.solve_bias(x, dt=dt)
        for x in np.linspace(x_prog, x_hold, 40):
            self.solve_bias(x, dt=dt)
        return self.solve_bias(x_hold, dt=dt)

    def sweep(self, x_max=6.0, n=401, t_total=None):
        """Double sweep x: -x_max -> +x_max -> -x_max. In dynamic trap
        mode, t_total is the duration of the full double sweep (s)."""
        self.fe.reset(-1)
        self.reset_traps()
        dt = None if t_total is None else t_total / (2 * n + 50)
        for x in np.linspace(0.0, -x_max, 50):
            self.solve_bias(x, dt=dt)
        xs_f = np.linspace(-x_max, x_max, n)
        xs_b = xs_f[::-1]
        I_f, p_f = [], []
        for x in xs_f:
            _, p = self.solve_bias(x, dt=dt)
            p_f.append(p)
            I_f.append(self.drain_current(p))
        I_b, p_b = [], []
        for x in xs_b:
            _, p = self.solve_bias(x, dt=dt)
            p_b.append(p)
            I_b.append(self.drain_current(p))
        return (xs_f, np.array(I_f), xs_b, np.array(I_b),
                np.array(p_f), np.array(p_b))


def memory_window_ensemble(eps_pct=0.0, n_seeds=8, n_dom=6400, x_max=6.0,
                           n=401, **kw):
    """Memory window averaged over independent draws of the coercive-field
    distribution, returned as (mean, standard deviation, all values).

    A single draw of a finite hysteron ensemble carries a statistical
    spread that does not vanish with grid refinement alone: at n_dom = 400
    the window scatters by about 14 % from seed to seed, at n_dom = 6400
    by about 2 %. Reported windows therefore quote the ensemble mean and
    its spread rather than one realization."""
    vals = []
    for s in range(1, n_seeds + 1):
        d = FeFET(eps_pct=eps_pct, n_dom=n_dom, seed=s, **kw)
        sw = d.sweep(x_max=x_max, n=n)
        vals.append(memory_window(*sw[:4])[0])
    v = np.array(vals, dtype=float)
    return float(v.mean()), float(v.std()), v


def memory_window(xs_f, I_f, xs_b, I_b, I_crit=None):
    """Constant-current memory window (V)."""
    if I_crit is None:
        I_crit = 1e-7 * (P.W_ch / 1e-6)  # 100 nA per um of width

    def vt(xs, Is):
        idx = np.where(Is > I_crit)[0]
        if len(idx) == 0 or idx[0] == 0:
            return np.nan
        i = idx[0]
        x0, x1 = xs[i - 1], xs[i]
        y0, y1 = np.log10(Is[i - 1]), np.log10(Is[i])
        return x0 + (np.log10(I_crit) - y0) * (x1 - x0) / (y1 - y0)

    vt_f = vt(xs_f, I_f)
    vt_b = vt(xs_b[::-1], I_b[::-1])
    return vt_f - vt_b, vt_f, vt_b


if __name__ == "__main__":
    import time
    t0 = time.time()
    for eps in [0.0, -0.5]:
        d = FeFET(eps_pct=eps)
        xs_f, I_f, xs_b, I_b, p_f, p_b = d.sweep()
        mw, vtf, vtb = memory_window(xs_f, I_f, xs_b, I_b)
        onoff = I_f.max() / max(I_f.min(), 1e-13)
        print(f"eps={eps:+.1f}%: mu={d.mu*1e4:6.1f} cm2/Vs  MW={mw:.2f} V "
              f"(VTf={vtf:.2f}, VTb={vtb:.2f})  Imax={I_f.max()*1e6:.2f} uA  "
              f"on/off={onoff:.1e}")
    print(f"[{time.time()-t0:.1f} s]")
