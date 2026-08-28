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


def psi_of_p(p, eps_pct):
    """Surface potential (V) needed to hold hole density p (m^-2)."""
    if p <= 0:
        return -10.0
    NK = NK_dos()
    arg = p / (NK * P.kT)
    if arg < 30:
        val = np.expm1(arg)
        return phi_F(eps_pct) + P.kT_eV * np.log(val)
    return phi_F(eps_pct) + P.kT_eV * arg


class FeFET:
    def __init__(self, t_FE=None, t_hBN=None, eps_pct=0.0, n_dom=400,
                 Dit_cm2=0.0):
        """Dit_cm2: interface trap density at the WSe2/h-BN interface in
        cm^-2 eV^-1, treated as slow (worst-case) traps: they capture
        holes as the surface potential rises but do not re-emit within a
        sweep, producing the volatile clockwise hysteresis component that
        opposes the ferroelectric window. Dit_cm2 = 0 reproduces the
        trap-free device bit-for-bit."""
        self.t_FE = P.t_FE_default if t_FE is None else t_FE
        self.t_hBN = P.t_hBN if t_hBN is None else t_hBN
        self.C_hBN = P.eps0 * P.eps_hBN / self.t_hBN
        self.eps = eps_pct
        self.fe = PreisachFE(n_dom=n_dom)
        self.mu = M.hole_mobility(eps_pct)  # m^2/Vs
        # --- interface trap model (slow, worst case) ---
        self.Dit = Dit_cm2 * 1e4          # states / (m^2 eV)
        self.trap_band = 0.60             # eV of gap states that can charge
        # traps start filling once psi_s enters the subthreshold band
        self.trap_lo = phi_F(eps_pct) - 0.45
        self.psi_hist = -1e9              # highest psi_s seen so far

    def Q_slow(self):
        """Frozen (captured) trap charge per area (C/m^2)."""
        if self.Dit <= 0.0:
            return 0.0
        dE = min(max(self.psi_hist - self.trap_lo, 0.0), self.trap_band)
        return P.q * self.Dit * dE

    def reset_traps(self):
        self.psi_hist = -1e9

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

    def solve_bias(self, x, max_flips=4000, batch=4):
        """Self-consistent bias point: alternate field solution and
        self-limited hysteron flipping (lowest-coercive-field first)."""
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
        # slow-trap dynamics (quasi-static worst case for the window):
        # capture whenever the surface potential exceeds the highest value
        # seen so far; complete emission once the channel is fully
        # depleted (trap levels lifted above the Fermi level at the erase
        # extreme). No emission while the channel conducts.
        if self.Dit > 0.0:
            if p > 0.0:
                psi = psi_of_p(p, self.eps)
                if psi > self.psi_hist:
                    self.psi_hist = psi
                    # captured charge screens the gate; re-solve once
                    E, p = self._solve_field(x)
            else:
                self.psi_hist = -1e9
        return E, p

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

    def program(self, x_prog, x_hold=0.0):
        """Apply a program drive and return to hold, tracking states."""
        for x in np.linspace(x_hold, x_prog, 40):
            self.solve_bias(x)
        for x in np.linspace(x_prog, x_hold, 40):
            self.solve_bias(x)
        return self.solve_bias(x_hold)

    def sweep(self, x_max=6.0, n=401):
        """Double sweep x: -x_max -> +x_max -> -x_max."""
        self.fe.reset(-1)
        self.reset_traps()
        for x in np.linspace(0.0, -x_max, 50):
            self.solve_bias(x)
        xs_f = np.linspace(-x_max, x_max, n)
        xs_b = xs_f[::-1]
        I_f, p_f = [], []
        for x in xs_f:
            _, p = self.solve_bias(x)
            p_f.append(p)
            I_f.append(self.drain_current(p))
        I_b, p_b = [], []
        for x in xs_b:
            _, p = self.solve_bias(x)
            p_b.append(p)
            I_b.append(self.drain_current(p))
        return (xs_f, np.array(I_f), xs_b, np.array(I_b),
                np.array(p_f), np.array(p_b))


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
