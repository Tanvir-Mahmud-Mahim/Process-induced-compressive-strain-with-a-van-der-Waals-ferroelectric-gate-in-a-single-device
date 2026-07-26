"""
circuit.py
Circuit-level projection of strain-augmented complementary nonvolatile
logic built from the p-type CIPS/WSe2 FeFET (pull-up) and a WSe2 n-type
FET (pull-down), replacing the 10 MOhm resistor load of the
experimental CIPS latch (Lee 2026).

Transistors use a smooth EKV-style all-region model whose threshold
voltages are extracted from the self-consistent FeFET simulation
(fefet.py) and whose gain factor scales with the strain-dependent
mobility mu_h(eps) from the two-valley transport model (mobility.py).
"""
import numpy as np
from scipy.optimize import brentq
import params as P
import mobility as M

PHI_T = 0.02585  # V at 300 K
N_SS = 1.35      # subthreshold slope factor


def _veff(Vov, n=N_SS):
    """Smooth overdrive: -> Vov for Vov >> phi_t, exponential below VT."""
    u = Vov / (2.0 * n * PHI_T)
    return 2.0 * n * PHI_T * np.log1p(np.exp(np.clip(u, -60, 60)))


def nfet_current(VG, VS, VD, k_n, VT):
    """n-FET drain current (A): smooth all-region square-law model with
    subthreshold blending and drain saturation."""
    VGS = VG - VS
    VDS = VD - VS
    if VDS < 0:
        return -nfet_current(VG, VD, VS, k_n, VT)
    Ve = _veff(VGS - VT)
    sat = 1.0 - np.exp(-VDS / max(Ve, PHI_T))
    return 0.5 * k_n * Ve ** 2 * sat


def pfet_current(VG, VS, VD, k_p, VTp_mag):
    """p-FET source-drain current (A, positive from source VS into drain
    VD). VTp_mag is the threshold magnitude referred to VSG (negative for
    a programmed, normally-on state)."""
    VSG = VS - VG
    VSD = VS - VD
    if VSD < 0:
        return -pfet_current(VG, VD, VS, k_p, VTp_mag)
    Ve = _veff(VSG - VTp_mag)
    sat = 1.0 - np.exp(-VSD / max(Ve, PHI_T))
    return 0.5 * k_p * Ve ** 2 * sat


def stack_Ceff(t_FE=None, t_hBN=None):
    t_FE = P.t_FE_default if t_FE is None else t_FE
    t_hBN = P.t_hBN if t_hBN is None else t_hBN
    return 1.0 / (t_FE / (P.eps0 * P.eps_FE_b) + t_hBN / (P.eps0 * P.eps_hBN))


def k_pfet(eps_pct, t_FE=None, t_hBN=None):
    """Gain factor of the p-FeFET, k = mu Ceff W/L (A/V^2)."""
    mu = M.hole_mobility(eps_pct)
    return mu * stack_Ceff(t_FE, t_hBN) * (P.W_ch / P.L_ch)


def k_nfet():
    C_n = P.eps0 * P.eps_hBN / P.t_hBN  # h-BN gated n-FET (Lee 2026 style)
    return P.mu_n * C_n * (P.W_ch / P.L_ch)


class NVInverter:
    """Complementary nonvolatile inverter: p-FeFET pull-up (source at VDD)
    with state-dependent threshold, n-FET pull-down."""

    def __init__(self, eps_pct=0.0, VT_p_states=(0.80, -0.43), state=0,
                 VDD=None):
        self.VDD = P.VDD if VDD is None else VDD
        self.k_p = k_pfet(eps_pct)
        self.k_n = k_nfet()
        self.VT_p_states = VT_p_states  # (erased/high-VT, programmed/low-VT)
        self.state = state              # 0: high-VT, 1: low-VT (programmed on)
        self.eps = eps_pct

    @property
    def VT_p(self):
        return self.VT_p_states[self.state]

    def currents(self, Vin, Vout):
        Ip = pfet_current(Vin, self.VDD, Vout, self.k_p, self.VT_p)
        In = nfet_current(Vin, 0.0, Vout, self.k_n, P.VT_n)
        return Ip, In

    def vout(self, Vin):
        def f(Vo):
            Ip, In = self.currents(Vin, Vo)
            return Ip - In
        lo, hi = 1e-6, self.VDD - 1e-6
        flo, fhi = f(lo), f(hi)
        if flo <= 0:
            return 0.0
        if fhi >= 0:
            return self.VDD
        return brentq(f, lo, hi, xtol=1e-9)

    def vtc(self, n=201):
        Vin = np.linspace(0, self.VDD, n)
        return Vin, np.array([self.vout(v) for v in Vin])

    def transient(self, Vin, V0, t_end=None, n=4000, C=None):
        """Integrate C dVout/dt = Ip - In for a step input."""
        C = P.C_load if C is None else C
        # crude time scale estimate
        I_scale = max(self.k_p, self.k_n) * self.VDD ** 2 / 2.0
        tau0 = C * self.VDD / max(I_scale, 1e-12)
        t_end = 20.0 * tau0 if t_end is None else t_end
        t = np.linspace(0, t_end, n)
        dt = t[1] - t[0]
        V = np.empty_like(t)
        V[0] = V0
        for i in range(1, n):
            Ip, In = self.currents(Vin, V[i - 1])
            V[i] = V[i - 1] + dt * (Ip - In) / C
            V[i] = np.clip(V[i], 0.0, self.VDD)
        return t, V

    def delay_energy(self, C=None):
        """Propagation delay (50% crossing) and dynamic energy for a
        falling input (output pulled up by the p-FeFET, programmed state)."""
        C = P.C_load if C is None else C
        old_state = self.state
        self.state = 1  # read with programmed pull-up
        t, V = self.transient(Vin=0.0, V0=0.0, C=C)
        idx = np.where(V >= 0.5 * self.VDD)[0]
        tpLH = t[idx[0]] if len(idx) else np.inf
        self.state = old_state
        E_dyn = C * self.VDD ** 2
        return tpLH, E_dyn

    def static_power(self):
        """Static power in both logic states (W)."""
        # output high (input low): n-FET leaks
        In_off = nfet_current(0.0, 0.0, self.VDD, self.k_n, P.VT_n)
        # output low (input high): p-FeFET leaks (worst case: high-VT state)
        Ip_off = pfet_current(self.VDD, self.VDD, 0.0, self.k_p,
                              self.VT_p_states[0])
        return In_off * self.VDD, Ip_off * self.VDD


def resistor_inverter_vtc(eps_pct=0.0, R=10e6, VT_state=1, n=201,
                          VT_p_states=(0.80, -0.43), VDD=None):
    """Reference: experimental-style resistor-load inverter (Lee 2026)
    built with the same p-FeFET as driver (pull-down configuration is
    n-FeFET in the experiment; here we model the resistor limit)."""
    VDD = P.VDD if VDD is None else VDD
    k_p = k_pfet(eps_pct)
    Vin = np.linspace(0, VDD, n)
    Vout = []
    for v in Vin:
        def f(Vo):
            Ip = pfet_current(v, VDD, Vo, k_p, VT_p_states[VT_state])
            return Ip - Vo / R
        try:
            Vout.append(brentq(f, 1e-9, VDD - 1e-9, xtol=1e-9))
        except ValueError:
            Ip0 = pfet_current(v, VDD, 0.0, k_p, VT_p_states[VT_state])
            Vout.append(0.0 if Ip0 < 0.5 * VDD / R else VDD)
    return Vin, np.array(Vout)


def butterfly_snm(inv_a, inv_b, n=301):
    """Static noise margin of the cross-coupled pair (largest square)."""
    Vin, Va = inv_a.vtc(n)
    _, Vb = inv_b.vtc(n)
    # curve 1: (Vin, Va); curve 2 mirrored: (Vb, Vin)
    # SNM via maximal square between curve1 and mirrored curve2
    from scipy.interpolate import interp1d
    f1 = interp1d(Vin, Va, bounds_error=False, fill_value=(Va[0], Va[-1]))
    f2i = interp1d(Vb[::-1], Vin[::-1], bounds_error=False,
                   fill_value=(Vin[-1], Vin[0]))
    # diagonal sweep method
    u = np.linspace(0, inv_a.VDD, n)
    d1 = f1(u) - u    # distance of curve 1 above diagonal
    d2 = f2i(u) - u
    snm_high = np.max((f1(u) - f2i(u))[f1(u) > f2i(u)] if np.any(f1(u) > f2i(u)) else [0]) / np.sqrt(2)
    snm_low = np.max((f2i(u) - f1(u))[f2i(u) > f1(u)] if np.any(f2i(u) > f1(u)) else [0]) / np.sqrt(2)
    return min(snm_high, snm_low)


if __name__ == "__main__":
    for eps in [0.0, -0.5, -1.0]:
        inv = NVInverter(eps_pct=eps)
        tp, Ed = inv.delay_energy()
        Pn, Pp = inv.static_power()
        print(f"eps={eps:+.1f}%: k_p={inv.k_p*1e6:.2f} uA/V^2  "
              f"tpLH={tp*1e12:.1f} ps  E_dyn={Ed*1e15:.1f} fJ  "
              f"P_static={max(Pn, Pp)*1e12:.3f} pW")
    inv = NVInverter()
    snm = butterfly_snm(NVInverter(state=1), NVInverter(state=1))
    print(f"SNM (programmed pair): {snm:.3f} V")
    # resistor reference static power when output low
    print(f"Resistor-load static power (output low): {P.VDD**2/10e6*1e6:.2f} uW")
