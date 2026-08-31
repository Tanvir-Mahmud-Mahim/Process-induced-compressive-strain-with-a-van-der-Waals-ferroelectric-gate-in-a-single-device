"""
ferro.py
CuInP2S6 (CIPS) ferroelectric model.

Two complementary descriptions are implemented:

1. Quasi-static multidomain (Preisach-type) model: an ensemble of square
   hysterons with a Gaussian distribution of coercive fields. This
   reproduces the measured saturated loop (Pr = 5.5 uC/cm^2) and gives
   realistic minor loops for the FeFET sweeps.

2. Landau-Khalatnikov (LK) dynamics for a single domain with distributed
   coercive fields, used for switching-time analysis:
       rho dP/dt = -(alpha P + beta P^3) + E(t)
   The kinetic coefficient rho is calibrated so that full polarization
   reversal at roughly 3x Ec completes in about 60 us, consistent with
   the measured programming time of CIPS MFMIS FeFETs (Lee 2026).
"""
import numpy as np
import params as P


class PreisachFE:
    """Ensemble of square hysterons with Gaussian coercive-field spread."""

    def __init__(self, n_dom=400, Ec=None, sigma_rel=None, Pr=None, seed=7):
        Ec = P.Ec_CIPS if Ec is None else Ec
        sigma_rel = P.sigma_Ec if sigma_rel is None else sigma_rel
        self.Pr = P.Pr_CIPS if Pr is None else Pr
        rng = np.random.default_rng(seed)
        self.Ecs = np.abs(rng.normal(Ec, sigma_rel * Ec, n_dom))
        self.Ecs = np.clip(self.Ecs, 0.15 * Ec, 3.0 * Ec)
        self.state = -np.ones(n_dom)  # start fully polarized "down"

    def reset(self, direction=-1):
        self.state[:] = direction

    def apply_field(self, E):
        """Update hysteron states for applied field E (V/m)."""
        self.state[E > self.Ecs] = 1.0
        self.state[E < -self.Ecs] = -1.0
        return self.P(E)

    def P(self, E):
        """Total polarization including linear background response."""
        return self.Pr * self.state.mean() + P.eps0 * (P.eps_FE_b - 1.0) * E

    def P_switch(self):
        """Switching part only."""
        return self.Pr * self.state.mean()


def pe_loop(E_max=3.0e7, n=1200, sigma_rel=None):
    """Quasi-static P-E loop."""
    fe = PreisachFE(sigma_rel=sigma_rel)
    E_up = np.linspace(-E_max, E_max, n)
    E_dn = np.linspace(E_max, -E_max, n)
    # pre-cycle to saturate
    for E in np.concatenate([E_up, E_dn, E_up]):
        fe.apply_field(E)
    P_dn = [fe.apply_field(E) for E in E_dn]
    P_up = [fe.apply_field(E) for E in E_up]
    return E_up, np.array(P_up), E_dn, np.array(P_dn)


def lk_switch(E_applied, t_end=1e-3, n_dom=60, n_t=4000):
    """LK switching transient for a step field E_applied starting from -Pr.

    Each domain i has Landau coefficients scaled to its own coercive field
    (alpha_i = alpha * (Ec_i/Ec), beta_i chosen to keep Pr fixed)."""
    rng = np.random.default_rng(11)
    Ecs = np.abs(rng.normal(P.Ec_CIPS, P.sigma_Ec * P.Ec_CIPS, n_dom))
    Ecs = np.clip(Ecs, 0.15 * P.Ec_CIPS, 3.0 * P.Ec_CIPS)
    t = np.logspace(-9, np.log10(t_end), n_t)
    dt = np.diff(t, prepend=t[0] * 0.5)
    Ptot = np.zeros_like(t)
    for Ec_i in Ecs:
        s = Ec_i / P.Ec_CIPS
        alpha_i = P.alpha_CIPS * s
        beta_i = P.beta_CIPS * s  # keeps Pr = sqrt(-alpha/beta) fixed
        Pd = -P.Pr_CIPS * 0.999
        for j in range(len(t)):
            dPdt = (-(alpha_i * Pd + beta_i * Pd ** 3) + E_applied) / P.rho_visc
            Pd = Pd + dPdt * dt[j]
            Pd = np.clip(Pd, -1.5 * P.Pr_CIPS, 1.5 * P.Pr_CIPS)
            Ptot[j] += Pd / n_dom
    return t, Ptot


def switching_time(E_applied, frac=0.9):
    """Time to reach frac * Pr starting from -Pr under field E_applied."""
    t, Ptot = lk_switch(E_applied, t_end=1e-1)
    idx = np.where(Ptot >= frac * P.Pr_CIPS)[0]
    return t[idx[0]] if len(idx) else np.inf


if __name__ == "__main__":
    E_up, P_up, E_dn, P_dn = pe_loop()
    print(f"Pr (loop) = {P_up[np.argmin(np.abs(E_up))] * 1e2:.2f} uC/cm^2 "
          f"(target {P.Pr_CIPS * 1e2:.2f})")
    for mult in [1.5, 2.0, 3.0, 5.0]:
        ts = switching_time(mult * P.Ec_CIPS)
        print(f"E = {mult:.1f} Ec  ->  t_sw = {ts * 1e6:9.2f} us")
