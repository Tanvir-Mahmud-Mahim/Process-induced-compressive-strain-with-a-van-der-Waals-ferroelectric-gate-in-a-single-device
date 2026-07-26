"""
mobility.py
Reduced-order two-valley (K, Gamma) Boltzmann transport model for hole
mobility in monolayer WSe2 under biaxial strain.

The model implements Fermi golden rule scattering rates for:
  - acoustic deformation potential (elastic, ADP)
  - zero-order optical deformation potential (ODP)
  - intervalley K <-> Gamma phonon scattering (IV)
  - screened charged impurity scattering (CI)
and evaluates the mobility with the Kubo-Greenwood integral per valley.

Strain enters through the Gamma-K valley separation
    dE_GK(eps) = dE_GK0 + dE_GK_gauge * eps   (eps in %, tensile positive)
following the first-principles gauge of Afrid et al. (npj 2DM Appl. 2026).
The intervalley deformation potential is calibrated once (at zero strain)
so that the model reproduces the published extrinsic mobility gauge
(~2.4 per % compressive strain) and baseline mobility (~25 cm^2/Vs at
p = 1e13 cm^-2, n_imp = 5e12 cm^-2, SiO2 environment).
"""
import numpy as np
from scipy.optimize import brentq
from scipy.special import erf
import params as P

# Final-state broadening of the intervalley threshold (represents phonon
# dispersion, non-parabolicity, and collision broadening that smear the
# sharp parabolic-band onset; calibrated in SI Section S2).
SIGMA_IV = 0.090 * 1.602176634e-19  # J (90 meV, calibrated)
# Screening reduction factor for charged-impurity scattering (accounts for
# incomplete free-carrier screening of interface Coulomb centers).
SCR_FACTOR = 0.8


def smooth_step(x, sigma):
    """Broadened unit step with exponential (logistic) tails, representing
    the quasi-exponential energy tail of the broadened final-state DOS."""
    return 1.0 / (1.0 + np.exp(-np.clip(x / sigma, -60, 60)))


def dos_2d(m_rel, g):
    """2D density of states (states / J m^2), spin included."""
    return g * m_rel * P.m0 / (2.0 * np.pi * P.hbar ** 2) * 2.0  # x2 spin


def valley_edges(eps_pct):
    """Hole-picture band edges (J), energy measured downward from K-VBM.

    Returns (E_K_edge, E_G_edge): K edge fixed at 0, Gamma edge at
    dE_GK(eps) > 0 below the K-VBM. Compression (eps<0) increases the
    separation, decoupling the valleys."""
    # Tensile strain (eps > 0) raises Gamma toward the K-VBM (smaller
    # separation, stronger IV scattering); compression lowers Gamma.
    dE = (P.dE_GK0 - P.dE_GK_gauge * eps_pct) * P.eV
    # dE may become negative under strong tensile strain, meaning the
    # Gamma valley rises above the K-VBM and heavy holes dominate.
    return 0.0, dE


def fermi_level(p_sheet, eps_pct):
    """Find hole Fermi level Ef (J, hole convention, measured from K-VBM
    downward: occupied hole states have E < Ef ... we use Fermi-Dirac in
    hole energies) for total sheet density p_sheet across both valleys."""
    EK, EG = valley_edges(eps_pct)
    NK = dos_2d(P.mK_h, P.gK)
    NG = dos_2d(P.mG_h, P.gG)

    def p_of_Ef(Ef):
        # 2D parabolic band filled with Fermi-Dirac (hole energies)
        pK = NK * P.kT * np.log1p(np.exp((Ef - EK) / P.kT))
        pG = NG * P.kT * np.log1p(np.exp((Ef - EG) / P.kT))
        return pK + pG

    lo, hi = -1.5 * P.eV, 1.5 * P.eV
    Ef = brentq(lambda x: p_of_Ef(x) - p_sheet, lo, hi, xtol=1e-30)
    pK = NK * P.kT * np.log1p(np.exp((Ef - EK) / P.kT))
    pG = NG * P.kT * np.log1p(np.exp((Ef - EG) / P.kT))
    return Ef, pK, pG


def bose(E_ph):
    return 1.0 / (np.expm1(E_ph / P.kT))


def rate_acoustic(E, m_rel, D_ac):
    """Elastic ADP rate, 2D (energy independent)."""
    m = m_rel * P.m0
    r = (m * D_ac ** 2 * P.kT) / (P.hbar ** 3 * P.rho_2D * P.v_s ** 2)
    return np.full_like(E, r)


def rate_optical(E, m_rel, D0, E_ph, E_edge=0.0):
    """Zero-order ODP rate with absorption + emission, intravalley."""
    m = m_rel * P.m0
    pref = (D0 ** 2 * m) / (2.0 * P.rho_2D * (E_ph / P.hbar) * P.hbar ** 2)
    N = bose(E_ph)
    absr = N * np.ones_like(E)
    emis = (N + 1.0) * ((E - E_edge) > E_ph)
    return pref * (absr + emis)


def rate_intervalley(E, m_final_rel, g_final, D_iv, E_ph, E_final_edge):
    """IV rate from a state at hole energy E (from K-VBM) into the final
    valley whose edge sits at E_final_edge (hole energies increase into
    the band)."""
    m_f = m_final_rel * P.m0
    pref = (D_iv ** 2 * m_f * g_final) / (2.0 * P.rho_2D * (E_ph / P.hbar) * P.hbar ** 2)
    N = bose(E_ph)
    r = np.zeros_like(E)
    # absorption: final energy E + E_ph must lie inside final valley
    r += N * smooth_step((E + E_ph) - E_final_edge, SIGMA_IV)
    # emission: final energy E - E_ph
    r += (N + 1.0) * smooth_step((E - E_ph) - E_final_edge, SIGMA_IV)
    return pref * r


def rate_impurity(E, m_rel, g, p_screen):
    """Screened charged-impurity scattering (2D, momentum relaxation)."""
    m = m_rel * P.m0
    E = np.maximum(E, 1e-4 * P.eV)
    k = np.sqrt(2.0 * m * E) / P.hbar
    eps_bg = 2.0 * P.eps0 * P.eps_env  # top+bottom half-space average
    # Thomas-Fermi screening from the K-valley 2D gas (dominant, degenerate)
    qTF = SCR_FACTOR * (P.q ** 2 * dos_2d(P.mK_h, P.gK)) / (2.0 * eps_bg)
    # partial screening factor for non-degenerate part
    theta = np.linspace(1e-4, np.pi, 181)
    out = np.zeros_like(E)
    for i, Ei in enumerate(E):
        ki = k[i]
        qq = 2.0 * ki * np.sin(theta / 2.0)
        V = P.q ** 2 / (eps_bg * (qq + qTF))
        integ = np.trapezoid(V ** 2 * (1.0 - np.cos(theta)), theta)
        out[i] = (P.n_imp * m) / (2.0 * np.pi * P.hbar ** 3) * integ
    return out


def valley_mobility(E_edge, m_rel, g, other_edge, m_other_rel, g_other,
                    D_ac, Ef, p_screen, D_iv=None):
    """Kubo-Greenwood mobility of one parabolic valley."""
    if D_iv is None:
        D_iv = P.D_iv
    m = m_rel * P.m0
    # hole kinetic energy grid inside this valley
    Ekin = np.linspace(1e-4 * P.eV, 0.6 * P.eV, 900)
    E_abs = E_edge + Ekin  # absolute hole energy from K-VBM

    r = rate_acoustic(Ekin, m_rel, D_ac)
    r = r + rate_optical(E_abs, m_rel, P.D_op, P.E_op, E_edge)
    r = r + rate_intervalley(E_abs, m_other_rel, g_other, D_iv, P.E_iv, other_edge)
    r = r + rate_impurity(Ekin, m_rel, g, p_screen)
    tau = 1.0 / r

    # -df/dE with hole Fermi level Ef (hole energies)
    x = (E_abs - Ef) / P.kT
    dfdE = np.exp(np.clip(x, -60, 60)) / (P.kT * (1.0 + np.exp(np.clip(x, -60, 60))) ** 2)
    f = 1.0 / (1.0 + np.exp(np.clip(x, -60, 60)))

    num = np.trapezoid(tau * Ekin * dfdE, Ekin)
    den = np.trapezoid(f, Ekin)
    if den <= 0:
        return 0.0
    return (P.q / m) * num / den


def hole_mobility(eps_pct, p_sheet=None, D_iv=None, return_parts=False):
    """Effective hole mobility (m^2/Vs) of monolayer WSe2 at strain eps (%)."""
    if p_sheet is None:
        p_sheet = P.p_sheet0
    EK, EG = valley_edges(eps_pct)
    Ef, pK, pG = fermi_level(p_sheet, eps_pct)

    muK = valley_mobility(EK, P.mK_h, P.gK, EG, P.mG_h, P.gG,
                          P.D_ac_K, Ef, p_sheet, D_iv)
    muG = valley_mobility(EG, P.mG_h, P.gG, EK, P.mK_h, P.gK,
                          P.D_ac_G, Ef, p_sheet, D_iv)
    mu = (pK * muK + pG * muG) / (pK + pG)
    if return_parts:
        return mu, muK, muG, pK, pG, Ef
    return mu


def scattering_spectrum(eps_pct, p_sheet=None):
    """Total K-valley scattering rate vs energy for plotting."""
    if p_sheet is None:
        p_sheet = P.p_sheet0
    EK, EG = valley_edges(eps_pct)
    Ekin = np.linspace(1e-4 * P.eV, 0.30 * P.eV, 400)
    E_abs = EK + Ekin
    r = rate_acoustic(Ekin, P.mK_h, P.D_ac_K)
    r = r + rate_optical(E_abs, P.mK_h, P.D_op, P.E_op, EK)
    r_iv = rate_intervalley(E_abs, P.mG_h, P.gG, P.D_iv, P.E_iv, EG)
    return Ekin / P.eV, r + r_iv, r, r_iv


if __name__ == "__main__":
    for eps in [0.0, -0.25, -0.5, -1.0, 0.5, 1.0]:
        mu = hole_mobility(eps) * 1e4
        print(f"eps = {eps:+.2f} %  ->  mu_h = {mu:8.2f} cm^2/Vs")
    mu0 = hole_mobility(0.0)
    g = (hole_mobility(-0.25) / mu0 - 1.0) / 0.25
    print(f"gauge factor at -0.25%: {g:.2f} per % strain")
