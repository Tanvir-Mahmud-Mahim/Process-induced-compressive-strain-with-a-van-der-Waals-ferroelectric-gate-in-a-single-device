"""
params.py
Central parameter file for the strain-augmented CIPS/WSe2 p-type FeFET study.

All parameters are anchored to published measurements or open databases:
- WSe2 band and valley parameters: C2DB open database (https://c2db.fysik.dtu.dk)
  and Afrid et al., npj 2D Mater. Appl. 10, 57 (2026).
- Strain gauge factors: Zhao et al., ACS Nano 20, 18252 (2026) and
  Afrid et al., npj 2D Mater. Appl. 10, 57 (2026).
- CIPS ferroelectric parameters: Lee et al., ACS Nano 20, 16203 (2026)
  (Pr = 5.5 uC/cm2, MFMIS stack thicknesses) and Liu et al.,
  Nat. Commun. 7, 12357 (2016).
"""
import numpy as np

# ----------------------------------------------------------------------
# Physical constants (SI)
# ----------------------------------------------------------------------
q     = 1.602176634e-19      # C
kB    = 1.380649e-23         # J/K
hbar  = 1.054571817e-34      # J s
m0    = 9.1093837015e-31     # kg
eps0  = 8.8541878128e-12     # F/m
eV    = q                    # J

T     = 300.0                # K
kT    = kB * T               # J
kT_eV = kT / q               # eV

# ----------------------------------------------------------------------
# Monolayer WSe2 valence band model (two-valley: K and Gamma)
# ----------------------------------------------------------------------
# Lattice constant (C2DB): a = 3.32 A
a_WSe2 = 3.32e-10            # m

# Effective masses (units of m0). K valley: light holes dominate transport.
# Gamma valley: heavy holes. Values follow DFT (C2DB / literature consensus).
mK_h   = 0.36                # K-valley hole mass
mG_h   = 2.20                # Gamma-valley hole mass (heavy)
gK     = 2                   # K/K' valley degeneracy
gG     = 1                   # Gamma valley degeneracy

# Valley energetics (hole picture: energies measured DOWN from K-VBM, in eV).
# Unstrained Gamma-K separation from full-band study: 157 meV (Gamma below K).
dE_GK0 = 0.157               # eV, E_K - E_Gamma at zero strain
# Biaxial strain gauge of the Gamma-K separation: 341 meV per % strain
# (tensile strain, eps > 0, raises Gamma toward K; compressive lowers it).
dE_GK_gauge = 0.341          # eV per % biaxial strain

# Slow shift of the K-VBM itself with strain (affects threshold voltage);
# compression pushes the K valley up slightly (increases p-doping).
dEK_gauge = -0.030           # eV per % strain (E_K rises under compression)

# Areal mass density of monolayer WSe2
rho_2D = 5.93e-6             # kg/m^2

# Acoustic phonon (LA) parameters
v_s    = 3.3e3               # m/s, sound velocity
D_ac_K = 2.3 * eV            # J, acoustic deformation potential, K valley
D_ac_G = 3.1 * eV            # J, acoustic deformation potential, Gamma valley

# Optical phonon (intravalley, zero order)
E_op   = 0.031 * eV          # J, ~31 meV homopolar phonon
D_op   = 4.6e10 * eV         # J/m (4.6e8 eV/cm), zero-order ODP

# Intervalley K <-> Gamma phonon
E_iv   = 0.027 * eV          # J, ~27 meV
# D_iv is calibrated so the model reproduces the published hole-mobility
# enhancement of monolayer WSe2 (mu/mu0 = 2.37 at -1% biaxial strain and
# mu0 ~ 25 cm^2/Vs under extrinsic conditions, Afrid 2026; gauge factor
# 340 +/- 95 %/% at -0.22% strain, Zhao 2026). See SI Section S2.
D_iv   = 1.5e11 * eV         # J/m (calibrated)

# Charged impurity scattering (Coulomb centers at the substrate interface)
n_imp  = 5.0e16              # m^-2 -> 5e12 cm^-2 (npj baseline)
eps_env = 0.5 * (3.9 + 1.0)  # average of SiO2 substrate and vacuum top

# Default carrier density for mobility evaluation
p_sheet0 = 1.0e17            # m^-2 -> 1e13 cm^-2 (npj baseline)

# ----------------------------------------------------------------------
# CuInP2S6 (CIPS) ferroelectric parameters
# ----------------------------------------------------------------------
Pr_CIPS   = 5.5e-2           # C/m^2  (5.5 uC/cm^2, measured PUND value)
Ec_CIPS   = 3.0e7            # V/m    (300 kV/cm; matches the 2-4 V
                             # switching onset over 86.5 nm CIPS, Lee 2026)
eps_FE_b  = 25.0             # background (non-switching) permittivity
sigma_Ec  = 0.22             # relative spread of coercive field (Preisach)
rho_visc  = 2.0e4            # Ohm m, LK kinetic coefficient (calibrated to
                             # ~60 us full switching at 1.5x Ec, cf. the
                             # measured 60 us programming pulse of Lee 2026)

# Landau coefficients for a second-order (alpha<0, beta>0) single well:
#   E = alpha*P + beta*P^3 ;  Pr = sqrt(-alpha/beta),
#   Ec = 2/(3*sqrt(3)) * |alpha|^{3/2} / beta^{1/2}
alpha_CIPS = -(3.0 * np.sqrt(3.0) / 2.0) * Ec_CIPS / Pr_CIPS
beta_CIPS  = -alpha_CIPS / Pr_CIPS ** 2

# ----------------------------------------------------------------------
# MFMIS FeFET stack (metal / CIPS / metal / h-BN / WSe2)
# ----------------------------------------------------------------------
t_FE_default = 30e-9         # m, CIPS thickness (design value; expt used 86.5 nm)
t_hBN        = 10e-9         # m, h-BN interlayer dielectric
eps_hBN      = 3.5           # out-of-plane permittivity of h-BN

# WSe2 2D density of states (K valley, both spins split off; use transport DOS)
# DOS_2D = g * m / (2 pi hbar^2)

# Channel geometry
W_ch = 4e-6                  # m
L_ch = 2e-6                  # m
VDS_read = -1.0              # V (p-FET read bias)

# Contact resistance per width (TOS-doped Pd contacts, Zhao 2026: 200-370 kOhm um)
Rc_W = 50e3 * 1e-6           # Ohm m  (50 kOhm um; TOS-doped contacts, Zhao 2026)

# Velocity saturation
v_sat = 3.0e6                # cm/s -> set in SI units in transport code (3e4 m/s)

# Load capacitance for transient circuit analysis
C_load = 1.0e-15             # F (1 fF, interconnect-dominated node)
VDD    = 3.0                 # V, supply for nonvolatile logic

# n-FET pull-down (WSe2 n-type FET, Lee 2026 latch style) simple model
mu_n   = 30e-4               # m^2/Vs (30 cm^2/Vs)
VT_n   = 0.6                 # V, normally-off n-FET
