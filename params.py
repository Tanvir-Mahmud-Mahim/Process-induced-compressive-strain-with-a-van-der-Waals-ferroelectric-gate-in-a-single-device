"""
params.py
Central parameter file for the strain-augmented CIPS/WSe2 p-type FeFET study.

Each parameter is labeled by the kind of source it comes from:
  [DB]   open database entry (C2DB, PBE level)
  [DFT]  published first-principles calculation
  [EXP]  published measurement (read from the cited figure where noted)
  [CAL]  calibrated once against published anchors (SI Section S2)
  [ASM]  assumed value, varied in the sensitivity study (revision_studies.py)
Sources:
- C2DB: Haastrup et al., 2D Mater. 5, 042002 (2018); Gjerding et al.,
  2D Mater. 8, 044002 (2021); entry 1WSe2-1 (2H WSe2 monolayer).
- Afrid et al., npj 2D Mater. Appl. 10, 57 (2026) (first-principles, no SOC).
- Zhao et al., ACS Nano 20, 18252 (2026) (strained WSe2 p-FETs).
- Lee et al., ACS Nano 20, 16203 (2026), Supporting Information
  (CIPS capacitor P-V loop and MFMIS stack dimensions).
- Laturia et al., npj 2D Mater. Appl. 2, 6 (2018) and Author Correction,
  npj 2D Mater. Appl. 4, 28 (2020) (first-principles permittivities).
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
# Lattice constant [DB]: a = 3.319 A (C2DB 1WSe2-1)
a_WSe2 = 3.32e-10            # m

# Effective masses (units of m0). K valley: light holes dominate transport.
# K: [DB] C2DB 1WSe2-1, VBM density-of-states mass 0.40 m0 (PBE).
# Gamma: [ASM] heavy-hole mass; no database entry, varied +-20 % in the
# sensitivity study (it does not enter the memory window).
mK_h   = 0.40                # K-valley hole mass
mG_h   = 2.20                # Gamma-valley hole mass (heavy)
gK     = 2                   # K/K' valley degeneracy
gG     = 1                   # Gamma valley degeneracy

# Valley energetics (hole picture: energies measured DOWN from K-VBM, in eV).
# [DFT] Afrid 2026: Gamma-K separation 157 meV at zero strain, rising to
# 341 meV at -1 % biaxial compression, i.e. an increase of 184 meV per %
# (their "largest increase in dE_GK (184 meV/%eps)"). The same linear rate
# is assumed on the tensile side (tension raises Gamma toward K).
dE_GK0 = 0.157               # eV, E_K - E_Gamma at zero strain
dE_GK_gauge = 0.184          # eV per % biaxial strain

# [ASM] Slow shift of the K-VBM itself with strain. Afrid 2026 report only
# that the K valley moves "marginally"; -30 meV/% is assumed. It shifts the
# flat-band voltage only and cancels from the memory window.
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
# [CAL] D_iv, together with the broadening and screening factor in
# mobility.py, is calibrated once so the model reproduces the extrinsic
# first-principles hole mobility of monolayer WSe2 (Afrid 2026, Fig. 5b:
# mu0 = 25 cm^2/Vs at p = 1e13 cm^-2, n_imp = 5e12 cm^-2, SiO2; and
# mu/mu0 = 2.37 at -1 % compression). See SI Section S2.
D_iv   = 2.1e11 * eV         # J/m (calibrated)

# Charged impurity scattering (Coulomb centers at the substrate interface)
n_imp  = 5.0e16              # m^-2 -> 5e12 cm^-2 (npj baseline)
eps_env = 0.5 * (3.9 + 1.0)  # average of SiO2 substrate and vacuum top

# Default carrier density for mobility evaluation
p_sheet0 = 1.0e17            # m^-2 -> 1e13 cm^-2 (npj baseline)

# ----------------------------------------------------------------------
# CuInP2S6 (CIPS) ferroelectric parameters
# ----------------------------------------------------------------------
# [EXP] Pr and Ec read from the P-V loop of a ~105 nm CIPS capacitor in
# the Supporting Information of Lee 2026 (Fig. S4e: remanent polarization
# about 5.4 uC/cm^2, zero crossings near -2.5 V and +3.5 V, i.e. about
# 300 kV/cm). Wang et al., Nat. Commun. 12, 1109 (2021) report ~300 kV/cm
# and ~4 uC/cm^2 on a 200 nm capacitor.
Pr_CIPS   = 5.5e-2           # C/m^2  (5.5 uC/cm^2)
Ec_CIPS   = 3.0e7            # V/m    (300 kV/cm)
eps_FE_b  = 25.0             # [ASM] background (non-switching) permittivity;
                             # the dielectric constant of ferroelectric CIPS
                             # is about 30 (Yan 2026); varied 12 to 40 in the
                             # sensitivity study
sigma_Ec  = 0.22             # [ASM] relative spread of coercive field
rho_visc  = 2.0e4            # [CAL] Ohm m, LK kinetic coefficient, set so
                             # that full switching at 1.5 Ec takes ~70 us, cf.
                             # the 60 us switching time reported by Lee 2026

# Landau coefficients for a second-order (alpha<0, beta>0) single well:
#   E = alpha*P + beta*P^3 ;  Pr = sqrt(-alpha/beta),
#   Ec = 2/(3*sqrt(3)) * |alpha|^{3/2} / beta^{1/2}
alpha_CIPS = -(3.0 * np.sqrt(3.0) / 2.0) * Ec_CIPS / Pr_CIPS
beta_CIPS  = -alpha_CIPS / Pr_CIPS ** 2

# ----------------------------------------------------------------------
# MFMIS FeFET stack (metal / CIPS / metal / h-BN / WSe2)
# ----------------------------------------------------------------------
t_FE_default = 30e-9         # m, CIPS thickness (design value; the MFMIS
                             # FeFET of Lee 2026 used ~86.5 nm CIPS on
                             # ~18.6 nm h-BN, their Fig. S2)
t_hBN        = 10e-9         # m, h-BN interlayer dielectric
eps_hBN      = 3.76          # [DFT] static out-of-plane permittivity of bulk
                             # h-BN (Laturia 2018, Table 1)

# WSe2 2D density of states (K valley, both spins split off; use transport DOS)
# DOS_2D = g * m / (2 pi hbar^2)

# Channel geometry
W_ch = 4e-6                  # m
L_ch = 2e-6                  # m
VDS_read = -1.0              # V (p-FET read bias)

# [ASM] Contact resistance per contact width. Zhao 2026 report total
# contact resistances 2Rc of 200-370 kOhm for their TOS-doped Pd contacts;
# 50 kOhm um is assumed here and varied from 5 to 370 kOhm um in the
# sensitivity study. It affects currents only, not the window.
Rc_W = 50e3 * 1e-6           # Ohm m  (50 kOhm um)

# Velocity saturation
v_sat = 3.0e6                # cm/s -> set in SI units in transport code (3e4 m/s)

# Load capacitance for transient circuit analysis
C_load = 1.0e-15             # F (1 fF, interconnect-dominated node)
VDD    = 3.0                 # V, supply for nonvolatile logic

# [ASM] generic n-FET pull-down for the complementary circuit projection
mu_n   = 30e-4               # m^2/Vs (30 cm^2/Vs)
VT_n   = 0.6                 # V, normally-off n-FET
