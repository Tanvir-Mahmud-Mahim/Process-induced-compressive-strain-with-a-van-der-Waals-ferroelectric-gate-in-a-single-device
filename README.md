# CIPS/WSe2 p-Type FeFET: Multiscale Simulation Code Base

Code accompanying the article "Channel decoupling of the memory window in
CuInP2S6-gated two-dimensional ferroelectric transistors: conditions,
limits, and implications for interface processing" (revised version).

Repository:
https://github.com/Tanvir-Mahmud-Mahim/Process-induced-compressive-strain-with-a-van-der-Waals-ferroelectric-gate-in-a-single-device

Archived release (code, benchmarks, and figure data):
https://doi.org/10.5281/zenodo.22084359

## Requirements

Python 3.10+ with numpy, scipy, matplotlib:

    pip install numpy scipy matplotlib

## Reproduce every figure and number

    python3 run_all.py                 # main study, about 4 minutes
    python3 revision_studies.py        # scope and robustness studies, about 4 minutes
    python3 revision_figures.py        # fig8_scope and figS8_extensions
    python3 s10_expt_geometry.py       # stacks of fabricated devices, about 1 minute
    python3 s11_kinetic_convergence.py # kinetic-trap convergence checks

These write all figures to ../figures/ and the numbers quoted in the
article to results.json, revision_results.json, s10_results.json and
s11_results.json. All random seeds are fixed, so results are
reproducible bit for bit on the same platform.

## Modules

| File                       | Contents                                                        |
|----------------------------|-----------------------------------------------------------------|
| params.py                  | Physical constants and parameters, each labeled by source type  |
| mobility.py                | Two-valley Boltzmann transport of strained monolayer WSe2       |
| ferro.py                   | Multidomain Preisach + Landau-Khalatnikov model of CIPS         |
| fefet.py                   | Self-consistent MFMIS FeFET electrostatics and transfer sweeps  |
| fefet_ext.py               | Generalized MFMIS solver (area ratio, ferroelectric-side charge, back gate, ambipolar channel, branch-dependent or density-dependent mobility, scaled density of states) |
| circuit.py                 | Compact models, nonvolatile inverters, SNM, latch transients    |
| run_all.py                 | Main driver: figures 1 to 8 and S1 to S7, results.json          |
| revision_studies.py        | Studies S0 to S9 of the revision, revision_results.json         |
| revision_figures.py        | Figures fig8_scope and figS8_extensions                         |
| s10_expt_geometry.py       | Windows for the stacks of fabricated CIPS MFMIS devices         |
| s11_kinetic_convergence.py | Convergence of the finite-rate trap model                       |

## Notes on numerical practice

**Memory windows are ensemble averages.** The ferroelectric is a finite
sample from a Gaussian distribution of coercive fields, so one draw is a
random variable. At 400 hysterons the window scatters by about 3.6
percent from draw to draw and its mean is about 10 percent below the
converged value; at 6400 hysterons the scatter is 0.5 percent. Use
`fefet.memory_window_ensemble()`.

**Crossing-state interpolation.** The state at a constant-current
crossing is linearly interpolated between sweep points. Because hysterons
switch in discrete steps, the interpolated hole densities on the two
branches can differ on coarse grids, which leaves a small residual in the
window law. The law table uses a 1.9 mV grid (n = 6401).

**Kinetic trap sweeps need fine grids.** The finite-rate trap model is
converged only with about 2400 or more sweep points at high trap density;
run_all.py uses 4801.

## Parameter provenance

Each entry in params.py is labeled [DB] database, [DFT] published
first-principles result, [EXP] published measurement, [CAL] calibrated,
or [ASM] assumed. Main sources:

- C2DB (entry 1WSe2-1): K-valley hole mass 0.40 m0, lattice constant.
- Afrid et al., npj 2D Mater. Appl. 10, 57 (2026), DOI
  10.1038/s41699-026-00689-y: Gamma-K separation 157 meV at zero strain,
  increasing by 184 meV per percent of compression; extrinsic mobility
  25 cm^2/Vs and ratio 2.37 at -1 percent (calibration targets).
- Zhao et al., ACS Nano 20, 18252 (2026), DOI 10.1021/acsnano.6c03313:
  demonstrated compression of -0.22 percent and measured mobility factor.
- Lee et al., ACS Nano 20, 16203 (2026), DOI 10.1021/acsnano.6c02883,
  Supporting Information: CIPS P-V loop (Pr, Ec), MFMIS dimensions,
  60 us switching time, 5 MOhm inverter load.
- Laturia et al., npj 2D Mater. Appl. 2, 6 (2018) and Author Correction
  4, 28 (2020): h-BN (3.76) and WSe2 (15.6) static permittivities.

Assumed values (Gamma-valley mass, background permittivity, coercive-field
spread, contact resistance, K-edge shift, several phonon parameters) are
varied in revision_studies.py (study S8).

## Changes in version 2 (revision)

- Gamma-K strain rate corrected from 341 to 184 meV per percent (341 meV
  is the separation at -1 percent); K mass 0.36 to 0.40 m0 (C2DB); h-BN
  permittivity 3.5 to 3.76; transport model recalibrated (D_iv 2.1e11
  eV/m, sigma_iv 50 meV, screening factor 0.86).
- Resistor-loaded reference stage 10 to 5 MOhm.
- Law table on a 1.9 mV grid; kinetic trap sweeps on 4801 points.
- New generalized solver and revision studies.
