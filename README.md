# CIPS/WSe2 p-Type FeFET: Multiscale Simulation Code Base

Code accompanying the article "Channel decoupling of the memory window in
CuInP2S6-gated two-dimensional ferroelectric transistors: an exact law and
its consequences for interface processing".

Repository:
https://github.com/Tanvir-Mahmud-Mahim/Process-induced-compressive-strain-with-a-van-der-Waals-ferroelectric-gate-in-a-single-device

Archived release (code, benchmarks, and figure data):
https://doi.org/10.5281/zenodo.22084359

## Requirements

Python 3.10+ with numpy, scipy, matplotlib:

    pip install numpy scipy matplotlib

## Reproduce every figure and number

    python3 run_all.py

This regenerates all main-text figures (fig1 to fig7), all supplementary
figures (figS1 to figS7) in ../figures/, and results.json with the
headline numbers quoted in the article. Runtime is about 70 seconds on a
laptop. All random seeds are fixed, so results are bit-for-bit
reproducible.

## Modules

| File         | Contents                                                        |
|--------------|-----------------------------------------------------------------|
| params.py    | All physical constants and material parameters, with sources    |
| mobility.py  | Two-valley Boltzmann transport of strained monolayer WSe2       |
| ferro.py     | Multidomain Preisach + Landau-Khalatnikov model of CIPS         |
| fefet.py     | Self-consistent MFMIS FeFET electrostatics and transfer sweeps  |
| circuit.py   | Compact models, nonvolatile inverters, SNM, latch transients    |
| run_all.py   | Driver that produces every figure and results.json              |

Each physics module also runs standalone and prints its own validation
summary, e.g. `python3 mobility.py` prints the strain-enhancement
calibration against the published full-band data.

## Two notes on numerical practice

**Channel electrostatics uses both hole valleys.** `fefet.p_of_psi`
fills the light K valley and the heavy Gamma valley with Fermi-Dirac
statistics, using the same strain-dependent Gamma-K separation as the
transport model. The Gamma valley carries about three times the density
of states of K, so its strain-driven population changes the channel
quantum capacitance by more than an order of magnitude. Setting
`fefet.TWO_VALLEY_ES = False` recovers the earlier K-only electrostatics
for regression testing; under compressive strain the two agree to better
than 0.01 percent, because the Gamma valley is depopulated there.

**Memory windows are ensemble averages, not single draws.** The
ferroelectric is a finite sample from a Gaussian distribution of
coercive fields, so one draw is a random variable. At 400 hysterons the
window scatters by about 3 percent (standard deviation) from seed to
seed and its mean is 9.5 percent below the converged value; at 6400
hysterons the scatter is 0.5 percent. Use
`fefet.memory_window_ensemble()`, which averages over independent seeds
and returns the mean and standard deviation. Reporting a single draw at
a few hundred hysterons will not reproduce.

## Parameter provenance

- WSe2 band parameters: C2DB open database (https://c2db.fysik.dtu.dk)
- Gamma-K separation (157 meV) and its biaxial strain gauge
  (341 meV per percent): Afrid et al., npj 2D Mater. Appl. 10, 57 (2026),
  DOI 10.1038/s41699-026-00689-y
- Strain gauges and contact resistance: Zhao et al., ACS Nano 20, 18252
  (2026), DOI 10.1021/acsnano.6c03313
- CIPS parameters: Lee et al., ACS Nano 20, 16203 (2026), DOI
  10.1021/acsnano.6c02883; Liu et al., Nat. Commun. 7, 12357 (2016),
  DOI 10.1038/ncomms12357
- Dielectric constants: Laturia et al., npj 2D Mater. Appl. 2, 6 (2018),
  DOI 10.1038/s41699-018-0050-x

Calibrated quantities (D_iv, sigma_iv, screening factor) are documented
in Supplementary Section S2. No parameter is fitted to the results
reported in the article.
