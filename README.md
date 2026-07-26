# Strain-Augmented CIPS/WSe2 p-Type FeFET: Simulation Code Base

Code accompanying the article "Combining Strain and Ferroelectricity in
WSe2 Transistors for Fast, Low-Power Nonvolatile Logic: A Multiscale
Simulation Study".

Repository:
https://github.com/Tanvir-Mahmud-Mahim/Process-induced-compressive-strain-with-a-van-der-Waals-ferroelectric-gate-in-a-single-device

## Requirements

Python 3.10+ with numpy, scipy, matplotlib:

    pip install numpy scipy matplotlib

## Reproduce every figure and number

    python3 run_all.py

This regenerates all main-text figures (fig1 to fig5), all supplementary
figures (figS1 to figS5) in ../figures/, and results.json with the headline
numbers quoted in the article. Runtime is about 30 s on a laptop. All random
seeds are fixed, so results are bit-for-bit reproducible.

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

## Parameter provenance

- WSe2 band parameters: C2DB open database (https://c2db.fysik.dtu.dk)
- Strain gauges: Zhao et al., ACS Nano 20, 18252 (2026), DOI
  10.1021/acsnano.6c03313; Afrid et al., npj 2D Mater. Appl. 10, 57 (2026),
  DOI 10.1038/s41699-026-00689-y
- CIPS parameters: Lee et al., ACS Nano 20, 16203 (2026), DOI
  10.1021/acsnano.6c02883; Liu et al., Nat. Commun. 7, 12357 (2016), DOI
  10.1038/ncomms12357

Calibrated quantities (D_iv, sigma_iv, screening factor) are documented in
Supporting Information Section S2.
