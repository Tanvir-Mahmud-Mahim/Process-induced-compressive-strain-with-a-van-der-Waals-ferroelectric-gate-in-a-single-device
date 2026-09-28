"""
s11_kinetic_convergence.py
Convergence checks for the finite-rate (SRH) trap model (revision study S11):
  (a) window at Dit = 3e12 cm^-2 eV^-1, 1 s double sweep, versus the number
      of sweep points;
  (b) retained on-current after a 100 us program pulse, 1e3 s hold,
      versus the number of steps in the program ramp.
Writes s11_results.json.
"""
import json
import fefet

out = {"sweep_points": [], "MW_V": [], "ramp_steps": [], "Ion_uA": {}}
for n in [301, 601, 1201, 2401, 4801]:
    d = fefet.FeFET(eps_pct=0.0, Dit_cm2=3e12, trap_mode="dynamic")
    s = d.sweep(x_max=6.0, n=n, t_total=1.0)
    mw = float(fefet.memory_window(*s[:4])[0])
    out["sweep_points"].append(n)
    out["MW_V"].append(mw)
    print(f"n = {n:5d}: MW = {mw:.4f} V", flush=True)
for Dit in [1e12, 3e12]:
    vals = []
    for nr in [40, 160, 640]:
        d = fefet.FeFET(eps_pct=0.0, n_dom=1600, Dit_cm2=Dit, trap_mode="dynamic")
        d.fe.reset(-1)
        d.reset_traps()
        d.program(-6.0, 0.0, t_pulse=1e-4, n_ramp=nr)
        d.program(+6.0, 0.0, t_pulse=1e-4, n_ramp=nr)
        ts, ps = d.hold(1e3, 0.0, n_sub=25)
        vals.append(float(d.drain_current(ps[-1]) * 1e6))
    out["Ion_uA"][f"{Dit:.0e}"] = vals
    print(f"Dit = {Dit:.0e}: I_on after 1e3 s = {vals} uA for ramps of 40/160/640", flush=True)
out["ramp_steps"] = [40, 160, 640]
json.dump(out, open("s11_results.json", "w"), indent=1)
