"""
s10_expt_geometry.py
Windows computed for the stack dimensions of two fabricated CIPS MFMIS
FeFETs (revision study S10), with the CIPS parameters of this work.
  Wang et al. 2021: 87 nm CIPS, 4-9 nm h-BN, graphene floating gate,
                    MoS2 channel, 4 V sweep range (reported MW 3.8 V)
  Lee et al. 2026:  86.5 nm CIPS, 18.6 nm h-BN, metal middle gate,
                    MoTe2 channel, top-gate sweeps +-2.5 to +-4 V
The channel, area ratio and read criterion of those devices differ from
the model, so these are order-of-magnitude comparisons.
Writes s10_results.json.
"""
import json
import numpy as np
from fefet_ext import FeFETX, probe, law_terms

out = []
for label, tfe, thbn, xmax, ar in [
        ("Wang 2021, h-BN 4 nm, +-4 V", 87e-9, 4e-9, 4.0, 1.0),
        ("Wang 2021, h-BN 6.5 nm, +-4 V", 87e-9, 6.5e-9, 4.0, 1.0),
        ("Wang 2021, h-BN 9 nm, +-4 V", 87e-9, 9e-9, 4.0, 1.0),
        ("Wang 2021, h-BN 6.5 nm, +-4 V, A_R 0.5", 87e-9, 6.5e-9, 4.0, 0.5),
        ("Lee 2026, +-3.0 V", 86.5e-9, 18.6e-9, 3.0, 1.0),
        ("Lee 2026, +-3.5 V", 86.5e-9, 18.6e-9, 3.5, 1.0),
        ("Lee 2026, +-4.0 V", 86.5e-9, 18.6e-9, 4.0, 1.0),
        ("Lee 2026, +-4.0 V, A_R 0.5", 86.5e-9, 18.6e-9, 4.0, 0.5),
        ("Lee 2026, +-4.0 V, A_R 0.25", 86.5e-9, 18.6e-9, 4.0, 0.25)]:
    mws = []
    for seed in (1, 2, 3, 4):
        d = FeFETX(t_FE=tfe, t_hBN=thbn, A_R=ar, seed=seed)
        f, b, fw, bw = probe(d, x_max=xmax, n=401)
        mws.append(np.nan if (f is None or b is None) else law_terms(d, f, b)["mw_sim"])
    mws = np.array(mws, dtype=float)
    row = dict(case=label, t_FE_nm=tfe * 1e9, t_hBN_nm=thbn * 1e9,
               x_max_V=xmax, A_R=ar, MW_mean_V=float(np.nanmean(mws)),
               MW_sd_V=float(np.nanstd(mws)))
    out.append(row)
    print(f"{label:>34}: MW {row['MW_mean_V']:.3f} +- {row['MW_sd_V']:.3f} V", flush=True)
json.dump(out, open("s10_results.json", "w"), indent=1)
