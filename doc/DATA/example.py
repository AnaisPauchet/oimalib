import numpy as np
import oimalib
from matplotlib import pyplot as plt

oifile = "doc/DATA/example_GRAVITY_Binary_s3_pa45.oifits"

data = [oimalib.load(oifile, cam="SC", simu=True)]

param = {"model": "binary", "x0": 0, "y0": 0, "sep": 3, "pa": 45, "dm": 3}

fitOnly = ["dm", "sep", "pa"]

fit = oimalib.smartfit(data, param, fitOnly=fitOnly)

obs = np.concatenate([oimalib.format_obs(x) for x in data])

prior = {"dm": [0, 6], "sep": [0, 10], "pa": [0, 90]}

sampler = oimalib.mcmcfit(
    data,
    param,
    nwalkers=10,
    niter=1000,
    prior=prior,
    fitOnly=fitOnly,
    threads=8,
    fast=True,
)

oimalib.plot_mcmc_results(sampler, labels=fitOnly)

mod_v2, mod_cp = oimalib.compute_geom_model(data, fit["best"])

oimalib.plot_oidata(data)

oimalib.plot_uv(data)

oimalib.plot_residuals(data, param)

plt.show(block=True)
