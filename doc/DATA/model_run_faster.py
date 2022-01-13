from glob import glob

import numpy as np
import oimalib
from oimalib.fitting import model_standard, select_model, smartfit2
from oimalib.modelling import comput_CP
from matplotlib import pyplot as plt

datadir = "/Users/soulaina/Desktop/data_binary_splited/"
l_file = glob(datadir + "*.fits")

d = [oimalib.load(x, cam="SC", simu=True) for x in l_file]

use_flag = True
obs = np.concatenate([oimalib.format_obs(x, use_flag=use_flag) for x in d])

fitted = ["V2", "CP"]

new_obs = []
for o in obs:
    if o[1] in fitted:
        new_obs.append(o)


param = {"model": "binary", "x0": 0, "y0": 0, "sep": 3, "pa": 45, "dm": 3}

import time


def model_standard_fast(d, param):
    l_mod_cp, l_mod_cvis = [], []
    for data in d:
        nbl = len(data.u)
        ncp = len(data.cp)
        model_target = select_model(param["model"])
        mod_cvis = np.zeros_like(data.vis2, dtype=complex)
        for i in range(nbl):
            flag = [False] * len(data.flag_dvis[i])
            if use_flag:
                flag = data.flag_dvis[i]
            u, v, wl = data.u[i], data.v[i], data.wl[~flag]
            mod_cvis[i] = model_target(u, v, wl, param)
        mod_cp = np.zeros_like(data.cp)
        for i in range(ncp):
            u1, u2, u3 = data.u1[i], data.u2[i], data.u3[i]
            v1, v2, v3 = data.v1[i], data.v2[i], data.v3[i]
            wl2 = data.wl
            X = [u1, u2, u3, v1, v2, v3, wl2]
            tmp = comput_CP(X, param, model_target)
            mod_cp[i, :] = tmp
        l_mod_cp.append(mod_cp)
        l_mod_cvis.append(mod_cvis)
    l_mod_cp = np.array(l_mod_cp)
    l_mod_cvis = np.array(l_mod_cvis)

    model_fast = []
    for i in range(len(l_mod_cvis)):
        if "V2" in fitted:
            model_fast += list(np.abs(l_mod_cvis[i].flatten()) ** 2)
        if "V" in fitted:
            model_fast += np.abs(l_mod_cvis[i].flatten())
        if "phi" in fitted:
            model_fast += np.angle(l_mod_cvis[i].flatten(), deg=True)
        if "CP":
            model_fast += list(l_mod_cp[i].flatten())
    model_fast = np.array(model_fast)
    return model_fast


start_time = time.time()
model_fast = model_standard_fast(d, param)
new_time = time.time() - start_time

start_time = time.time()
model_loop = model_standard(new_obs, param)
old_time = time.time() - start_time


plt.figure()
plt.plot(model_fast)
plt.plot(model_loop, "--")

print("Number of data points = %i" % (len(model_fast)))
print("New loop t = %2.3f s" % (new_time))

print("Number of data points = %i" % (len(model_loop)))
print("Old loop t = %2.3f s" % (old_time))
print("Improvement factor %i" % (old_time / new_time))

plt.show()
