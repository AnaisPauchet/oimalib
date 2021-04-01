# OIMALIB

(**O**ptical **I**nterferometry **M**odelisation and **A**nalysis **LIB**rary)

[![version](http://img.shields.io/badge/OIMALIB-v0.1dev-orange.svg?style=flat)](https://github.com/DrSoulain/oimalib/)
<!-- 
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 10 15:30:57 2019

@author: asoulain

Different useful function to load, display and analyze interferometrical data.

Example of use to fit data: 
---------------------------

datadir = "Path of the folder containing dataset"

listfile = glob(datadir + '*.fits')

data = OiFile2Class(file, cam='SC') # Use a single file of data.
#/ or / 
data = dir2data(datadir) # Use a all files in the folder datadir

use_flag = False # Don't use flagged data
cond_wl = False # Fit only a spectral range
cond_uncer = False # Fit only a good snr data
wl_min, wl_max, rel_max = 2.1, 2.3, 3 #(band inf, band sup, snr min)

plotUV([data1, data2], use_flag=use_flag, cond_wl=cond_wl, wl_min=wl_min, wl_max=wl_max, cond_uncer=cond_uncer, rel_max=rel_max)

Obs1 = OiClass2Obs(data1, use_flag=use_flag, cond_wl=cond_wl, wl_min=wl_min, wl_max=wl_max, cond_uncer=cond_uncer, rel_max=rel_max)
Obs2 = OiClass2Obs(data2, use_flag=use_flag, cond_wl=cond_wl, wl_min=wl_min, wl_max=wl_max, cond_uncer=cond_uncer, rel_max=rel_max)
Obs = np.concatenate([Obs1, Obs2])

N_CP = len(Obs[(Obs[:,1] == 'CP')])
N_V2 = len(Obs[(Obs[:,1] == 'V2')])
N = len(Obs)

paramM = {'diam' : 1,
            'model' : 'disk',
            'x0' : 0,
            'y0' : 0}
#
fit = Smartfit(Obs, paramM, fitOnly=['diam'], multiproc=False)
#
plot_oidata([data1], model=True, param=fit['best'],
            use_flag=use_flag, fit=fit,
            cond_wl=cond_wl, wl_min=wl_min, wl_max=wl_max,
            cond_uncer=cond_uncer, rel_max=rel_max)

plt.show()
""" -->
