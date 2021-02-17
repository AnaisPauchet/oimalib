# -*- coding: utf-8 -*-
"""
Created on Wed Nov  4 13:16:58 2015

@author: asoulain
"""

import numpy as np


def shiftFourier(Utable, Vtable, l, C_in, x0, y0):
    """ Shift the image (apply a phasor in Fourier space."""
    u = Utable / l
    v = Vtable / l
    C_out = C_in * np.exp(-2j*np.pi*(u*x0+v*y0))
    return C_out


def UVGrid(bmax, npts):
    """ Construct the grid in (u-v) plan (Fourier space)"""
    if type(npts) == int:
        N = npts
    else:
        N = 100

    x = np.linspace(-bmax, bmax, N)
    y = np.linspace(-bmax, bmax, N)
    Utable, vv = np.meshgrid(x, y)
    Vtable = np.transpose(Utable)

    uTab = np.reshape(Utable, [1, N*N])
    vTab = np.reshape(Vtable, [1, N*N])

    UVtable = np.zeros((N*N, 2))
    UVtable[:, 0] = uTab
    UVtable[:, 1] = vTab
    return UVtable


def UVLine(basemin, basemax, angle, npts):
    """ Select an line in the (u-v) plan"""

    UVtable = np.zeros([npts, 2])

    x = np.linspace(basemin, basemax, npts)
    UVtable[:, 0] = x*np.sin(angle)
    UVtable[:, 1] = x*np.cos(angle)

    return UVtable
