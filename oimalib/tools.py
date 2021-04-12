#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug  7 16:31:48 2019

@author: asoulain
"""

import math

import numpy as np
from astropy import constants as cs


def norm(tab):
    """Normalize the tab array by the maximum."""
    return tab / np.max(tab)


def rad2mas(rad):
    """Convert angle in radians to milli arc-sec."""
    mas = rad * (3600.0 * 180 / np.pi) * 10.0 ** 3
    return mas


def mas2rad(mas):
    """Convert angle in milli arc-sec to radians."""
    rad = mas * (10 ** (-3)) / (3600 * 180 / np.pi)
    return rad


def round_sci_digit(number):
    """ Rounds a float number with a significant digit number. """
    ff = str(number).split('.')[0]
    d = str(number).split('.')[1]
    d, ff = math.modf(number)
    sig_digit = 1
    if ff == 0:
        res = str(d).split('.')[1]
        for i in range(len(res)):
            if float(res[i]) != 0.:
                sig_digit = i + 1
                break
    else:
        sig_digit = 1

    return float(np.round(number, sig_digit)), sig_digit


def planck_law(T, wl, norm=False):
    h = cs.h.value
    c = cs.c.value
    k = cs.k_B.value
    sigma = cs.sigma_sb.value
    P = (4 * np.pi**2) * sigma * T**4

    B = ((2 * h * c**2 * wl**-5) /
         (np.exp(h * c / (wl * k * T)) - 1)) / 1e6  # W/m2/micron
    if norm:
        res = B / P  # kW/m2/sr/m
    else:
        res = B
    return res
