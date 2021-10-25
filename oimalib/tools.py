#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug  7 16:31:48 2019

@author: asoulain
"""

import math

import numpy as np
from astropy import constants as cs
from bisect import bisect_left, insort
from collections import deque
from itertools import islice
from matplotlib import pyplot as plt
from scipy.constants.constants import R
from uncertainties import umath


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


def rad2arcsec(rad):
    arcsec = rad / (1 / mas2rad(1000))
    return arcsec


def incl2elong(incl):
    elong = 1.0 / umath.cos(incl * np.pi / 180.0)
    try:
        print("elong = %2.3f +/- %2.3f" % (elong.nominal_value, elong.std_dev))
    except AttributeError:
        print("elong = %2.2f" % elong)
    return elong


def check_hour_obs(list_data):
    mjd0 = list_data[0].info.mjd
    l_hour = []
    for d in list_data:
        l_hour.append(((d.info.mjd - mjd0) * 24))
    l_hour = np.array(l_hour)
    print(l_hour)
    return l_hour


def round_sci_digit(number):
    """ Rounds a float number with a significant digit number. """
    ff = str(number).split(".")[0]
    d = str(number).split(".")[1]
    d, ff = math.modf(number)
    sig_digit = 1
    if ff == 0:
        res = str(d).split(".")[1]
        for i in range(len(res)):
            if float(res[i]) != 0.0:
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
    P = (4 * np.pi ** 2) * sigma * T ** 4

    B = (
        (2 * h * c ** 2 * wl ** -5) / (np.exp(h * c / (wl * k * T)) - 1)
    ) / 1e6  # W/m2/micron
    if norm:
        res = B / P  # kW/m2/sr/m
    else:
        res = B
    return res


def _running_median(seq, M):
    """
     Purpose: Find the median for the points in a sliding window (odd number in size)
              as it is moved from left to right by one point at a time.
      Inputs:
            seq -- list containing items for which a running median (in a sliding window)
                   is to be calculated
              M -- number of items in window (window size) -- must be an integer > 1
      Otputs:
         medians -- list of medians with size N - M + 1
       Note:
         1. The median of a finite list of numbers is the "center" value when this list
            is sorted in ascending order.
         2. If M is an even number the two elements in the window that
            are close to the center are averaged to give the median (this
            is not by definition)
    """
    seq = iter(seq)
    s = []
    m = M // 2

    # Set up list s (to be sorted) and load deque with first window of seq
    s = [item for item in islice(seq, M)]
    d = deque(s)

    # Simple lambda function to handle even/odd window sizes
    def median():
        return s[m] if bool(M & 1) else (s[m - 1] + s[m]) * 0.5

    # Sort it in increasing order and extract the median ("center" of the sorted window)
    s.sort()
    medians = [median()]

    # Now slide the window by one point to the right for each new position (each pass through
    # the loop). Stop when the item in the right end of the deque contains the last item in seq
    for item in seq:
        old = d.popleft()  # pop oldest from left
        d.append(item)  # push newest in from right
        # locate insertion point and then remove old
        del s[bisect_left(s, old)]
        # insert newest such that new sort is not required
        insort(s, item)
        medians.append(median())
    return np.array(medians)


def substract_run_med(spectrum, wave=None, n_box=50, shift_wl=0, div=False):
    """ Substract running median from a raw spectrum `f`. The median
    is computed at each points from n_box/2 to -n_box/2+1 in a
    'box' of size `n_box`. The Br gamma line in vaccum and telluric
    lines can be displayed if wavelengths table (`wave`) is specified.
    `shift_wl` can be used to shift wave table to estimate the
    spectral shift w.r.t. the telluric lines.
    """
    r_med = _running_median(spectrum, n_box)
    boxed_flux = spectrum[n_box // 2 : -n_box // 2 + 1]

    boxed_wave = np.arange(len(boxed_flux))
    if wave is not None:
        boxed_wave = wave[n_box // 2 : -n_box // 2 + 1] - shift_wl

    res = boxed_flux - r_med
    if div:
        r_med[r_med == 0] = np.nan
        res = boxed_flux / r_med
    return res, boxed_wave


def hide_xlabel():
    plt.xticks(color="None")
    plt.grid(lw=0.5, alpha=0.5)


def plot_vline(x, color="#eab15d"):
    plt.axvline(x, lw=1, color=color, zorder=-1, alpha=0.5)


def wtmn(values, weights, axis=0, cons=False):
    """
    Return the weighted average and standard deviation.

    values, weights -- Numpy ndarrays with the same shape.
    """
    mn = np.average(values, weights=weights, axis=axis)
    
    # Fast and numerically precise:
    variance = np.average((values - mn) ** 2, weights=weights, axis=axis)
    
    std = np.sqrt(variance)
    if not cons:
        std_unbias = std / np.sqrt(len(weights))
    else:
        std_unbias = std
    return (mn, std_unbias)


def binning_tab(data, nbox=50, force=False, rel_err=0.01):
    """ Compute spectrally binned observables using weigthed averages (based
    on squared uncertainties). 
    
    Parameters:
    -----------
    
    `data` {class}:
        Data class (see oimalib.load() for details),\n
    `nbox` {int}:
        Size of the box,\n
    `flag` {bool}:
        If True, obs flag are used and avoided by the average,\n
    `force` {bool}:
        If True, force the uncertainties as the relative error `rel_err`,\n
    `rel_err` {float}:
        If `force`, relative uncertainties to be used [%].
        
    Outputs:
    --------
    `l_wl`, `l_vis2`, `l_e_vis2`, `l_cp`, `l_e_cp` {array}:
        Wavelengths, squared visibilities, V2 errors, closure phases and CP errors.
    """
    vis2 = data.vis2
    e_vis2 = data.e_vis2
    flag_vis2 = data.flag_vis2
    flag_cp = data.flag_cp
    cp = data.cp
    e_cp = data.e_cp
    wave = data.wl
    dvis = data.dvis
    e_dvis = data.e_dvis
    dphi = data.dphi
    e_dphi = data.e_dphi

    nwl = len(wave)
    l_wl, l_vis2, l_cp = [], [], []
    l_dvis, l_dphi, l_e_dvis, l_e_dphi = [], [], [], []
    l_e_cp, l_e_vis2 = [], []
    for i in np.arange(0, nwl, nbox):
        try:
            ind = i + nbox
            i_wl = np.mean(wave[i:ind])
            i_vis2, i_e_vis2 = [], []
            i_cp, i_e_cp = [], []
            i_dvis, i_e_dvis = [], []
            i_dphi, i_e_dphi = [], []
            for j in range(len(vis2)):
                range_vis2 = vis2[j][i:ind]
                range_e_vis2 = e_vis2[j][i:ind]
                range_dvis = dvis[j][i:ind]
                range_e_dvis = e_dvis[j][i:ind]
                range_dphi = dphi[j][i:ind]
                range_e_dphi = e_dphi[j][i:ind]
                cond_flag_vis2 = ~flag_vis2[j][i:ind]
                if len(range_vis2[cond_flag_vis2]) != 0:
                    weigths = 1.0 / range_e_vis2[cond_flag_vis2] ** 2
                    weigth_dvis = 1.0 / range_e_dvis[cond_flag_vis2] ** 2
                    weigth_dphi = 1.0 / range_e_dphi[cond_flag_vis2] ** 2
                    vis2_med, e_vis2_med = wtmn(range_vis2[cond_flag_vis2], weigths)
                    dvis_med, e_dvis_med = wtmn(range_dvis[cond_flag_vis2], weigth_dvis)
                    dphi_med, e_dphi_med = wtmn(range_dphi[cond_flag_vis2], weigth_dphi)
                else:
                    vis2_med, e_vis2_med = np.nan, np.nan
                    dvis_med, e_dvis_med = np.nan, np.nan
                    dphi_med, e_dphi_med = np.nan, np.nan

                if force:
                    e_vis2_med = rel_err * vis2_med

                i_vis2.append(vis2_med)
                i_e_vis2.append(e_vis2_med)
                i_dvis.append(dvis_med)
                i_e_dvis.append(e_dvis_med)
                i_dphi.append(dphi_med)
                i_e_dphi.append(e_dphi_med)

            for k in range(len(cp)):
                range_cp = cp[k][i:ind]
                range_e_cp = e_cp[k][i:ind]
                cond_flag_cp = ~flag_cp[k][i:ind]
                if len(range_cp[cond_flag_cp]) != 0:
                    weigths_cp = 1.0 / range_e_cp[cond_flag_cp] ** 2
                    cp_med, e_cp_med = wtmn(range_cp[cond_flag_cp], weigths_cp)
                else:
                    cp_med, e_cp_med = np.nan, np.nan
                i_cp.append(cp_med)
                i_e_cp.append(e_cp_med)

            if (np.mean(i_e_vis2) != 0.0) and (len(wave[i:ind]) == int(nbox)):
                l_wl.append(i_wl)
                l_vis2.append(i_vis2)
                l_cp.append(i_cp)
                l_e_cp.append(i_e_cp)
                l_e_vis2.append(i_e_vis2)
                l_dvis.append(i_dvis)
                l_dphi.append(i_dphi)
                l_e_dvis.append(i_e_dvis)
                l_e_dphi.append(i_e_dphi)
        except (IndexError, ZeroDivisionError):
            pass

    l_vis2 = np.array(l_vis2).T
    l_e_vis2 = np.array(l_e_vis2).T
    l_cp = np.array(l_cp).T
    l_e_cp = np.array(l_e_cp).T
    l_wl = np.array(l_wl)
    l_dvis = np.array(l_dvis).T
    l_dphi = np.array(l_dphi).T
    l_e_dvis = np.array(l_e_dvis).T
    l_e_dphi = np.array(l_e_dphi).T
    return l_wl, l_vis2, l_e_vis2, l_cp, l_e_cp, l_dvis, l_dphi, l_e_dvis, l_e_dphi

