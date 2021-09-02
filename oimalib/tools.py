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
    return medians


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
        res = boxed_flux / r_med
    return res, boxed_wave


def hide_xlabel():
    frame1 = plt.gca()
    frame1.axes.xaxis.set_ticklabels([])
    plt.grid(lw=0.5, alpha=0.5)


def plot_vline(x, color="#eab15d"):
    plt.axvline(x, lw=1, color=color, zorder=-1, alpha=0.5)
