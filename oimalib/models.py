# -*- coding: utf-8 -*-
"""
Created on Wed Nov  4 13:14:23 2015

@author: asoulain
"""

import numpy as np
from matplotlib import pyplot as plt
from scipy import special
from termcolor import cprint

from .binary import getBinaryPos, kepler_solve
from .fourier import shiftFourier
from .tools import mas2rad, planck_law, rad2mas


def visPointSource(Utable, Vtable, Lambda, param):
    """
    Compute complex visibility of a point source.

    Params:
    -------
    x0, y0: {float}
        Shift along x and y position [rad].
    """
    C_centered = np.ones(np.size(Utable))
    C = shiftFourier(Utable, Vtable, Lambda, C_centered,
                     param["x0"], param["y0"])
    return C


def visBinary(Utable, Vtable, Lambda, param):
    sep = mas2rad(param["sep"])
    dm = param["dm"]
    theta = np.deg2rad(90-param["pa"])

    if dm < 0:
        return np.array([np.nan]*len(Lambda))
    f1 = 1
    f2 = f1 / (2.512 ** dm)
    ftot = f1 + f2

    rel_f1 = f1 / ftot
    rel_f2 = f2 / ftot

    p_s1 = {"x0": 0, "y0": 0}
    p_s2 = {"x0": sep * np.cos(theta), "y0": sep * np.sin(theta)}
    s1 = rel_f1 * visPointSource(Utable, Vtable, Lambda, p_s1)
    s2 = rel_f2 * visPointSource(Utable, Vtable, Lambda, p_s2)
    C_centered = s1 + s2
    return C_centered


def visResBinary(Utable, Vtable, Lambda, param):
    sep = mas2rad(param["sep"])
    dm = param["dm"]
    theta = np.deg2rad(90-param["pa"])
    diam = param['diam']
    if dm < 0:
        return np.array([np.nan]*len(Lambda))
    f1 = 1
    f2 = f1 / (10 ** (0.4*dm))
    ftot = f1 + f2

    rel_f1 = f1 / ftot
    rel_f2 = f2 / ftot

    p_s1 = {"x0": 0, "y0": 0, 'diam': diam}
    p_s2 = {"x0": sep * np.cos(theta), "y0": sep * np.sin(theta)}
    s1 = rel_f1 * visUniformDisk(Utable, Vtable, Lambda, p_s1)
    s2 = rel_f2 * visPointSource(Utable, Vtable, Lambda, p_s2)
    C_centered = s1 + s2
    return C_centered


def visUniformDisk(Utable, Vtable, Lambda, param):
    """
    Compute complex visibility of an uniform disk

    Params:
    -------
    diam: {float}
        Diameter of the disk [rad],\n
    x0, y0: {float}
        Shift along x and y position [rad].
    """
    u = Utable / Lambda
    v = Vtable / Lambda

    diam = mas2rad(param["diam"])

    r = np.sqrt(u ** 2 + v ** 2)

    C_centered = 2 * special.j1(np.pi * r * diam) / (np.pi * r * diam)
    C = shiftFourier(
        Utable, Vtable, Lambda, C_centered, mas2rad(
            param["x0"]), mas2rad(param["y0"])
    )
    return C


def visEllipticakUniformDisk(Utable, Vtable, Lambda, param):
    """
    Compute complex visibility of a elliplical uniform disk

    Params:
    -------
    majorAxis: {float}
        Major axis of the disk [rad],\n
    elong: {float}
        Elongation ratio (i.e: major = elong * minor),\n
    angle: {float}
        Orientation of the disk [deg],\n
    x0, y0: {float}
        Shift along x and y position [rad].
    """
    u = Utable / Lambda
    v = Vtable / Lambda

    # List of parameter
    elong = param["elong"]
    minorAxis = mas2rad(param["minorAxis"])
    majorAxis = elong * minorAxis
    angle = np.deg2rad(param["angle"])
    x0 = mas2rad(param["x0"])
    y0 = mas2rad(param["y0"])

    r = np.sqrt(
        ((u * np.sin(angle) + v * np.cos(angle)) * majorAxis) ** 2
        + ((u * np.cos(angle) - v * np.sin(angle)) * minorAxis) ** 2
    )

    C_centered = 2 * special.j1(np.pi * r) / (np.pi * r)

    C = shiftFourier(Utable, Vtable, Lambda, C_centered, x0, y0)

    try:
        if elong < 1:
            C = 2
    except NameError:
        pass
    return C


def visGaussianDisk(Utable, Vtable, Lambda, param):
    """
    Compute complex visibility of a gaussian disk

    Params:
    -------
    fwhm: {float}
        fwhm of the disk [rad],\n
    x0, y0: {float}
        Shift along x and y position [rad].
    """
    u = Utable / Lambda
    v = Vtable / Lambda

    fwhm = param["fwhm"]
    x0 = param["x0"]
    y0 = param["y0"]

    r2 = ((np.pi ** 2) * (u ** 2 + v ** 2) * (fwhm ** 2)) / (4.0 * np.log(2.0))
    C_centered = np.exp(-r2)

    # Deplacement du plan image
    C = shiftFourier(Utable, Vtable, Lambda, C_centered, x0, y0)
    return C


def visEllipticalGaussianDisk(Utable, Vtable, Lambda, param):
    """
    Compute complex visibility of an elliptical gaussian disk

    Params:
    -------
    majorAxis: {float}
        Major axis of the disk [rad],\n
    minorAxis: {float}
        Minor axis of the disk [rad],\n
    angle: {float}
        Orientation of the disk [rad],\n
    x0, y0: {float}
        Shift along x and y position [rad].
    """
    u = Utable / Lambda
    v = Vtable / Lambda

    # List of parameter
    majorAxis = param["majorAxis"]
    minorAxis = param["minorAxis"]
    angle = param["angle"]
    x0 = param["x0"]
    y0 = param["y0"]

    r2 = (
        (np.pi * 2)
        * (
            ((u * np.sin(angle) + v * np.cos(angle)) * majorAxis) ** 2
            + ((u * np.cos(angle) - v * np.sin(angle)) * minorAxis) ** 2
        )
        / (4.0 * np.log(2.0))
    )

    C_centered = np.exp(-r2)
    C = shiftFourier(Utable, Vtable, Lambda, C_centered, x0, y0)
    return C


def visThickEllipticalRing(Utable, Vtable, Lambda, param):
    """
    Compute complex visibility of an elliptical thick ring.

    Params:
    -------
    majorAxis: {float}
        Major axis of the disk [rad],\n
    minorAxis: {float}
        Minor axis of the disk [rad],\n
    angle: {float}
        Orientation of the disk [rad],\n
    thickness: {float}
        Thickness of the ring [rad],\n
    x0, y0: {float}
        Shift along x and y position [rad].
    """

    majorAxis = param["majorAxis"]
    minorAxis = param["minorAxis"]
    angle = param["angle"]
    thickness = param["thickness"]
    x0 = param["x0"]
    y0 = param["y0"]

    u = Utable / Lambda
    v = Vtable / Lambda

    r = np.sqrt(
        ((u * np.sin(angle) + v * np.cos(angle)) * majorAxis) ** 2
        + ((u * np.cos(angle) - v * np.sin(angle)) * minorAxis) ** 2
    )

    C_centered = special.j0(np.pi * r)
    C_shifted = shiftFourier(Utable, Vtable, Lambda, C_centered, x0, y0)
    C = C_shifted * visGaussianDisk(
        Utable, Vtable, Lambda, {"fwhm": thickness, "x0": 0.0, "y0": 0.0}
    )
    return C


def visEllipticalRing(Utable, Vtable, Lambda, param):
    """
    Compute complex visibility of an elliptical ring (infinitly thin).

    Params:
    -------
    majorAxis: {float}
        Major axis of the disk [rad],\n
    minorAxis: {float}
        Minor axis of the disk [rad],\n
    angle: {float}
        Orientation of the disk [rad],\n
    x0, y0: {float}
        Shift along x and y position [rad].
    """

    # List of parameter
    majorAxis = param["majorAxis"]
    minorAxis = param["minorAxis"]
    angle = param["angle"]
    x0 = param["x0"]
    y0 = param["y0"]

    u = Utable / Lambda
    v = Vtable / Lambda

    r = np.sqrt(
        ((u * np.sin(angle) + v * np.cos(angle)) * majorAxis) ** 2
        + ((u * np.cos(angle) - v * np.sin(angle)) * minorAxis) ** 2
    )

    C_centered = special.j0(np.pi * r)
    C = shiftFourier(Utable, Vtable, Lambda, C_centered, x0, y0)
    return C


def visDebrisDisk(Utable, Vtable, Lambda, param):
    """
    Compute complex visibility of an elliptical thick ring.

    Params:
    -------
    majorAxis: {float}
        Major axis of the disk [rad],\n
    minorAxis: {float}
        Minor axis of the disk [rad],\n
    angle: {float}
        Orientation of the disk [rad],\n
    thickness: {float}
        Thickness of the ring [rad],\n
    x0, y0: {float}
        Shift along x and y position [rad].
    """

    majorAxis = mas2rad(param["majorAxis"])*2
    inclination = np.deg2rad(param['incl'])
    posang = np.deg2rad(param["posang"])
    thickness = mas2rad(param["thickness"])
    cr_star = param["cr"]
    x0 = param["x0"]
    y0 = param["y0"]

    minorAxis = majorAxis * np.cos(inclination)

    #majorAxis = majorAxis_c * np.cos(inclination) - minorAxis_c * np.sin(inclination)
    #minorAxis = -majorAxis_c * np.sin(inclination) + minorAxis_c * np.cos(inclination)

    u = Utable / Lambda
    v = Vtable / Lambda

    r = np.sqrt(
        ((u * np.sin(posang) + v * np.cos(posang)) * majorAxis) ** 2
        + ((u * np.cos(posang) - v * np.sin(posang)) * minorAxis) ** 2
    )

    C_centered = special.j0(np.pi * r)
    C_shifted = shiftFourier(Utable, Vtable, Lambda, C_centered, x0, y0)
    C = C_shifted * visGaussianDisk(Utable, Vtable, Lambda,
                                    {"fwhm": thickness, "x0": 0.0, "y0": 0.0})

    fstar = cr_star
    fdisk = 1
    total_flux = fstar + fdisk

    rel_star = fstar / total_flux
    rel_disk = fdisk / total_flux

    p_s1 = {'x0': x0, 'y0': y0}
    s1 = rel_star * visPointSource(Utable, Vtable, Lambda, p_s1)
    s2 = rel_disk * C
    return s1 + s2


def visClumpDebrisDisk(Utable, Vtable, Lambda, param):
    """
    Compute complex visibility of an elliptical thick ring.

    Params:
    -------
    majorAxis: {float}
        Major axis of the disk [rad],\n
    minorAxis: {float}
        Minor axis of the disk [rad],\n
    angle: {float}
        Orientation of the disk [rad],\n
    thickness: {float}
        Thickness of the ring [rad],\n
    x0, y0: {float}
        Shift along x and y position [rad].
    """

    majorAxis = mas2rad(param["majorAxis"])*2
    inclination = np.deg2rad(param['incl'])
    posang = np.deg2rad(param["posang"])
    thickness = mas2rad(param["thickness"])
    cr_star = param["cr"]
    x0 = param["x0"]
    y0 = param["y0"]

    minorAxis = majorAxis * np.cos(inclination)

    #majorAxis = majorAxis_c * np.cos(inclination) - minorAxis_c * np.sin(inclination)
    #minorAxis = -majorAxis_c * np.sin(inclination) + minorAxis_c * np.cos(inclination)

    u = Utable / Lambda
    v = Vtable / Lambda

    r = np.sqrt(
        ((u * np.sin(posang) + v * np.cos(posang)) * majorAxis) ** 2
        + ((u * np.cos(posang) - v * np.sin(posang)) * minorAxis) ** 2
    )

    d_clump = mas2rad(param['d_clump'])
    cr_clump = param['cr_clump']/100.

    x1 = 0
    y1 = majorAxis * np.cos(inclination)
    x_clump = ((x1 * np.cos(posang) - y1 * np.sin(posang))/2.)
    y_clump = ((x1 * np.sin(posang) + y1 * np.cos(posang))/2.)

    p_clump = {"fwhm": d_clump,
               "x0": x_clump,
               "y0": y_clump}

    C_clump = visGaussianDisk(Utable, Vtable, Lambda, p_clump)

    C_centered = special.j0(np.pi * r)
    C_shifted = shiftFourier(Utable, Vtable, Lambda, C_centered, x0, y0)
    C = C_shifted * visGaussianDisk(Utable, Vtable, Lambda,
                                    {"fwhm": thickness, "x0": 0.0, "y0": 0.0})

    fstar = cr_star
    fdisk = 1
    total_flux = fstar + fdisk

    f_clump = cr_clump * total_flux
    f_debrisdisk = (1 - cr_clump) * total_flux

    rel_star = fstar / total_flux
    rel_disk = fdisk / total_flux

    p_s1 = {'x0': x0, 'y0': y0}
    s1 = rel_star * visPointSource(Utable, Vtable, Lambda, p_s1)
    s2 = rel_disk * C
    deb_disk = s1 + s2
    return f_debrisdisk * deb_disk + f_clump * C_clump


def compute_param_elts(ruse, tetuse, alpha, thick, incl, angleSky, angle_0,
                       step, rounds, rnuc=0, proj=True, limit_speed=False, display=False,
                       verbose=False):
    angle0 = tetuse + angle_0
    x0 = ruse * np.cos(angle0)
    y0 = ruse * np.sin(angle0)
    fwhmy0 = thick
    x1 = x0 * np.cos(incl)
    y1 = y0
    angle1 = np.arctan2(x1, y1)
    fwhmx1 = fwhmy0 * np.cos(angle0) * np.sin(incl)
    fwhmy1 = fwhmy0
    angle2 = np.transpose(angle1 + angleSky)
    x2 = np.transpose(x1 * np.cos(angleSky) + y1 * np.sin(angleSky))
    y2 = np.transpose(-x1 * np.sin(angleSky) + y1 * np.cos(angleSky))
    fwhmx2 = np.transpose(fwhmx1)
    fwhmy2 = np.transpose(fwhmy1)

    if proj:
        proj_fact = np.cos(alpha/2.)
        if verbose:
            print('Projection factor θ (%2.1f) = %2.2f' %
                  (np.rad2deg(alpha), proj_fact))
    else:
        proj_fact = 1

    decx = rad2mas(fwhmy2/2.) * np.cos(angle2)*proj_fact
    decy = rad2mas(fwhmy2/2.) * np.sin(angle2)*proj_fact
    px0, py0 = -rad2mas(x2)*proj_fact, -rad2mas(y2)*proj_fact
    px1, py1 = px0 - decx, py0 + decy
    px2, py2 = px0 + decx, py0 - decy
    dwall1 = (px1**2+py1**2)**0.5
    dwall2 = (px1**2+py1**2)**0.5

    lim = rounds*rad2mas(step)
    if limit_speed:
        limit_speed_cond = (dwall1 <= lim) & (
            dwall2 <= lim)
    else:
        limit_speed_cond = [True]*len(dwall1)

    if display:
        tmp = np.linspace(0, 2*np.pi, 300)
        xrnuc, yrnuc = rnuc*np.cos(tmp), rnuc*np.sin(tmp)
        plt.figure(figsize=(9, 8))
        plt.plot(px0[limit_speed_cond], py0[limit_speed_cond],
                 '.', color='#0d4c36', label='Archimedean spiral')
        for i in range(len(px1[limit_speed_cond])):
            plt.plot([px1[i], px2[i]], [py1[i], py2[i]],
                     '-', color='#00b08b', alpha=.5, lw=1)
        plt.plot(px1[limit_speed_cond], py1[limit_speed_cond],
                 '--', color='#ce0058', label='Spiral wall', lw=0.5)
        plt.plot(px2[limit_speed_cond], py2[limit_speed_cond],
                 '--', color='#ce0058', lw=0.5)
        for j in range(int(rounds)):
            radius = (j+1) * rad2mas(step)
            prop_limx, prop_limy = radius*np.cos(tmp), radius*np.sin(tmp)
            plt.plot(prop_limx, prop_limy, 'k-', lw=1)  # , label='S%i')
        plt.plot(xrnuc, yrnuc, 'r:', lw=1)
        plt.plot(0, 0, 'rx', label='WR star')
        plt.legend(fontsize=7, loc=1)
        plt.axis(np.array([-lim, lim, lim, -lim])*2.)
        plt.gca().set_aspect('equal', adjustable='box')
        plt.tight_layout()
        plt.show(block=False)
    return x0*proj_fact, y0*proj_fact, x2*proj_fact, y2*proj_fact, fwhmx2*proj_fact, fwhmy2*proj_fact, angle2


def compute_param_elts_spec(mjd, param, verbose=True, display=True):
    """ Compute only once the elements parameters fo the spiral. """
    # Pinwheel parameters
    rounds = float(param["Nturns"])

    # Convert appropriate units
    alpha = np.deg2rad(param["opening_angle"])
    r_nuc = mas2rad(param['r_nuc'])
    step = mas2rad(param["step"])

    # angle_0 = omega (orientation of the binary @ PA (0=north) counted counter-clockwise
    angle_0 = np.deg2rad(float(param["angle_0"]))
    incl = np.deg2rad(float(param["incl"]))
    angleSky = np.deg2rad(float(param["angleSky"]))

    totalSize = param["Nturns"] * step

    # rarely changed
    power = 1.  # float(param["power"])
    minThick = 0.  # float(param["minThick"])

    # Set opening angle
    maxThick = ((2*np.tan(alpha / 2.0) * (param["Nturns"] * step)))

    # Ring type of the pinwheel
    types = param["compo"]  # "r2"

    # fillFactor = param["fillFactor"]
    thick = np.mean((minThick + maxThick)) / 2.0

    N = int(param['nelts'] * param["Nturns"])

    # Full angular coordinates unaffected by the ecc.
    theta = np.linspace(0, rounds * 2 * np.pi, N)

    offst = mjd - param['mjd0']

    # Use kepler solver solution to compute the eccentric angular coordinates
    time = np.linspace(0, rounds*param['P'], N) - offst
    theta_ecc, E = kepler_solve(time, param['P'], param['e'])

    a_bin = param['a']  # [AU]
    sep_bin = a_bin * (1 - param['e']*np.cos(E))

    fact = totalSize * rounds / \
        (((rounds * 2.0 * np.pi) / (2.0 * np.pi)) ** power)
    r = (((theta) / (2.0 * np.pi)) ** power) * (fact / rounds)

    # Dust production if the binary separation is close enought (sep_bin < 's_prod')
    try:
        cond_prod = sep_bin <= param['s_prod']
    except KeyError:
        if verbose:
            cprint(
                'Warning: s_prod is not given, dust production is set to constant.', 'red')
        cond_prod = np.array([True]*len(sep_bin))

    dst = r * theta

    # opti = True
    # if opti:
    #     # optimal number of ring
    #     step1 = abs(np.diff(dst % (thick / fillFactor))
    #                 ) > (thick / fillFactor / 2.0)
    #     step1 = np.concatenate((step1, np.array([True])))

    step1 = np.array([True]*N)
    # Add the dust sublimation radius
    if r_nuc != 0:
        step1[(r <= r_nuc/np.cos(alpha/2.))] = False

    step1[~cond_prod] = False
    step2 = np.array(list(step1))
    step3 = np.transpose(step2)

    pr2 = np.array(np.where((abs(dst) == max(abs(dst))) | (
        abs(dst) == min(abs(dst))) | step3)[0])
    pr2 = pr2[1:]

    N2 = int(len(pr2))
    if verbose:
        print("Number of ring in the pinwheel N = %2.1f" % N2)

    # Use only selected rings position (sublimation and production limits applied)
    ruse = r[pr2]
    tetuse = theta_ecc[pr2]

    typei = [types] * N2

    thick = minThick + ruse / (max(r) + (max(r) == 0)) * maxThick

    proj = param['proj']
    tab_orient = compute_param_elts(ruse, tetuse, alpha, thick,
                                    incl, angleSky, angle_0, step, rounds, param['r_nuc'], proj=proj, limit_speed=False, display=display)
    tab_faceon = compute_param_elts(ruse, tetuse, alpha, thick,
                                    0, 0, angle_0, step=step, rounds=rounds, rnuc=param['r_nuc'],
                                    proj=proj, limit_speed=False, display=False, verbose=verbose)
    return tab_orient, tab_faceon, typei, N2, r_nuc, step, alpha


def sed_pwhl(wl, mjd, param, verbose=True, display=True):
    if 'a' not in param.keys():
        tab = getBinaryPos(mjd, param, mjd0=param['mjd0'], revol=1,
                           v=2, au=True, display=False)
        param['a'] = tab['a']

    tab_orient, tab_faceon, typei, N2, r_nuc, step, alpha = compute_param_elts_spec(
        mjd, param, verbose=verbose, display=display)

    dmas = rad2mas(np.sqrt(tab_faceon[2] ** 2 + tab_faceon[3] ** 2))
    Tin = param["T_sub"]
    q = param["q"]
    Tr = Tin * (dmas / rad2mas(r_nuc)) ** (-q)

    spec = []
    for xx in wl:
        l_Tr = Tr.copy()
        l_Tr[dmas >= np.cos(alpha/2.)*rad2mas(step)
             ] /= param["gap_factor"]
        spectrumi = param['f_scale_pwhl'] * planck_law(l_Tr, xx)
        spec.append(spectrumi)

    wl_sed = np.logspace(-8, -3.5, 1000)
    spec_all, spec_all1, spec_all2 = [], [], []
    for i in range(len(l_Tr)):
        spectrum_r = param['f_scale_pwhl'] * planck_law(l_Tr[i], wl_sed)
        spec_all.append(spectrum_r)
        if dmas[i] >= np.cos(alpha/2.)*rad2mas(step):
            spec_all2.append(spectrum_r)
        else:
            spec_all1.append(spectrum_r)

    spec_all, spec_all1, spec_all2 = np.array(
        spec_all), np.array(spec_all1), np.array(spec_all2)

    total_sed = np.sum(spec_all, axis=0)
    wl_peak = wl_sed[np.argmax(total_sed)]*1e6
    T_wien = 3000./wl_peak

    spec = np.array(spec)
    flux0 = np.sum(spec[0, :])
    if display:
        plt.figure()
        plt.loglog(wl_sed*1e6, spec_all1.T, color='grey')
        try:
            plt.loglog(wl_sed*1e6, spec_all2.T, color='lightgrey')
        except ValueError:
            pass
        plt.loglog(wl_sed*1e6, total_sed, color='#008080', lw=3, alpha=.8,
                   label=r'Total SED (T$_{wien}$ = %i K)' % T_wien)
        plt.plot(-1, -1, '-', color='grey', lw=3, label='Illuminated dust')
        plt.plot(-1, -1, '-', color='lightgrey', lw=3, label='Shadowed dust')
        plt.ylim(total_sed.max()/1e6, total_sed.max()*1e1)
        plt.legend(loc=2, fontsize=9)
        plt.xlabel("Wavelength [µm]")
        plt.ylabel("Blackbodies flux [arbitrary unit]")

        plt.figure()
        plt.plot(dmas, l_Tr)
        plt.xlabel('Distance [mas]')
        plt.ylabel('Temperature [K]')

    if verbose:
        print('Temperature law: r0 = %2.2f mas, T0 = %i K' %
              (dmas[0], Tr[0]))

    return np.array(spec)/N2, total_sed/N2, wl_sed*1e6, T_wien, spec_all1.T/N2, spec_all2.T/N2, flux0/N2, dmas, l_Tr


def visPinwheel(Utable, Vtable, Lambda, mjd, param, spec=None, fillFactor=20, verbose=True, display=True):
    """
    Compute complex visibility of an empty spiral.

    Params:
    -------
    rounds: {float}
        Number of turns,\n
    minThick: {float}
        Size of the smallest ring (given an opening angle) [rad],\n
    d_choc: {float}
        Dust formation radius [rad],\n
    anglePhi, angleSky, incl: {float}
        Orientaiton angle along the 3 axis [rad],\n
    opening_angle: {float}
        Opening angle [rad],\n
    compo: {str}
        Composition of the spiral ('r2': ring, 'g': , etc.),\n
    d_first_turn: {float}:
        Spiral step [rad],\n
    fillFactor: {int}
        Scale parameter to set the number of rings inside the spiral,\n
    x0, y0: {float}
        Shift along x and y position [rad].
    """

    # Start using the 'compute_param_elts_spec' function to determine the elements
    # parameters composing the spiral. Used to easely get these parameters and
    # determine the SED of the actual elements of the spiral.
    tab_orient, tab_faceon, typei, N2, r_nuc, step, alpha = compute_param_elts_spec(
        mjd, param, verbose=verbose, display=display)

    x0, y0, x2, y2, fwhmx2, fwhmy2, angle2 = tab_orient
    list_param = []

    if (len(x2)) == 0:
        if verbose:
            cprint(
                'Warning: No dust in the pinwheel given the parameters fov/rnuc/rs.', 'red')
        C = np.zeros(len(Utable))
        return C

    if len(Lambda) != len(x2):
        for i in range(len(x2)):
            list_param.append(
                {
                    "Lambda": Lambda[0],
                    "x0": x2[i],
                    "y0": y2[i],
                    "majorAxis": fwhmx2[i],
                    "minorAxis": fwhmy2[i],
                    "angle": angle2[i],
                }
            )
    else:
        for i in range(len(x2)):
            list_param.append(
                {
                    "Lambda": Lambda[i],
                    "x0": x2[i],
                    "y0": y2[i],
                    "majorAxis": fwhmx2[i],
                    "minorAxis": fwhmy2[i],
                    "angle": angle2[i],
                }
            )

    if param["q"] == 1.0:
        spectrumi = list(np.linspace(1, 0, N2))
    else:
        dmas = rad2mas(np.sqrt(tab_faceon[2] ** 2 + tab_faceon[3] ** 2))
        Tin = param["T_sub"]
        q = param["q"]
        Tr = Tin * (dmas / rad2mas(r_nuc)) ** (-q)

        if verbose:
            print('Temperature law: r0 = %2.2f mas, T0 = %i K' %
                  (dmas[0], Tr[0]))
        spectrumi = spec  # param["f_scale_pwhl"] * Planck_law(Tr, Lambda)
        # spectrumi[dmas >= np.cos(alpha/2.)*rad2mas(step)
        #           ] /= param["gap_factor"]

    C_centered = visMultipleResolved(Utable, Vtable, Lambda, typei, spectrumi,
                                     list_param)

    x0, y0 = mas2rad(param["x0"]), mas2rad(param["y0"])
    C = shiftFourier(Utable, Vtable, Lambda, C_centered, x0, y0)
    return C


def visMultipleResolved(Utable, Vtable, Lambda, typei, spec, list_param):
    """ Compute the complex visibility of a multi-component object."""
    n_obj = len(typei)
    corrFluxTmp = 0
    # for i in tqdm(range(n_obj), desc='Compute vis. components', leave=True, ncols=100,):
    for i in range(n_obj):
        if typei[i] == "r":
            Ci = visEllipticalRing(Utable, Vtable, Lambda, list_param[i])
        elif typei[i] == "t_r":
            Ci = visThickEllipticalRing(Utable, Vtable, Lambda, list_param[i])
        elif typei[i] == "ud":
            Ci = visUniformDisk(Utable, Vtable, Lambda, list_param[i])
        elif typei[i] == "e_ud":
            Ci = visEllipticakUniformDisk(
                Utable, Vtable, Lambda, list_param[i])
        elif typei[i] == "gd":
            Ci = visGaussianDisk(Utable, Vtable, Lambda, list_param[i])
        elif typei[i] == "e_gd":
            Ci = visEllipticalGaussianDisk(
                Utable, Vtable, Lambda, list_param[i])
        elif typei[i] == "star":
            Ci = visPointSource(Utable, Vtable, Lambda, list_param[i])
        elif typei[i] == "pwhl":
            Ci = visPinwheel(Utable, Vtable, Lambda, list_param[i])
        else:
            print("Model not yet in VisModels")
        spec2 = spec[i]
        Ci2 = Ci.reshape(1, np.size(Ci))
        corrFluxTmp += np.dot(spec2, Ci2)
    corrFlux = corrFluxTmp
    flux = np.sum(spec, 0)
    try:
        vis = corrFlux / flux
    except Exception:
        print("Error : flux = 0")
        vis = None
    return vis[0]
