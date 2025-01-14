"""
Created on Wed Nov  4 13:14:23 2015

@author: asoulain
"""
import numpy as np
from matplotlib import pyplot as plt
from scipy import special
from termcolor import cprint

from .binary import getBinaryPos
from .binary import kepler_solve
from .fourier import shiftFourier
from .tools import computeBinaryRatio
from .tools import mas2rad
from .tools import planck_law
from .tools import rad2mas


def _elong_gauss_disk(u, v, a=1.0, cosi=1.0, pa=0.0):
    """
    Return the complex visibility of an ellongated gaussian
    of size a cosi (a is the radius),
    position angle PA.
    PA is major-axis, East from North
    """

    # Elongated Gaussian
    rPA = pa - np.deg2rad(90)
    uM = u * np.cos(rPA) - v * np.sin(rPA)
    um = +u * np.sin(rPA) + v * np.cos(rPA)

    # a = rad2mas(a)
    aq2 = (a * uM) ** 2 + (a * cosi * um) ** 2
    return np.exp(-np.pi ** 2 * aq2 / (np.log(2))).astype(complex)


# tmp_f = np.exp(
#         -2
#         * np.pi
#         / np.sqrt(3)
#         * q
#         * a
#         * np.sqrt((np.cos(psi - theta) * cos_i) ** 2 + (np.sin(psi - theta)) ** 2)
#     )
def _elong_lorentz_disk(u, v, a, cosi, pa):
    # U = (u * np.sin(pa) + v * np.cos(pa)) * majorAxis
    # V = (u * np.cos(pa) - v * np.sin(pa)) * minorAxis
    rPA = pa - np.deg2rad(90)
    uM = u * np.cos(rPA) - v * np.sin(rPA)
    um = +u * np.sin(rPA) + v * np.cos(rPA)
    aq = ((a * uM) ** 2 + (a * cosi * um) ** 2) ** 0.5

    return np.exp(-(2 * np.pi * aq) / np.sqrt(3))

    # r2 = (
    #     (2 * np.pi) * (((U ** 2 + V ** 2) ** 0.5)) / np.sqrt(3)
    # )  # (2.31 close to gaussian fwhm)
    # C_centered = np.exp(-r2)
    # return C_centered


def norm(x, y):
    return np.sqrt(x ** 2 + y ** 2)


def _elong_ring(u, v, a=1.0, cosi=1.0, pa=0.0, c1=0.0, s1=0.0):
    """
    Return the complex visibility of an elongated ring
    of size a cosi,
    position angle PA.
    PA is major-axis, East from North
    """

    # Squeeze and rotation
    rPA = pa - np.deg2rad(90)
    uM = u * np.cos(rPA) - v * np.sin(rPA)
    um = u * np.sin(rPA) + v * np.cos(rPA)

    # Polar coordinates (check angle)
    z = 2.0 * np.pi * a * norm(uM, cosi * um)
    psi = np.arctan2(um, uM)

    # Modulation in polar
    rho1 = norm(c1, s1)
    phi1 = np.arctan2(s1, c1)

    if rho1 == 0:
        mod = 0
    else:
        # print(1.0j * rho1 * np.cos(psi - phi1) * special.jv(1, z))
        mod = -1.0j * rho1 * np.cos(psi - phi1) * special.jv(1, z)

    # Visibility
    v = special.jv(0, z) + mod
    return v.astype(complex)


def _azimuth_modulation_mod(u, v, lam, pa, cosi, ar, cj, sj):
    q = np.sqrt(u ** 2 + (cosi * v) ** 2) / lam
    psi = np.arctan2(u, v)

    rho_j = np.sqrt(cj ** 2 + sj ** 2)
    theta_j = np.arctan2(sj, cj)
    theta = pa

    sum_mod = complex(0, 0)
    if not hasattr(cj, "__len__"):
        sum_mod += (
            (-1j)
            * rho_j
            * np.cos(psi - theta_j)
            * special.jn(
                2 * np.pi * q * ar,
                1,
            )
        )
    else:
        for i in range(1, len(rho_j) + 1):
            sum_mod += (
                (-1j) ** i
                * rho_j[i - 1]
                * np.cos(i * (psi - theta - theta_j[i - 1]))
                * special.jn(
                    2
                    * np.pi
                    * q
                    * ar
                    * np.sqrt(
                        (np.cos(psi - theta) * cosi) ** 2 + np.sin(psi - theta) ** 2
                    ),
                    i,
                )
            )
    return sum_mod


def visPointSource(Utable, Vtable, Lambda, param):
    """
    Compute complex visibility of a point source.

    Params:
    -------
    x0, y0: {float}
        Shift along x and y position [rad].
    """
    C_centered = np.ones(np.size(Utable))
    C = shiftFourier(Utable, Vtable, Lambda, C_centered, param["x0"], param["y0"])
    return C


def visBinary(Utable, Vtable, Lambda, param):
    sep = mas2rad(param["sep"])
    dm = param["dm"]
    theta = np.deg2rad(90 - param["pa"])

    if dm < 0:
        return np.array([np.nan] * len(Lambda))

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


def visBinary_res(Utable, Vtable, Lambda, param):
    sep = mas2rad(param["sep"])
    dm = param["dm"]
    theta = np.deg2rad(90 - param["pa"])
    diam = param["diam"]
    if dm < 0:
        return np.array([np.nan] * len(Lambda))
    f1 = 1
    f2 = f1 / (10 ** (0.4 * dm))
    ftot = f1 + f2

    rel_f1 = f1 / ftot
    rel_f2 = f2 / ftot

    p_s1 = {"x0": 0, "y0": 0, "diam": diam}
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
    u = Utable / Lambda[:, None]
    v = Vtable / Lambda[:, None]

    diam = mas2rad(param["diam"])

    r = np.sqrt(u ** 2 + v ** 2)

    C_centered = 2 * special.j1(np.pi * r * diam) / (np.pi * r * diam)
    C = shiftFourier(
        Utable, Vtable, Lambda, C_centered, mas2rad(param["x0"]), mas2rad(param["y0"])
    )
    return C


def visEllipticalUniformDisk(Utable, Vtable, Lambda, param):
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
    u = Utable / Lambda[:, None]
    v = Vtable / Lambda[:, None]

    # List of parameter
    elong = np.cos(np.deg2rad(param["incl"]))
    majorAxis = mas2rad(param["majorAxis"])
    minorAxis = elong * majorAxis
    angle = np.deg2rad(param["pa"])
    x0 = mas2rad(param["x0"])
    y0 = mas2rad(param["y0"])

    r = np.sqrt(
        ((u * np.sin(angle) + v * np.cos(angle)) * majorAxis) ** 2
        + ((u * np.cos(angle) - v * np.sin(angle)) * minorAxis) ** 2
    )

    r[r == 0] = np.nan

    C_centered = 2 * special.j1(np.pi * r) / (np.pi * r)
    C = shiftFourier(Utable, Vtable, Lambda, C_centered, x0, y0)
    return C


def visGaussianDisk(Utable, Vtable, Lambda, param):
    """
    Compute complex visibility of a gaussian disk

    Params:
    -------
    fwhm: {float}
        fwhm of the disk [mas],\n
    x0, y0: {float}
        Shift along x and y position [rad].
    """
    u = Utable / Lambda[:, None]
    v = Vtable / Lambda[:, None]

    fwhm = mas2rad(param["fwhm"])
    x0 = mas2rad(param.get("x0", 0))
    y0 = mas2rad(param.get("y0", 0))

    q = (u ** 2 + v ** 2) ** 0.5
    r2 = ((np.pi * q * fwhm) ** 2) / (4 * np.log(2.0))
    C_centered = np.exp(-r2)

    if x0 == y0 == 0:
        C = C_centered
    else:
        C = shiftFourier(Utable, Vtable, Lambda, C_centered, x0, y0)
    return C


def visEllipticalGaussianDisk(Utable, Vtable, Lambda, param):
    """
    Compute complex visibility of an elliptical gaussian disk

    Params:
    -------
    majorAxis: {float}
        Major axis of the disk [,as],\n
    minorAxis: {float}
        Minor axis of the disk [rad],\n
    angle: {float}
        Orientation of the disk [rad],\n
    x0, y0: {float}
        Shift along x and y position [rad].
    """
    u = Utable / Lambda[:, None]
    v = Vtable / Lambda[:, None]

    # List of parameter
    elong = np.cos(np.deg2rad(param["incl"]))
    majorAxis = mas2rad(param["majorAxis"])
    minorAxis = elong * majorAxis
    angle = np.deg2rad(param["pa"])
    x0 = param["x0"]
    y0 = param["y0"]

    r2 = (
        (np.pi ** 2)
        * (
            ((u * np.sin(angle) + v * np.cos(angle)) * majorAxis) ** 2
            + ((u * np.cos(angle) - v * np.sin(angle)) * minorAxis) ** 2
        )
        / (4.0 * np.log(2.0))
    )

    C_centered = np.exp(-r2)
    C = shiftFourier(Utable, Vtable, Lambda, C_centered, x0, y0)
    return C


def visCont(Utable, Vtable, Lambda, param):
    """
    Compute complex visibility double gaussian model (same as TW Hya for the
    IR continuum).

    Params:
    -------
    `fwhm` {float}:
        Full major axis of the disk [rad],\n
    `incl` {float}:
        Inclination (minorAxis = `majorAxis` * elong (`elong` = cos(`incl`)),\n
    `pa` {float}:
        Orientation of the disk (from north to East) [rad],\n
    `fratio` {float}:
        Stellar to total flux ratio (i.e: 1/fratio = f* [%]),\n
    `rstar` {float}:
        Radius of the star [mas],\n
    """
    u = Utable / Lambda[:, None]
    v = Vtable / Lambda[:, None]

    pa = np.deg2rad(param["pa"])

    # Define the reduction ratio apply on the fwhm
    # because of the inclination
    incl = np.deg2rad(param["incl"])
    elong = np.cos(incl)
    majorAxis = mas2rad(param["fwhm"])

    # Define stellar flux ratio
    fratio = param["fratio"]
    Fstar = 1.0 / fratio
    Fdisc = 1 - Fstar

    # Stellar radius (resolved disk)
    rstar = 2 * param["rstar"]

    Vdisc = _elong_gauss_disk(u, v, a=majorAxis, cosi=elong, pa=pa)

    p_star = {"fwhm": rstar, "x0": 0, "y0": 0}
    Vstar = visGaussianDisk(Utable, Vtable, Lambda, p_star)

    Vcont = (Fdisc * Vdisc) + (Fstar * Vstar)
    return Vcont


def visYSO(Utable, Vtable, Lambda, param):
    """
    Compute complex visibility of a YSO model (star + gaussian disk + resolved
    halo). The halo contribution is computed with 1 - fc - fs.

    Params:
    -------
    `hfr` {float}:
        Half major axis of the disk [rad],\n
    `incl` {float}:
        Inclination (minorAxis = `majorAxis` * elong (`elong` = cos(`incl`)),\n
    `pa` {float}:
        Orientation of the disk (from north to East) [rad],\n
    `fs` {float}:
        Flux contribution of the star [%],\n
    `fc` {float}:
        Flux contribution of the disk [%],\n
    """
    fs = param["fs"]
    fc = param["fc"]
    fh = 1 - fs - fc

    param_disk = {
        "fwhm": 2 * param["hfr"],  # For ellipsoid, fwhm is the radius
        "flor": param["flor"],
        "pa": param["pa"],
        "incl": param["incl"],
    }

    C = visEllipsoid(Utable, Vtable, Lambda, param_disk)

    p_s1 = {"x0": 0, "y0": 0}
    s1 = fs * visPointSource(Utable, Vtable, Lambda, p_s1)
    s2 = fc * C
    ftot = fs + fh + fc
    return (s1 + s2) / ftot


def visLazareff(Utable, Vtable, Lambda, param):
    """
    Compute complex visibility of a Lazareff model (star + thick ring + resolved
    halo). The halo contribution is computed with 1 - fc - fs.

    Params:
    -------
    `la` {float}:
        Half major axis of the disk (log),\n
    `lr` {float}:
        Kernel half light (log),\n
    `flor` {float}:
        Weighting for radial profile (0 gaussian kernel,
        1 Lorentizian kernel),\n
    `incl` {float}:
        Inclination (minorAxis = `majorAxis` * elong (`elong` = cos(`incl`)),\n
    `pa` {float}:
        Orientation of the disk (from north to East) [rad],\n
    `fs` {float}:
        Flux contribution of the star [%],\n
    `fc` {float}:
        Flux contribution of the disk [%],\n
    `ks` {float}:
        Spectral index compared to reference wave at 2.2 µm,\n
    `c1`, `s1` {float}:
        Cosine and sine amplitude for the mode 1 (azimutal changes),\n

    """
    u = Utable / Lambda[:, None]
    v = Vtable / Lambda[:, None]
    # List of parameter

    elong = np.cos(np.deg2rad(param["incl"]))
    la = param["la"]
    lk = param["lk"]

    kr = 10.0 ** (lk)
    ar = 10 ** la / (np.sqrt(1 + kr ** 2))
    ak = ar * kr
    # print(ar, ak)
    # self.a_r = 10 ** self.l_a / np.sqrt(1 + 10 ** (2 * self.l_kr))
    # self.a_k = 10 ** self.l_kr * self.a_r

    ar_rad = mas2rad(ar)
    semi_majorAxis = ar_rad

    pa = np.deg2rad(param["pa"])

    fs = param["fs"]
    fc = param["fc"]
    fh = 1 - fc - fs

    if param["type"] == "smooth":
        param_ker = {
            "pa": param["pa"],
            "incl": param["incl"],
            "fwhm": 2 * ak,
            "flor": param["flor"],
        }
        Vkernel = visEllipsoid(Utable, Vtable, Lambda, param_ker)
    elif param["type"] == "uniform":
        param_ker = {
            "diam": 2 * ak,
            "x0": 0,
            "y0": 0,
        }
        Vkernel = visUniformDisk(Utable, Vtable, Lambda, param_ker)
    else:
        Vkernel = 1

    try:
        cj = param["cj"]
        sj = param["sj"]
    except Exception:
        cj = 0
        sj = 0

    Vring = (
        _elong_ring(u, v, a=semi_majorAxis, cosi=elong, pa=pa, c1=cj, s1=sj) * Vkernel
    )

    ks = param["ks"]
    kc = param["kc"]
    wl0 = 2.2e-6

    fs_lambda = fs * (wl0 / Lambda) ** ks
    fc_lambda = fc * (wl0 / Lambda) ** kc
    fh_lambda = fh * (wl0 / Lambda) ** ks
    p_s1 = {"x0": 0, "y0": 0}
    s1 = fs_lambda[:, None] * visPointSource(Utable, Vtable, Lambda, p_s1)
    s2 = fc_lambda[:, None] * Vring
    ftot = fs_lambda + fh_lambda + fc_lambda
    return s1 + s2 / ftot[:, None]


def visLazareff_halo(Utable, Vtable, Lambda, param):
    """
    Compute complex visibility of a Lazareff model (star + thick ring + resolved
    halo). The star contribution is computed with 1 - fc - fh.

    Params:
    -------
    `la` {float}:
        Half major axis of the disk (log),\n
    `lr` {float}:
        Kernel half light (log),\n
    `flor` {float}:
        Weighting for radial profile (0 gaussian kernel,
        1 Lorentizian kernel),\n
    `incl` {float}:
        Inclination (minorAxis = `majorAxis` * elong (`elong` = cos(`incl`)),\n
    `pa` {float}:
        Orientation of the disk (from north to East) [rad],\n
    `fh` {float}:
        Flux contribution of the halo [%],\n
    `fc` {float}:
        Flux contribution of the disk [%],\n
    `ks` {float}:
        Spectral index compared to reference wave at 2.2 µm,\n
    `c1`, `s1` {float}:
        Cosine and sine amplitude for the mode 1 (azimutal changes),\n

    """
    u = Utable / Lambda[:, None]
    v = Vtable / Lambda[:, None]
    # List of parameter

    elong = np.cos(np.deg2rad(param["incl"]))
    la = param["la"]
    lk = param["lk"]

    kr = 10.0 ** (lk)
    ar = 10 ** la / (np.sqrt(1 + kr ** 2))
    ak = ar * kr

    ar_rad = mas2rad(ar)
    semi_majorAxis = ar_rad

    pa = np.deg2rad(param["pa"])

    fh = param["fh"]
    fc = param["fc"]
    fs = 1 - fc - fh

    if param["type"] == "smooth":
        param_ker = {
            "pa": param["pa"],
            "incl": param["incl"],
            "fwhm": 2 * ak,
            "flor": param["flor"],
        }
        Vkernel = visEllipsoid(Utable, Vtable, Lambda, param_ker)
    elif param["type"] == "uniform":
        param_ker = {
            "diam": 2 * ak,
            "x0": 0,
            "y0": 0,
        }
        Vkernel = visUniformDisk(Utable, Vtable, Lambda, param_ker)
    else:
        Vkernel = 1

    try:
        cj = param["cj"]
        sj = param["sj"]
    except Exception:
        cj = 0
        sj = 0

    Vring = (
        _elong_ring(u, v, a=semi_majorAxis, cosi=elong, pa=pa, c1=cj, s1=sj) * Vkernel
    )

    ks = param["ks"]
    kc = param["kc"]
    wl0 = 2.2e-6

    fs_lambda = fs * (wl0 / Lambda) ** ks
    fc_lambda = fc * (wl0 / Lambda) ** kc
    fh_lambda = fh * (wl0 / Lambda) ** ks
    p_s1 = {"x0": 0, "y0": 0}
    s1 = fs_lambda[:, None] * visPointSource(Utable, Vtable, Lambda, p_s1)
    s2 = fc_lambda[:, None] * Vring
    ftot = fs_lambda + fh_lambda + fc_lambda
    return s1 + s2 / ftot[:, None]


def visLazareff_line(Utable, Vtable, Lambda, param):
    """
    Compute complex visibility of a Lazareff model (star + thick ring + resolved
    halo). The halo contribution is computed with 1 - fc - fs.

    Params:
    -------
    `la` {float}:
        Half major axis of the disk (log),\n
    `lr` {float}:
        Kernel half light (log),\n
    `flor` {float}:
        Weighting for radial profile (0 gaussian kernel,
        1 Lorentizian kernel),\n
    `incl` {float}:
        Inclination (minorAxis = `majorAxis` * elong (`elong` = cos(`incl`)),\n
    `pa` {float}:
        Orientation of the disk (from north to East) [rad],\n
    `fs` {float}:
        Flux contribution of the star [%],\n
    `fc` {float}:
        Flux contribution of the disk [%],\n
    `ks` {float}:
        Spectral index compared to reference wave at 2.2 µm,\n
    `c1`, `s1` {float}:
        Cosine and sine amplitude for the mode 1 (azimutal changes),\n

    """
    u = Utable / Lambda[:, None]
    v = Vtable / Lambda[:, None]
    # List of parameter

    elong = np.cos(np.deg2rad(param["incl"]))
    la = param["la"]
    lk = param["lk"]

    kr = 10.0 ** (lk)
    ar = 10 ** la / (np.sqrt(1 + kr ** 2))
    ak = ar * (10 ** lk)

    ar_rad = mas2rad(ar)
    majorAxis = ar_rad
    # minorAxis = majorAxis * elong

    pa = np.deg2rad(param["pa"])

    fs = param["fs"]
    fc = param["fc"]
    fh = 1 - fs - fc

    param_ker = {
        "pa": param["pa"],
        "incl": param["incl"],
        "fwhm": ak,
        "flor": param["flor"],
    }

    Vkernel = visEllipsoid(Utable, Vtable, Lambda, param_ker)

    try:
        cj = param["cj"]
        sj = param["sj"]
    except Exception:
        cj = 0
        sj = 0

    Vring = (_elong_ring(u, v, a=majorAxis, cosi=elong, pa=pa, c1=cj, s1=sj)) * Vkernel

    # Vring = (_elong_ring(u, v, pa, majorAxis, minorAxis) + azimuth_mod) * Vkernel

    ks = param["ks"]
    kc = param["kc"]
    wl0 = 2.2e-6

    lF = param["lF"]

    param_line = {
        "pa": param["lpa"],
        "incl": param["lincl"],
        "fwhm": param["lT"],
        "flor": 0,
    }

    lbdBrg = param["wl_brg"] * 1e-6
    sigBrg = param["sig_brg"] * 1e-6

    # Line emission
    Fl = lF * np.exp(-0.5 * (Lambda - lbdBrg) ** 2 / sigBrg ** 2)
    Vl = visEllipsoid(Utable, Vtable, Lambda, param_line)

    shift_x = mas2rad(1e-3 * param["shift_x"])
    shift_y = mas2rad(1e-3 * param["shift_y"])

    # Shift of line emission
    Vl = shiftFourier(Utable, Vtable, Lambda, Vl, shift_x, shift_y)

    fs_lambda = fs * (wl0 / Lambda) ** ks
    fc_lambda = fc * (wl0 / Lambda) ** kc
    fh_lambda = fh * (wl0 / Lambda) ** ks
    p_s1 = {"x0": 0, "y0": 0}
    s1 = fs_lambda * visPointSource(Utable, Vtable, Lambda, p_s1)
    s2 = fc_lambda * Vring
    ftot = fs_lambda + fh_lambda + fc_lambda

    return (s1 + s2 + Fl * Vl) / (ftot + Fl)


def visThickEllipticalRing(Utable, Vtable, Lambda, param):
    """
    Compute complex visibility of an elliptical thick ring.

    Params:
    -------
    majorAxis: {float}
        Major axis of the disk [mas],\n
    incl: {float}
        Inclination of the disk [deg],\n
    angle: {float}
        Position angle of the disk [deg],\n
    w: {float}
        Thickness of the ring [mas],\n
    x0, y0: {float}
        Shift along x and y position [rad].
    """

    elong = np.cos(np.deg2rad(param["incl"]))
    majorAxis = mas2rad(param["majorAxis"])
    minorAxis = elong * majorAxis
    angle = np.deg2rad(param["pa"])
    thickness = mas2rad(param["w"])

    x0 = param.get("x0", 0)
    y0 = param.get("y0", 0)

    u = Utable / Lambda[:, None]
    v = Vtable / Lambda[:, None]

    r = np.sqrt(
        ((u * np.sin(angle) + v * np.cos(angle)) * majorAxis) ** 2
        + ((u * np.cos(angle) - v * np.sin(angle)) * minorAxis) ** 2
    )

    C_centered = special.j0(2 * np.pi * r)
    C_shifted = shiftFourier(Utable, Vtable, Lambda, C_centered, x0, y0)
    C = C_shifted * visGaussianDisk(
        Utable, Vtable, Lambda, {"fwhm": rad2mas(thickness), "x0": 0.0, "y0": 0.0}
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

    u = Utable / Lambda[:, None]
    v = Vtable / Lambda[:, None]

    r = np.sqrt(
        ((u * np.sin(angle) + v * np.cos(angle)) * majorAxis) ** 2
        + ((u * np.cos(angle) - v * np.sin(angle)) * minorAxis) ** 2
    )

    C_centered = special.j0(np.pi * r)
    C = shiftFourier(Utable, Vtable, Lambda, C_centered, x0, y0)
    return C


def visEllipsoid(Utable, Vtable, Lambda, param):
    """
    Compute complex visibility of an ellipsoid (as Lazareff+17).

    Params:
    -------
    `fwhm` {float}:
        FWHM of the disk,\n
    `incl` {float}:
        Inclination of the disk [deg],\n
    `pa` {float}:
        Orientation of the disk [deg],\n
    `flor` {float}:
        Hybridation between purely gaussian (flor=0)
        and Lorentzian radial profile (flor=1).
    """
    u = Utable / Lambda[:, None]
    v = Vtable / Lambda[:, None]

    pa = np.deg2rad(param["pa"])

    # Define the reduction ratio apply on the fwhm
    # because of the inclination
    incl = np.deg2rad(param["incl"])
    elong = np.cos(incl)
    semi_majorAxis = mas2rad(param["fwhm"]) / 2.0
    # minorAxis = elong * majorAxis

    # majorAxis is the half-radius

    flor = param["flor"]

    Vlor = _elong_lorentz_disk(u, v, a=semi_majorAxis, cosi=elong, pa=pa)
    Vgauss = _elong_gauss_disk(u, v, a=semi_majorAxis, cosi=elong, pa=pa)

    Vc = (1 - flor) * Vgauss + flor * Vlor
    return Vc


def visLorentzDisk(Utable, Vtable, Lambda, param):
    """
    Compute complex visibility of an Lorentzian disk.

    Params:
    -------
    fwhm: {float}
        Size of the disk [mas],\n
    x0, y0: {float}
        Shift along x and y position [mas].
    """
    u = Utable / Lambda[:, None]
    v = Vtable / Lambda[:, None]

    fwhm = mas2rad(param["fwhm"])
    x0 = mas2rad(param["x0"])
    y0 = mas2rad(param["y0"])

    q = (u ** 2 + v ** 2) ** 0.5
    r = 2 * np.pi * fwhm * q / np.sqrt(3)
    C_centered = np.exp(-r)
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

    majorAxis = mas2rad(param["majorAxis"])
    elong = np.cos(np.deg2rad(param["incl"]))
    posang = np.deg2rad(param["pa"])
    thickness = param["w"]
    cr_star = param["cr"]
    x0 = param["x0"]
    y0 = param["y0"]

    minorAxis = majorAxis * elong

    u = Utable / Lambda[:, None]
    v = Vtable / Lambda[:, None]

    r = np.sqrt(
        ((u * np.sin(posang) + v * np.cos(posang)) * majorAxis) ** 2
        + ((u * np.cos(posang) - v * np.sin(posang)) * minorAxis) ** 2
    )

    C_centered = special.j0(np.pi * r)
    C_shifted = shiftFourier(Utable, Vtable, Lambda, C_centered, x0, y0)
    C = C_shifted * visGaussianDisk(
        Utable, Vtable, Lambda, {"fwhm": thickness, "x0": 0.0, "y0": 0.0}
    )

    fstar = cr_star
    fdisk = 1
    total_flux = fstar + fdisk

    rel_star = fstar / total_flux
    rel_disk = fdisk / total_flux

    p_s1 = {"x0": x0, "y0": y0}
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

    majorAxis = mas2rad(param["majorAxis"])
    elong = np.cos(np.deg2rad(param["incl"]))
    posang = np.deg2rad(param["pa"])
    thickness = param["w"]
    fs = param["fs"] / 100.0
    x0 = param["x0"]
    y0 = param["y0"]

    minorAxis = majorAxis * elong

    u = Utable / Lambda[:, None]
    v = Vtable / Lambda[:, None]

    r = np.sqrt(
        ((u * np.sin(posang) + v * np.cos(posang)) * majorAxis) ** 2
        + ((u * np.cos(posang) - v * np.sin(posang)) * minorAxis) ** 2
    )

    d_clump = param["d_clump"]
    pa_clump = -np.deg2rad(param["pa_clump"])
    fc = param["fc"] / 100.0

    x1 = 0
    y1 = majorAxis * elong
    x_clump = (x1 * np.cos(pa_clump) - y1 * np.sin(pa_clump)) / 2.0
    y_clump = (x1 * np.sin(pa_clump) + y1 * np.cos(pa_clump)) / 2.0

    p_clump = {"fwhm": d_clump, "x0": rad2mas(x_clump), "y0": rad2mas(y_clump)}

    C_clump = visGaussianDisk(Utable, Vtable, Lambda, p_clump)

    C_centered = special.j0(np.pi * r)
    C_shifted = shiftFourier(Utable, Vtable, Lambda, C_centered, x0, y0)
    C = C_shifted * visGaussianDisk(
        Utable, Vtable, Lambda, {"fwhm": thickness, "x0": 0.0, "y0": 0.0}
    )

    fd = 1 - fs - fc

    p_s1 = {"x0": x0, "y0": y0}
    c_star = fs * visPointSource(Utable, Vtable, Lambda, p_s1)
    c_ring = fd * C
    c_clump = fc * C_clump
    return c_star + c_ring + c_clump


def _compute_param_elts(
    ruse,
    tetuse,
    alpha,
    thick,
    incl,
    angleSky,
    angle_0,
    step,
    rounds,
    rnuc=0,
    proj=True,
    limit_speed=False,
    display=False,
    verbose=False,
):
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
        proj_fact = np.cos(alpha / 2.0)
        if verbose:
            print(f"Projection factor θ ({np.rad2deg(alpha):2.1f}) = {proj_fact:2.2f}")
    else:
        proj_fact = 1

    decx = rad2mas(fwhmy2 / 2.0) * np.cos(angle2) * proj_fact
    decy = rad2mas(fwhmy2 / 2.0) * np.sin(angle2) * proj_fact
    px0, py0 = -rad2mas(x2) * proj_fact, -rad2mas(y2) * proj_fact
    px1, py1 = px0 - decx, py0 + decy
    px2, py2 = px0 + decx, py0 - decy
    dwall1 = (px1 ** 2 + py1 ** 2) ** 0.5
    dwall2 = (px1 ** 2 + py1 ** 2) ** 0.5

    lim = rounds * rad2mas(step)
    if limit_speed:
        limit_speed_cond = (dwall1 <= lim) & (dwall2 <= lim)
    else:
        limit_speed_cond = [True] * len(dwall1)

    if display:
        tmp = np.linspace(0, 2 * np.pi, 300)
        xrnuc, yrnuc = rnuc * np.cos(tmp), rnuc * np.sin(tmp)
        plt.figure(figsize=(5, 5))
        plt.plot(
            px0[limit_speed_cond],
            py0[limit_speed_cond],
            ".",
            color="#0d4c36",
            label="Archimedean spiral",
        )
        for i in range(len(px1[limit_speed_cond])):
            plt.plot(
                [px1[i], px2[i]],
                [py1[i], py2[i]],
                "-",
                color="#00b08b",
                alpha=0.5,
                lw=1,
            )
        plt.plot(
            px1[limit_speed_cond],
            py1[limit_speed_cond],
            "--",
            color="#ce0058",
            label="Spiral wall",
            lw=0.5,
        )
        plt.plot(
            px2[limit_speed_cond], py2[limit_speed_cond], "--", color="#ce0058", lw=0.5
        )
        for j in range(int(rounds)):
            radius = (j + 1) * rad2mas(step)
            prop_limx, prop_limy = radius * np.cos(tmp), radius * np.sin(tmp)
            plt.plot(prop_limx, prop_limy, "k-", lw=1)  # , label='S%i')
        plt.plot(xrnuc, yrnuc, "r:", lw=1)
        plt.plot(0, 0, "rx", label="WR star")
        plt.legend(fontsize=7, loc=1)
        plt.axis(np.array([-lim, lim, lim, -lim]) * 2.0)
        plt.gca().set_aspect("equal", adjustable="box")
        plt.xlabel("RA [mas]")
        plt.ylabel("DEC [mas]")
        plt.tight_layout()
        plt.show(block=False)
    return (
        x0 * proj_fact,
        y0 * proj_fact,
        x2 * proj_fact,
        y2 * proj_fact,
        fwhmx2 * proj_fact,
        fwhmy2 * proj_fact,
        angle2,
    )


def _compute_param_elts_spec(mjd, param, verbose=True, display=True):
    """Compute only once the elements parameters fo the spiral."""
    # Pinwheel parameters
    rounds = float(param["rounds"])

    # Convert appropriate units
    alpha = np.deg2rad(param["opening_angle"])
    r_nuc = mas2rad(param["r_nuc"])
    step = mas2rad(param["step"])

    # angle_0 = omega (orientation of the binary @ PA (0=north) counted counter-clockwise
    angle_0 = np.deg2rad(float(param["angle_0"]))
    incl = np.deg2rad(float(param["incl"]))
    angleSky = np.deg2rad(float(param["angleSky"]))

    totalSize = param["rounds"] * step

    # rarely changed
    power = 1.0  # float(param["power"])
    minThick = 0.0  # float(param["minThick"])

    # Set opening angle
    maxThick = 2 * np.tan(alpha / 2.0) * (rounds * step)

    # Ring type of the pinwheel
    types = param["compo"]  # "r2"

    # fillFactor = param["fillFactor"]
    thick = np.mean(minThick + maxThick) / 2.0

    N = int(param["nelts"] * rounds)

    # Full angular coordinates unaffected by the ecc.
    theta = np.linspace(0, rounds * 2 * np.pi, N)

    offst = mjd - param["mjd0"]

    # Use kepler solver solution to compute the eccentric angular coordinates
    time = np.linspace(0, rounds * param["P"], N) - offst
    theta_ecc, E = kepler_solve(time, param["P"], param["e"])

    a_bin = param["a"]  # [AU]
    sep_bin = a_bin * (1 - param["e"] * np.cos(E))

    fact = totalSize * rounds / (((rounds * 2.0 * np.pi) / (2.0 * np.pi)) ** power)
    r = (((theta) / (2.0 * np.pi)) ** power) * (fact / rounds)

    # Dust production if the binary separation is close enought (sep_bin < 's_prod')
    try:
        cond_prod = sep_bin <= param["s_prod"]
    except KeyError:
        if verbose:
            cprint(
                "Warning: s_prod is not given, dust production is set to constant.",
                "red",
            )
        cond_prod = np.array([True] * len(sep_bin))

    dst = r * theta

    # opti = True
    # if opti:
    #     # optimal number of ring
    #     step1 = abs(np.diff(dst % (thick / fillFactor))
    #                 ) > (thick / fillFactor / 2.0)
    #     step1 = np.concatenate((step1, np.array([True])))

    step1 = np.array([True] * N)
    # Add the dust sublimation radius
    if r_nuc != 0:
        step1[(r <= r_nuc / np.cos(alpha / 2.0))] = False

    step1[~cond_prod] = False
    step2 = np.array(list(step1))
    step3 = np.transpose(step2)

    pr2 = np.array(
        np.where((abs(dst) == max(abs(dst))) | (abs(dst) == min(abs(dst))) | step3)[0]
    )
    pr2 = pr2[1:]

    N2 = int(len(pr2))
    if verbose:
        print("Number of ring in the pinwheel N = %2.1f" % N2)

    # Use only selected rings position (sublimation and production limits applied)
    ruse = r[pr2]
    tetuse = theta_ecc[pr2]

    typei = [types] * N2

    thick = minThick + ruse / (max(r) + (max(r) == 0)) * maxThick

    proj = True  # param["proj"]
    tab_orient = _compute_param_elts(
        ruse,
        tetuse,
        alpha,
        thick,
        incl,
        angleSky,
        angle_0,
        step,
        rounds,
        param["r_nuc"],
        proj=proj,
        limit_speed=False,
        display=display,
    )
    tab_faceon = _compute_param_elts(
        ruse,
        tetuse,
        alpha,
        thick,
        0,
        0,
        angle_0,
        step=step,
        rounds=rounds,
        rnuc=param["r_nuc"],
        proj=proj,
        limit_speed=False,
        display=False,
        verbose=verbose,
    )
    return tab_orient, tab_faceon, typei, N2, r_nuc, step, alpha


def sed_pwhl(wl, mjd, param, verbose=True, display=True):
    if "a" not in param.keys():
        tab = getBinaryPos(
            mjd, param, mjd0=param["mjd0"], revol=1, v=2, au=True, display=False
        )
        param["a"] = tab["a"]

    tab_orient, tab_faceon, typei, N2, r_nuc, step, alpha = _compute_param_elts_spec(
        mjd, param, verbose=verbose, display=display
    )

    dmas = rad2mas(np.sqrt(tab_faceon[2] ** 2 + tab_faceon[3] ** 2))
    Tin = param["T_sub"]
    q = param["q"]
    Tr = Tin * (dmas / rad2mas(r_nuc)) ** (-q)

    spec = []
    for xx in wl:
        l_Tr = Tr.copy()
        l_Tr[dmas >= np.cos(alpha / 2.0) * rad2mas(step)] /= param["gap_factor"]
        spectrumi = param["f_scale_pwhl"] * planck_law(l_Tr, xx)
        spec.append(spectrumi)

    wl_sed = np.logspace(-7, -3.5, 1000)
    spec_all, spec_all1, spec_all2 = [], [], []
    for i in range(len(l_Tr)):
        spectrum_r = param["f_scale_pwhl"] * planck_law(l_Tr[i], wl_sed)
        spec_all.append(spectrum_r)
        if dmas[i] >= np.cos(alpha / 2.0) * rad2mas(step):
            spec_all2.append(spectrum_r)
        else:
            spec_all1.append(spectrum_r)

    spec_all, spec_all1, spec_all2 = (
        np.array(spec_all),
        np.array(spec_all1),
        np.array(spec_all2),
    )

    total_sed = np.sum(spec_all, axis=0)
    wl_peak = wl_sed[np.argmax(total_sed)] * 1e6
    T_wien = 3000.0 / wl_peak

    spec = np.array(spec)
    flux0 = np.sum(spec[0, :])
    if display:
        plt.figure()
        plt.loglog(wl_sed * 1e6, spec_all1.T, color="grey")
        try:
            plt.loglog(wl_sed * 1e6, spec_all2.T, color="lightgrey")
        except ValueError:
            pass
        plt.loglog(
            wl_sed * 1e6,
            total_sed,
            color="#008080",
            lw=3,
            alpha=0.8,
            label=r"Total SED (T$_{wien}$ = %i K)" % T_wien,
        )
        plt.plot(-1, -1, "-", color="grey", lw=3, label="Illuminated dust")
        plt.plot(-1, -1, "-", color="lightgrey", lw=3, label="Shadowed dust")
        plt.ylim(total_sed.max() / 1e6, total_sed.max() * 1e1)
        plt.legend(loc=2, fontsize=9)
        plt.xlabel("Wavelength [µm]")
        plt.ylabel("Blackbodies flux [arbitrary unit]")

        plt.figure()
        plt.plot(dmas, l_Tr)
        plt.xlabel("Distance [mas]")
        plt.ylabel("Temperature [K]")

    if verbose:
        print("Temperature law: r0 = %2.2f mas, T0 = %i K" % (dmas[0], Tr[0]))

    return (
        np.array(spec) / N2,
        total_sed / N2,
        wl_sed * 1e6,
        T_wien,
        spec_all1.T / N2,
        spec_all2.T / N2,
        flux0 / N2,
        dmas,
        l_Tr,
    )


def visSpiralTemp(
    Utable,
    Vtable,
    Lambda,
    mjd,
    param,
    spec=None,
    fillFactor=20,
    verbose=True,
    display=True,
):
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

    # Start using the '_compute_param_elts_spec' function to determine the elements
    # parameters composing the spiral. Used to easely get these parameters and
    # determine the SED of the actual elements of the spiral.
    tab_orient, tab_faceon, typei, N2, r_nuc, step, alpha = _compute_param_elts_spec(
        mjd, param, verbose=verbose, display=display
    )

    x0, y0, x2, y2, fwhmx2, fwhmy2, angle2 = tab_orient
    list_param = []

    if (len(x2)) == 0:
        if verbose:
            cprint(
                "Warning: No dust in the pinwheel given the parameters fov/rnuc/rs.",
                "red",
            )
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

    if (param["q"] == 1.0) or (spec is None):
        spectrumi = list(np.linspace(1, 0, N2))
    else:
        dmas = rad2mas(np.sqrt(tab_faceon[2] ** 2 + tab_faceon[3] ** 2))
        Tin = param["T_sub"]
        q = param["q"]
        Tr = Tin * (dmas / rad2mas(r_nuc)) ** (-q)

        if verbose:
            print("Temperature law: r0 = %2.2f mas, T0 = %i K" % (dmas[0], Tr[0]))
        spectrumi = spec

    C_centered = visMultipleResolved(
        Utable, Vtable, Lambda, typei, spectrumi, list_param
    )
    # print(C_centered.shape)
    x0, y0 = mas2rad(param["x0"]), mas2rad(param["y0"])
    C = shiftFourier(Utable, Vtable, Lambda, C_centered, x0, y0)
    return C


def visMultipleResolved(Utable, Vtable, Lambda, typei, spec, list_param):
    """Compute the complex visibility of a multi-component object."""
    n_obj = len(typei)
    if type(Utable) == np.float64:
        nbl = 1
    else:
        nbl = len(Utable)
    if type(Lambda) == np.float64:
        nwl = 1
    else:
        nwl = len(Lambda)
    corrFluxTmp = np.zeros([n_obj, nwl, nbl], dtype=complex)

    for i in range(n_obj):
        if typei[i] == "r":
            Ci = visEllipticalRing(Utable, Vtable, Lambda, list_param[i])
        elif typei[i] == "t_r":
            Ci = visThickEllipticalRing(Utable, Vtable, Lambda, list_param[i])
        elif typei[i] == "ud":
            Ci = visUniformDisk(Utable, Vtable, Lambda, list_param[i])
        elif typei[i] == "e_ud":
            Ci = visEllipticalUniformDisk(Utable, Vtable, Lambda, list_param[i])
        elif typei[i] == "gd":
            Ci = visGaussianDisk(Utable, Vtable, Lambda, list_param[i])
        elif typei[i] == "e_gd":
            Ci = visEllipticalGaussianDisk(Utable, Vtable, Lambda, list_param[i])
        elif typei[i] == "star":
            Ci = visPointSource(Utable, Vtable, Lambda, list_param[i])
        elif typei[i] == "pwhl":
            Ci = visSpiralTemp(Utable, Vtable, Lambda, list_param[i])
        else:
            print("Model not yet in VisModels")
        spec2 = spec[i]
        corrFluxTmp[i, :, :] = spec2 * Ci
        # Ci2 = Ci.reshape(1, np.size(Ci))
        # corrFluxTmp += np.dot(spec2, Ci2)
        # corrFluxTmp += Ci
    corrFlux = corrFluxTmp.sum(axis=0)
    flux = np.sum(spec, 0)
    try:
        vis = corrFlux / flux
    except Exception:
        print("Error : flux = 0")
        vis = None
    return vis


def visPwhl(Utable, Vtable, Lambda, param, verbose=False, expert_plot=False):
    """
    Compute a multi-component model of a pinwheel nebulae with:
    - The pinwheel: composed of a multiple rings, uniform disk or gaussian (`compo`). The shape of the
    pinwheel is given by the `step`, the `opening_angle`, etc. The fluxes are computed
    using a blackbodies fallowing a power law temperature (`T_sub`, `r_nuc`, and `q`),
    - The binary star: computed using binary_integrator package (stellar parameters as `M1`, `M2`, `e`, and
    `dpc` are required). If not, specify a separation (`s_bin`) to compute your own binary position.
    The binary relative flux (set by `contrib_star` in [%]) is computed @ 1 µm using
    blackbodies `T_WR`, `T_OB` and the SED of the pinwheel,
    - The halo (Optionnal): The pinwheel appeared to be surrounded by a fully
    resolved environment (similar to YSO halo properties, see Lazareff et al.
    2017). This contribution is set by `contrib_halo` [%], where the flux is
    taken from the spiral contribution.
    """

    mjd = param["mjd"]  # 50000.0
    phase = (mjd - param["mjd0"]) / param["P"] % 1
    if verbose:
        s = "Model pinwheel S = {:2.1f} mas, phase = {:1.2f} @ {:2.2f} µm:".format(
            param["step"],
            phase,
            Lambda * 1e6,
        )
        cprint(s, "cyan")
        cprint("-" * len(s), "cyan")

    angle_0 = param["angle_0"]
    angle_0_bis = (angle_0 - 0) * -1

    # -= 90  # Switch reference to the NORTH
    param["angle_0"] = angle_0_bis

    param["incl"] = param["incl"] + 180
    # Binary point source
    # --------------------------------------------------------------------------
    if ("M1" in param.keys()) & ("M2" in param.keys()):
        tab = getBinaryPos(
            mjd, param, mjd0=param["mjd0"], revol=1, v=2, au=True, display=expert_plot
        )
        param["a"] = tab["a"]
        param_star_WR = {
            "x0": mas2rad(tab["star1"]["x"]),
            "y0": mas2rad(tab["star1"]["y"]),
        }
        param_star_O = {
            "x0": mas2rad(tab["star2"]["x"]),
            "y0": mas2rad(tab["star2"]["y"]),
        }
    else:
        param["a"] = param["sep_bin"] * param["dpc"]
        param_star_WR = {
            "x0": 0,
            "y0": 0,
        }
        param_star_O = {
            "x0": 0,
            "y0": 0,
        }

    # Flux contribution of the different components (binary star, pinwheel and
    # resolved environment)
    # --------------------------------------------------------------------------
    wl_0 = 1e-6  # Wavelength 0 for the ratio

    # Contribution of each star in the binary system
    p_OB, p_WR = computeBinaryRatio(param, Lambda)
    p_OB = np.mean(p_OB)
    p_WR = np.mean(p_WR)
    if verbose:
        print(
            "Binary relative fluxes: WR = %2.2f %%, OB = %2.2f %%"
            % (p_WR * 100, p_OB * 100.0)
        )

    contrib_star = param["contrib_star"] / 100.0

    if type(Lambda) is float:
        wl = np.array([Lambda])
    else:
        wl = Lambda

    wl_sed = np.logspace(-7, -3.5, 1000)
    if param["r_nuc"] != 0:
        input_wl = [wl_0] + list(wl)
        tab_dust_fluxes = sed_pwhl(input_wl, mjd, param, display=False, verbose=False)
        full_sed = tab_dust_fluxes[1]
        wl_sed = tab_dust_fluxes[2]
        f_pinwheel_wl0 = tab_dust_fluxes[6]

        f_binary_wl0 = p_OB * planck_law(param["T_OB"], wl_0) + p_WR * planck_law(
            param["T_WR"], wl_0
        )

        f_binary_wl = p_OB * planck_law(
            param["T_OB"], wl_sed / 1e6
        ) + p_WR * planck_law(param["T_WR"], wl_sed / 1e6)

        f_binary_obs = p_OB * planck_law(param["T_OB"], wl) + p_WR * planck_law(
            param["T_WR"], wl
        )

    if param["r_nuc"] != 0:
        sed_pwhl_wl = tab_dust_fluxes[0][1, :]
        P_dust = np.sum(sed_pwhl_wl)
        n_elts = len(sed_pwhl_wl)

        if contrib_star != 1:
            scale_star = (f_pinwheel_wl0 / f_binary_wl0) * (
                contrib_star / (1.0 - contrib_star)
            )
        else:
            scale_star = 1e6
    else:
        P_dust = 1 - param["contrib_star"] / 100.0
        n_elts = param["nelts"]
        sed_pwhl_wl = None

    wl_m = wl * 1e6
    l_wl = [wl_m] * n_elts

    if param["r_nuc"] != 0:
        binary_sed = scale_star * f_binary_wl
        P_star = scale_star * f_binary_obs
        dmas = tab_dust_fluxes[7]
        Tr = tab_dust_fluxes[8]
        Twien = tab_dust_fluxes[3]
    else:
        P_star = param["contrib_star"] / 100.0

    if (expert_plot) & (param["r_nuc"] != 0):
        plt.figure()
        plt.plot(dmas, Tr, label=f"r0 = {dmas[0]:2.1f} mas, T0 = {Tr[0]:2.1f} K")
        plt.grid(alpha=0.2)
        plt.xlabel("Distance [mas]")
        plt.ylabel("Temperature [K]")
        plt.legend()
        plt.tight_layout()

        plt.figure()
        plt.loglog(
            wl_sed,
            full_sed,
            color="#af6d04",
            lw=3,
            alpha=0.8,
            label=r"Pinwheel (T$_{wien}$ = %i K)" % Twien,
        )
        plt.loglog(
            wl_sed,
            binary_sed,
            color="#008080",
            lw=3,
            alpha=0.8,
            label="Binary",
            zorder=3,
        )
        plt.loglog(l_wl, sed_pwhl_wl, ".", ms=3, color="#222223", zorder=10)
        plt.loglog(wl_sed, tab_dust_fluxes[4], "-", color="grey")
        try:
            plt.loglog(wl_sed, tab_dust_fluxes[5], "-", color="lightgrey")
        except ValueError:
            pass

        max_plot = full_sed.max()
        plt.ylim(max_plot / 1e6, max_plot * 1e2)
        plt.plot(-1, -1, "-", color="grey", lw=3, label="Illuminated dust")  # legend
        plt.plot(-1, -1, "-", color="lightgrey", lw=3, label="Shadowed dust")  # legend
        plt.loglog(
            wl_m,
            P_dust,
            "H",
            color="#71490a",
            zorder=5,
            ms=5,
            markeredgecolor="k",
            markeredgewidth=0.5,
            # label=f"$\Sigma F_{{{wl_m:2.1f}µm}}$ = {P_dust:2.1f} Jy",
        )
        plt.loglog(
            wl_m,
            P_star,
            "H",
            color="#009ace",
            zorder=5,
            ms=5,
            markeredgecolor="k",
            markeredgewidth=0.5,
            # label=f"$\Sigma F_{{*, {wl_m:2.1f}µm}}$ = {P_star:2.1f} Jy",
        )
        plt.xlabel("Wavelength [µm]")
        plt.ylabel("SED [Jy]")
        plt.legend(loc=1, fontsize=8)
        plt.grid(alpha=0.1, which="both")
        plt.tight_layout()

    if contrib_star == 1:
        P_star = 1
        P_dust = 0

    P_tot = P_star + P_dust

    # # Different contributions
    Fstar = P_star / P_tot
    Fpwhl = P_dust / P_tot

    # Gaussian disk background
    # --------------------------------------------------------------------------
    if "contrib_halo" in list(param.keys()):
        Fshell = param["contrib_halo"] / 100.0
        Fpwhl -= Fshell

    if verbose:
        print(
            "Relative component fluxes: Fstar = %2.3f %%; Fpwhl = %2.3f %%, Fenv = %2.3f %%"
            % (100 * Fstar, 100 * Fpwhl, 100 * Fshell)
        )

    # # Visibility
    # --------------------------------------------------------------------------
    param["x0"] = rad2mas(param_star_O["x0"]) * (2 / 3.0)
    param["y0"] = rad2mas(param_star_O["y0"]) * (2 / 3.0)

    thickness = param["thickness"]
    Vpwhl = visSpiralTemp(
        Utable,
        Vtable,
        wl,
        mjd,
        param,
        spec=sed_pwhl_wl,
        verbose=verbose,
        display=expert_plot,
    ) * visGaussianDisk(
        Utable, Vtable, Lambda, {"fwhm": thickness, "x0": 0.0, "y0": 0.0}
    )

    vis_OB = p_OB * Fstar[:, None] * visPointSource(Utable, Vtable, wl, param_star_O)
    vis_WR = p_WR * Fstar[:, None] * visPointSource(Utable, Vtable, wl, param_star_WR)

    vis = Fpwhl[:, None] * Vpwhl + vis_OB + vis_WR

    return vis
