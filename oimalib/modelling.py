"""
@author: Anthony Soulain (University of Sydney)
-----------------------------------------------------------------
OIMALIB: Optical Interferometry Modelisation and Analysis Library
-----------------------------------------------------------------

Set of function to extract complex visibilities from fits image/cube
or geometrical model.
-----------------------------------------------------------------
"""
import sys
import time
from pathlib import Path

import numpy as np
from astropy.io import fits
from munch import munchify as dict2class
from scipy.interpolate import interp1d
from scipy.interpolate import interp2d
from scipy.interpolate import RegularGridInterpolator as regip
from scipy.ndimage import gaussian_filter1d
from scipy.ndimage import rotate
from termcolor import cprint

from oimalib.fitting import check_params_model
from oimalib.fitting import comput_CP
from oimalib.fitting import comput_V2
from oimalib.fitting import select_model
from oimalib.tools import mas2rad
from oimalib.tools import rad2mas


def _print_info_model(wl_model, modelfile, fov, npix, s):
    nwl = len(wl_model)
    if nwl == 1:
        modeltype = "image"
    else:
        modeltype = "cube"
    try:
        fname = modelfile.split("/")[-1]
    except AttributeError:
        fname = modelfile.name
    title = f"Model grid from {modeltype} ({fname})"
    cprint(title, "cyan")
    cprint("-" * len(title), "cyan")
    print(
        "fov=%2.2f mas, npix=%i (%i padded), pix=%2.3f mas"
        % (fov, npix, s[2], fov / npix)
    )
    if nwl == 1:
        cprint("nwl=%i (%2.1f µm)" % (nwl, np.mean(wl_model) * 1e6), "green")
    else:
        wl1 = wl_model[0] * 1e6
        wl2 = wl_model[-1] * 1e6
        wl_step = np.diff(wl_model)[0] * 1e6
        cprint(
            "nwl=%i (wl0=%2.3f, wlmax=%2.3f µm, step=%2.3f nm)"
            % (nwl, wl1, wl2, wl_step * 1000.0),
            "green",
        )
    cprint("-" * len(title) + "\n", "cyan")


def model2grid(
    modelfile,
    wl_user=None,
    pix_user=None,
    rotation=0,
    scale=1,
    fliplr=False,
    pad_fact=2,
    method="linear",
    i1=0,
    i2=None,
    light=False,
):
    """Compute grid class from model as fits file cube.

    Parameters
    ----------
    `modelfile` {str}:
        Name of the model (path),\n
    `wl_user` {array}:
        If not found in the header, wavelength array is required [µm],\n
    `rotation` {int};
        Angle to rotate the model [deg], by default 0,\n
    `scale` {int}:
        Scale the model pixel size, by default 1,\n
    `pad_fact` {int}:
        Padding factor, by default 2.\n

    Returns
    -------
    `grid` {class}:
        class like containing model with keys:\n
            - 'real': 3-d interpolated real part of the complex vis,\n
            - 'imag': 3-d interpolated imaginmary part of the complex vis,\n
            - 'wl': Wavelength grid [m],\n
            - 'freq': Frequencies vector [m-1],\n
            - 'fov': Model field of view [mas],\n
            - 'cube': datacube model,\n
            - 'fft': 2-d fft of the cube,\n
            - 'name': file name of the model.\n
    """
    hdu = fits.open(modelfile)
    hdr = hdu[0].header

    n_wl = hdr.get("NAXIS3", 1)
    delta_wl = hdr.get("CDELT3", 0)

    if wl_user is not None:
        wl0 = wl_user
        if type(wl_user) is float or np.float64:
            n_wl = 1
        elif type(wl_user) is list:
            n_wl = len(wl_user)
        else:
            cprint("wl_user have a wrong format: need float or list.", "red")
            return 0
    else:
        wl0 = hdr.get("CRVAL3", None)
        if wl0 is None:
            wl0 = hdr.get("WLEN0", None)

        if wl0 is None:
            if wl_user is None:
                cprint("Wavelength not found: need wl_user [µm].", "red")
                return None
            else:
                wl0 = wl_user
                print(
                    "Wavelenght not found: argument wl_user (%2.1f) is used)." % wl_user
                )

    try:
        wl0 = float(wl0)
    except TypeError:
        pass
    npix = hdr["NAXIS1"]

    try:
        unit = hdr["CUNIT1"]
    except KeyError:
        unit = None

    try:
        if unit is None:
            if "deg" in (hdr.comments["CDELT1"]):
                unit = "deg"
            else:
                unit = "rad"
        if unit == "rad":
            pix_size = abs(hdr["CDELT1"]) * scale
        elif unit == "deg":
            pix_size = np.deg2rad(abs(hdr["CDELT1"])) * scale
        else:
            print("Wrong unit in CDELT1 header.")
    except KeyError:
        unit = "rad"
        if pix_user is None:
            cprint(
                "Error: Pixel size not found, please give pix_user [mas].",
                "red",
                file=sys.stderr,
            )
            return None
        pix_size = mas2rad(pix_user) * scale

    fov = rad2mas(npix * pix_size)

    if n_wl == 1:
        wl_model = np.array([wl0])
    else:
        wl_model = np.linspace(wl0, wl0 + delta_wl * (n_wl - 1), n_wl)

    unit_wl = hdr.get("CUNIT3", None)

    if unit_wl != "m":
        wl_model *= 1e-6

    if wl_user is not None:
        wl_model = wl_user / 1e6

    padding = pad_fact * np.array([npix // 2, npix // 2])
    padding = padding.astype(int)
    if len(hdu[0].data) != 1:
        image_input = hdu[0].data[i1:i2]
    else:
        image_input = hdu[0].data

    flux = []
    for _ in image_input:
        flux.append(_.sum())
    flux = np.array(flux) / image_input[0].sum()

    axes_rot = (1, 2)
    n_image = image_input.shape[0]
    if len(image_input.shape) == 2:
        image_input = image_input.reshape(
            [1, image_input.shape[0], image_input.shape[1]]
        )
        n_image = 1
    mod = rotate(image_input, rotation, axes=axes_rot, reshape=False)

    model_aligned = mod.copy()
    if fliplr:
        model_aligned = np.fliplr(model_aligned)

    mod_pad = np.pad(model_aligned, pad_width=((0, 0), padding, padding), mode="edge")

    if mod_pad.shape[1] % 2 == 0:
        mod_pad = mod_pad[:, :-1, :-1]

    mod_pad = mod_pad / np.max(mod_pad)

    s = np.shape(mod_pad)

    fft2D = np.fft.fftshift(
        np.fft.fft2(np.fft.fftshift(mod_pad, axes=[2, 1]), axes=[-2, -1]),
        axes=[2, 1],
    )
    maxi = np.max(np.abs(fft2D), axis=(1, 2))

    for i in range(n_image):
        fft2D[i, :, :] = fft2D[i, :, :] / maxi[i]

    freqVect = np.fft.fftshift(np.fft.fftfreq(s[2], pix_size))

    _print_info_model(wl_model, modelfile, fov, npix, s)
    if n_wl == 1:
        im3d_real = interp2d(freqVect, freqVect, fft2D.real, kind="cubic")
        im3d_imag = interp2d(freqVect, freqVect, fft2D.imag, kind="cubic")
    else:
        if method == "linear":
            im3d_real = regip(
                (wl_model, freqVect, freqVect),
                [x.T for x in fft2D.real],
                method="linear",
                bounds_error=False,
                fill_value=np.nan,
            )
            im3d_imag = regip(
                (wl_model, freqVect, freqVect),
                [x.T for x in fft2D.imag],
                method="linear",
                bounds_error=False,
                fill_value=np.nan,
            )
        else:
            print("Not implemented yet.")
            return None

    p = Path(modelfile)
    modelname = p.stem

    try:
        sign = np.sign(hdr["CDELT1"])
    except KeyError:
        sign = 1.0
    if light:
        grid = {
            "real": im3d_real,
            "imag": im3d_imag,
            "sign": sign,
            "pad_fact": pad_fact,
            "flux": flux,
        }
    else:
        grid = {
            "real": im3d_real,
            "imag": im3d_imag,
            "sign": sign,
            "wl": wl_model,
            "freq": freqVect,
            "fov": fov,
            "cube": model_aligned,
            "fft": fft2D,
            "name": modelname,
            "pad_fact": pad_fact,
            "flux": flux,
        }
    hdu.close()
    return dict2class(grid)


def _compute_grid_model_chromatic(data, grid, verbose=False):
    nbl = len(data.u)
    ncp = len(data.cp)
    nwl = len(data.wl)

    greal, gimag = grid.real, grid.imag

    if type(data) is not list:
        l_data = [data]
    else:
        l_data = data
    start_time = time.time()
    l_mod_v2, l_mod_cp = [], []

    for data in l_data:
        mod_v2 = np.zeros([nbl, nwl])
        for i in range(nbl):
            um, vm = data.u[i], data.v[i]
            for j in range(nwl):
                wl = data.wl[j]
                x = grid.sign * um / wl
                y = vm / wl
                pts = (wl, x, y)
                if not data.flag_vis2[i][j]:
                    v2 = abs(greal(pts) + 1j * gimag(pts)) ** 2
                else:
                    v2 = np.nan
                mod_v2[i, j] = v2

        mod_cp = np.zeros([ncp, nwl])
        for i in range(ncp):
            u1, u2, u3 = (
                grid.sign * data.u1[i],
                grid.sign * data.u2[i],
                grid.sign * data.u3[i],
            )
            v1, v2, v3 = data.v1[i], data.v2[i], data.v3[i]
            for j in range(nwl):
                wl = data.wl[j]
                u1m, u2m, u3m = u1 / wl, u2 / wl, u3 / wl
                v1m, v2m, v3m = v1 / wl, v2 / wl, v3 / wl
                if not data.flag_cp[i][j]:
                    cvis_1 = greal([wl, u1m, v1m]) + 1j * gimag([wl, u1m, v1m])
                    cvis_2 = greal([wl, u2m, v2m]) + 1j * gimag([wl, u2m, v2m])
                    cvis_3 = greal([wl, u3m, v3m]) + 1j * gimag([wl, u3m, v3m])
                    bispec = np.array(cvis_1) * np.array(cvis_2) * np.array(cvis_3)
                    cp = np.rad2deg(np.arctan2(bispec.imag, bispec.real))
                else:
                    cp = np.nan
                mod_cp[i, j] = cp
        l_mod_cp.append(mod_cp)
        l_mod_v2.append(mod_v2)

    if verbose:
        print("Execution time compute_grid_model: %2.3f s" % (time.time() - start_time))
    return l_mod_v2, l_mod_cp


def _compute_grid_model_nochromatic(data, grid, verbose=False):
    starttime = time.time()
    nbl = len(data.u)
    ncp = len(data.cp)
    nwl = len(data.wl)

    greal, gimag = grid.real, grid.imag

    if type(data) is not list:
        l_data = [data]
    else:
        l_data = data
    # start_time = time.time()

    l_mod_v2, l_mod_cp = [], []

    for data in l_data:
        mod_v2 = np.zeros([nbl, nwl])
        for i in range(nbl):
            um = grid.sign * data.u[i] / data.wl
            vm = data.v[i] / data.wl
            mod_v2[i] = [
                abs(grid.real(um[j], vm[j])[0] + 1j * grid.imag(um[j], vm[j])[0]) ** 2
                for j in range(nwl)
            ]

        mod_cp = np.zeros([ncp, nwl])
        for i in range(ncp):
            u1, u2, u3 = (
                grid.sign * data.u1[i],
                grid.sign * data.u2[i],
                grid.sign * data.u3[i],
            )
            v1, v2, v3 = data.v1[i], data.v2[i], data.v3[i]
            u1m, u2m, u3m = u1 / data.wl, u2 / data.wl, u3 / data.wl
            v1m, v2m, v3m = v1 / data.wl, v2 / data.wl, v3 / data.wl
            greal, gimag = grid.real, grid.imag
            cvis_1 = [
                greal(u1m[i], v1m[i]) + 1j * gimag(u1m[i], v1m[i]) for i in range(nwl)
            ]
            cvis_2 = [
                greal(u2m[i], v2m[i]) + 1j * gimag(u2m[i], v2m[i]) for i in range(nwl)
            ]
            cvis_3 = [
                greal(u3m[i], v3m[i]) + 1j * gimag(u3m[i], v3m[i]) for i in range(nwl)
            ]
            bispec = np.array(cvis_1) * np.array(cvis_2) * np.array(cvis_3)
            cp = np.rad2deg(np.arctan2(bispec.imag, bispec.real))
            mod_cp[i] = np.squeeze(cp)
        l_mod_cp.append(mod_cp)
        l_mod_v2.append(mod_v2)
    if verbose:
        print("Execution time compute_grid_model: %2.3f s" % (time.time() - starttime))
    return l_mod_v2, l_mod_cp


def compute_grid_model(data, grid, verbose=False):
    nwl = len(grid.wl)
    if nwl == 1:
        mod_v2, mod_cp = _compute_grid_model_nochromatic(data, grid, verbose=verbose)
    else:
        mod_v2, mod_cp = _compute_grid_model_chromatic(data, grid, verbose=verbose)
    return mod_v2, mod_cp


def compute_geom_model(data, param, verbose=False):
    if type(data) is not list:
        l_data = [data]
    else:
        l_data = data
    start_time = time.time()
    l_mod_v2, l_mod_cp = [], []
    for data in l_data:
        model_target = select_model(param["model"])
        isValid, log = check_params_model(param)
        if not isValid:
            cprint("\nWrong input parameters for %s model:" % (param["model"]), "green")
            print(log)
            cprint(
                "-" * len("Wrong input parameters for %s model." % (param["model"]))
                + "\n",
                "green",
            )
            return None, None

        nbl = len(data.u)
        ncp = len(data.cp)

        mod_v2 = np.zeros_like(data.vis2)
        for i in range(nbl):
            u, v, wl = data.u[i], data.v[i], data.wl
            mod = comput_V2([u, v, wl], param, model_target)
            mod_v2[i, :] = mod

        mod_cp = np.zeros_like(data.cp)
        for i in range(ncp):
            u1, u2, u3 = data.u1[i], data.u2[i], data.u3[i]
            v1, v2, v3 = data.v1[i], data.v2[i], data.v3[i]
            wl2 = data.wl
            X = [u1, u2, u3, v1, v2, v3, wl2]
            tmp = comput_CP(X, param, model_target)
            mod_cp[i, :] = tmp

        l_mod_cp.append(mod_cp)
        l_mod_v2.append(mod_v2)

    if verbose:
        print("Execution time compute_geom_model: %2.3f s" % (time.time() - start_time))

    return l_mod_v2, l_mod_cp


def decoratortimer(decimal):
    def decoratorfunction(f):
        def wrap(*args, **kwargs):
            time1 = time.monotonic()
            result = f(*args, **kwargs)
            time2 = time.monotonic()
            print(
                "{:s} function took {:.{}f} ms".format(
                    f.__name__, ((time2 - time1) * 1000.0), decimal
                )
            )
            return result

        return wrap

    return decoratorfunction


@decoratortimer(2)
def _compute_dobs_grid_bl(
    ibl,
    d,
    grid,
    obs_wl=True,
    w_line=0.0005,
    p_line=2.166128,
    scale=False,
):
    """Compute the differential observable (dvis, dphi) from the
    pre-computed interpolated grid (from chromatic models).
    """
    # Extract u-v coordinates from data at specific baseline (ibl)
    um = d.u[ibl]
    vm = d.v[ibl]

    if obs_wl:
        wlm = d.wl
    else:
        wlm = np.linspace(d.wl[0], d.wl[-1], 100)

    # Interpolated function (real and imaginary parts)
    greal, gimag = grid.real, grid.imag

    # Extract the specific points (from the data)
    pts = (wlm, um / wlm, vm / wlm)
    cvis = greal(pts) + 1j * gimag(pts)

    # Compute the reference vis and phase from the continuum
    cont = np.abs(wlm * 1e6 - p_line) >= 2 * w_line
    vis_ref = np.mean(abs(cvis[cont & ~np.isnan(cvis)]))
    phi_ref = np.rad2deg(np.mean(np.angle(cvis[cont & ~np.isnan(cvis)])))

    # Compute observables
    dvis = abs(cvis) / vis_ref
    dphi = np.rad2deg(np.angle(cvis)) - phi_ref
    vis2 = abs(cvis) ** 2

    # Compute flux on the data resolution
    res_obs = np.diff(d.wl).mean()
    res_model = np.diff(grid.wl).mean()
    scale_resol = (res_obs / res_model) / 2.355
    if scale:
        flux_scale = gaussian_filter1d(grid.flux, sigma=scale_resol)
    else:
        flux_scale = grid.flux
    fct_flux = interp1d(grid.wl, flux_scale, kind="cubic", bounds_error=False)
    flux_obs = fct_flux(wlm)

    output = {
        "cvis": cvis,
        "dvis": dvis,
        "dphi": dphi,
        "vis2": vis2,
        "cont": cont,
        "wl": wlm * 1e6,
        "p_line": p_line,
        "flux": flux_obs,
        "flux_model": grid.flux,
        "wl_model": grid.wl * 1e6,
        "ibl": ibl,
    }

    return dict2class(output)
