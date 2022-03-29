"""
@author: Anthony Soulain (University of Sydney)
-----------------------------------------------------------------
OIMALIB: Optical Interferometry Modelisation and Analysis Library
-----------------------------------------------------------------

Set of function to extract complex visibilities from fits image/cube
or geometrical model.
-----------------------------------------------------------------
"""
import multiprocessing
import sys
import time
from functools import partial
from pathlib import Path

import numpy as np
from astropy.io import fits
from munch import munchify as dict2class
from scipy import fft
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
from oimalib.fourier import UVGrid
from oimalib.tools import mas2rad
from oimalib.tools import rad2arcsec
from oimalib.tools import rad2mas


def _print_info_model(wl_model, modelfile, fov, npix, s, starttime):
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
    pixsize = fov / npix
    print(
        "fov=%2.2f mas, npix=%i (%i padded, equivalent %2.2f mas), pix=%2.3f mas"
        % (fov, npix, s[2], s[2] * pixsize, pixsize)
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
    print("Computation time = %2.2f s" % (time.time() - starttime))
    cprint("-" * len(title) + "\n", "cyan")


def _construct_ft_arr(cube, ncore=8):
    """Open the model cube and perform a series of roll (both axis) to avoid grid artefact
    (negative fft values).

    Parameters:
    -----------
    `cube` {array}: padded model cube.

    Returns:
    --------
    `ft_arr` {array}: complex array of the Fourier transform of the cube,\n
    `n_ps` {int}: Number of frames,\n
    `n_pix` {int}: Dimensions of one frames,\n

    """
    n_pix = cube.shape[1]
    cube = np.roll(np.roll(cube, n_pix // 2, axis=1), n_pix // 2, axis=2)

    ft_arr = fft.fft2(cube, workers=ncore)

    i_ps = ft_arr.shape
    n_ps = i_ps[0]

    return ft_arr, n_ps, n_pix


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
    ncore=1,
    window=None,
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
    starttime = time.time()

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

    from oimalib.tools import apply_windowing

    # mod_pad = np.array([apply_windowing(x, window=npix//2) for x in mod_pad])
    mod_pad[mod_pad < 1e-20] = 1e-50
    s = np.shape(mod_pad)

    fft2D, n_ps, n_pix = _construct_ft_arr(mod_pad, ncore=ncore)
    fft2D = np.roll(fft2D, n_pix // 2, axis=1)
    fft2D = np.roll(fft2D, n_pix // 2, axis=2)

    maxi = np.max(np.abs(fft2D), axis=(1, 2))

    for i in range(n_image):
        fft2D[i, :, :] = fft2D[i, :, :] / maxi[i]

    freqVect = np.fft.fftshift(np.fft.fftfreq(n_pix, pix_size))
    _print_info_model(wl_model, modelfile, fov, npix, s, starttime)

    fft2d_real = fft2D.real
    fft2d_imag = fft2D.imag
    if window is not None:
        fft2d_real = np.array([apply_windowing(x, window) for x in fft2d_real])
        fft2d_imag = np.array([apply_windowing(x, window) for x in fft2d_imag])

    if n_wl == 1:
        im3d_real = interp2d(freqVect, freqVect, fft2D.real, kind="cubic")
        im3d_imag = interp2d(freqVect, freqVect, fft2D.imag, kind="cubic")
    else:
        if method == "linear":
            im3d_real = regip(
                (wl_model, freqVect, freqVect),
                [x.T for x in fft2d_real],
                method="linear",
                bounds_error=False,
                fill_value=np.nan,
            )
            im3d_imag = regip(
                (wl_model, freqVect, freqVect),
                [x.T for x in fft2d_imag],
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
            "npix": s[1],
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
    """Compute interferometric observables baseline per baseline
    and for all wavelengths (slow)."""
    start_time = time.time()
    if type(data) is not list:
        l_data = [data]
    else:
        l_data = data
    start_time = time.time()
    l_mod_v2, l_mod_cp = [], []
    k = 0
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
            mod_v2[i, :] = np.squeeze(mod)

        mod_cp = np.zeros_like(data.cp)
        for i in range(ncp):
            u1, u2, u3 = data.u1[i], data.u2[i], data.u3[i]
            v1, v2, v3 = data.v1[i], data.v2[i], data.v3[i]
            wl2 = data.wl
            X = [u1, u2, u3, v1, v2, v3, wl2]
            tmp = comput_CP(X, param, model_target)
            mod_cp[i, :] = np.squeeze(tmp)

        l_mod_cp.append(mod_cp)
        l_mod_v2.append(mod_v2)
        k += 1

    if verbose:
        print("Execution time compute_geom_model: %2.3f s" % (time.time() - start_time))
    return l_mod_v2, l_mod_cp


def _compute_geom_model_ind(dataset, param, verbose=False):
    """Compute interferometric observables all at once (including all spectral
    channels) by using matrix computation. `dataset` corresponds to an individual
    fits file (from oimalib.load())."""
    startime = time.time()
    Utable = dataset.u
    Vtable = dataset.v
    Lambda = dataset.wl
    nobs = len(Utable) * len(Lambda)
    model_target = select_model(param["model"])
    isValid, log = check_params_model(param)

    if not isValid:
        cprint("\nWrong input parameters for %s model:" % (param["model"]), "green")
        print(log)
        cprint(
            "-" * len("Wrong input parameters for %s model." % (param["model"])) + "\n",
            "green",
        )
        return None
    # Compute complex visibility (for nbl baselines)
    # .T added to be in the same order as data (i.e.: [nbl, nwl])
    cvis = model_target(Utable, Vtable, Lambda, param).T

    vis2 = np.abs(cvis) ** 2
    vis_amp = np.abs(cvis)
    vis_phi = np.angle(cvis)

    # Compute bispectrum and closure phases (same for .T)
    u1, u2, u3 = dataset.u1, dataset.u2, dataset.u3
    v1, v2, v3 = dataset.v1, dataset.v2, dataset.v3
    V1 = model_target(u1, v1, Lambda, param)
    V2 = model_target(u2, v2, Lambda, param)
    V3 = model_target(u3, v3, Lambda, param)
    bispectrum = V1 * V2 * V3
    cp = np.rad2deg(np.arctan2(bispectrum.imag, bispectrum.real)).T
    endtime = time.time()
    if verbose:
        print("Time to compute %i points = %2.2f s" % (nobs, endtime - startime))
    mod = {"vis2": vis2, "cp": cp, "dvis": vis_amp, "dphi": vis_phi}
    return mod


def compute_geom_model_fast(data, param, ncore=1, verbose=False):
    """Compute interferometric observables using the matrix method (faster)
    for a list of data (type(data) == list) or only one file (type(data) ==
    dict). The multiple dataset can be computed in parallel if `ncore` > 1."""
    if type(data) is not list:
        l_data = [data]
    else:
        l_data = data
    start_time = time.time()
    pool = multiprocessing.Pool(processes=ncore)
    prod = partial(_compute_geom_model_ind, param=param)
    result_list = pool.map(prod, l_data)
    pool.close()
    etime = time.time() - start_time
    if verbose:
        print("Execution time compute_geom_model_fast: %2.3f s" % etime)
    return result_list


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
    scale_resol = res_obs / res_model
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


def combine_grid_geom_model_image(
    wl, grid, param, ampli_factor=1, fh=0, fc=0, fmag=1, fov=3, npts=256
):
    fov = mas2rad(fov)
    bmax = (wl / fov) * npts
    maxX = rad2mas(wl * npts / bmax) / 2.0
    xScales = np.linspace(0, 2 * maxX, npts) - maxX
    pixel_size = rad2mas(fov) / npts
    extent_ima = (
        np.array((xScales.max(), xScales.min(), xScales.min(), xScales.max()))
        + pixel_size / 2.0
    )
    pixel_size = rad2mas(fov) / npts  # Pixel size of the image [mas]
    # # Creat UV coord
    UVTable = UVGrid(bmax, npts) / 2.0  # Factor 2 due to the fft
    Utable = UVTable[:, 0]
    Vtable = UVTable[:, 1]

    model_target = select_model(param["model"])

    vis_disk = model_target(Utable, Vtable, wl, param)

    greal, gimag = grid.real, grid.imag
    # Extract the specific points (from the data)
    pts = (wl, Utable / wl, Vtable / wl)
    vis_mag = greal(pts) + 1j * gimag(pts)

    index_image = np.abs(grid.wl - wl).argmin()

    # Amplify spectral line (mimic temperature increase)

    fs = 1 - fh - fc

    mm = grid.flux.copy() - 1
    mm[mm > 0] = mm[mm > 0] * ampli_factor
    mm += 1
    fmag = fs * mm
    ftot = fmag + fh + fc
    ftot = gaussian_filter1d(ftot, sigma=23.0 / 2.355)

    fmag_im = fmag[index_image]
    ftot_im = ftot[index_image]

    vis = (fmag_im * vis_mag + fc * vis_disk) / (ftot_im)

    print(
        "Magnetosphere contribution = %2.1f %% (lcr = %2.2f)"
        % (100 * fmag_im / (fmag_im + fh + fc), mm[index_image])
    )
    fwhm_apod = 5e4
    # Apodisation
    x, y = np.meshgrid(range(npts), range(npts))
    freq_max = rad2arcsec(bmax / wl) / 2.0
    pix_vis = 2 * freq_max / npts
    freq_map = np.sqrt((x - (npts / 2.0)) ** 2 + (y - (npts / 2.0)) ** 2) * pix_vis

    x = np.squeeze(np.linspace(0, 1.5 * np.sqrt(freq_max ** 2 + freq_max ** 2), npts))
    y = np.squeeze(np.exp(-(x ** 2) / (2 * (fwhm_apod / 2.355) ** 2)))

    f = interp1d(x, y)
    img_apod = f(freq_map.flat).reshape(freq_map.shape)

    im_vis = vis.reshape(npts, -1) * img_apod
    fftVis = np.fft.ifft2(im_vis)
    image = np.fft.fftshift(abs(fftVis))
    tmp = np.fliplr(image)
    image_orient = tmp  # / np.max(tmp)
    image_orient /= image_orient.sum()
    image_orient *= ftot_im
    return image_orient, pixel_size, extent_ima, ftot
