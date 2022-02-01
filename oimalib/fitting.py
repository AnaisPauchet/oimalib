import corner
import emcee
import numpy as np
from matplotlib import pyplot as plt
from munch import munchify as dict2class
from scipy.interpolate import interp1d
from scipy.optimize import minimize
from tqdm import tqdm
from uncertainties import ufloat
from termcolor import cprint

import oimalib

from . import complex_models
from .fit.dpfit import leastsqFit
from .tools import mas2rad, round_sci_digit

import multiprocessing
import sys

if sys.platform == "darwin":
    multiprocessing.set_start_method("fork", force=True)

err_pts_style = {
    "linestyle": "None",
    "capsize": 1,
    "ecolor": "#364f6b",
    "mec": "#364f6b",
    "marker": ".",
    "elinewidth": 0.5,
    "alpha": 1,
    "ms": 14,
}


def select_model(name):
    """ Select a simple model computed in the Fourier space
    (check model.py) """
    if name == "disk":
        model = complex_models.visUniformDisk
    elif name == "binary":
        model = complex_models.visBinary
    elif name == "binary_res":
        model = complex_models.visBinary_res
    elif name == "edisk":
        model = complex_models.visEllipticalUniformDisk
    elif name == "debrisDisk":
        model = complex_models.visDebrisDisk
    elif name == "clumpyDebrisDisk":
        model = complex_models.visClumpDebrisDisk
    elif name == "gdisk":
        model = complex_models.visGaussianDisk
    elif name == "egauss":
        model = complex_models.visEllipticalGaussianDisk
    elif name == "ering":
        model = complex_models.visThickEllipticalRing
    elif name == "lor":
        model = complex_models.visLorentzDisk
    elif name == "yso":
        model = complex_models.visYSO
    elif name == "lazareff":
        model = complex_models.visLazareff
    elif name == "lazareff_halo":
        model = complex_models.visLazareff_halo
    elif name == "ellipsoid":
        model = complex_models.visEllipsoid
    elif name == "cont":
        model = complex_models.visCont
    elif name == "bohn":
        model = complex_models.get_full_visibility
    elif name == "yso_line":
        model = complex_models.visLazareff_line
    elif name == "pwhl":
        model = complex_models.visPwhl
    else:
        model = None
    return model


def check_params_model(param):
    """ Check if the user parameters are compatible
    with the model. """
    isValid = True
    log = ""
    if param["model"] == "edisk":
        elong = param["elong"]
        minorAxis = mas2rad(param["minorAxis"])
        angle = np.deg2rad(param["pa"])
        if (elong < 1) or (angle < 0) or (angle > np.pi) or (minorAxis < 0):
            log = "# elong > 1,\n# minorAxis > 0 mas,\n# 0 < pa < 180 deg."
            isValid = False
    elif param["model"] == "binary":
        dm = param["dm"]
        if dm < 0:
            isValid = False
    elif param["model"] == "lazareff":
        la = param["la"]
        lk = param["lk"]
        cj = param["cj"]
        sj = param["sj"]
        kc = param["kc"]

        flor = param["flor"]
        incl = param["incl"]
        pa = param["pa"]

        fs = param["fs"]
        fc = param["fc"]
        fh = 1 - fs - fc
        cond_flux = (fh >= 0) & (fc >= 0) & (fs >= 0)

        c1 = fs + fh + fc != 1
        c2 = (incl > 90.0) | (incl < 0)
        c3 = (pa < -180) | (pa > 180)
        c4 = np.invert(cond_flux)
        c5 = (la < -1) | (la > 1.5)
        c6 = (lk < -1) | (lk > 1.0)
        c7 = (cj < -1) | (cj > 1.0)
        c8 = (sj < -1) | (sj > 1.0)
        c9 = (kc < -1) | (kc > 0)
        c10 = (flor < 0) | (flor > 1)
        if c1 | c2 | c3 | c4 | c5 | c6 | c7 | c8 | c9 | c10:
            log = (
                "# fs + fh + fc = 1,\n"
                + "# 0 < incl < 90,\n"
                + "# 0 < pa < 180 deg,\n"
                + "# -1 < la < 1.5,\n"
                + "# -1 < lk < 1.\n"
                + "# -1 < cj, sj < 1.\n"
            )
            isValid = False
    elif param["model"] == "lazareff_halo":
        la = param["la"]
        lk = param["lk"]
        cj = param["cj"]
        sj = param["sj"]
        kc = param["kc"]

        flor = param["flor"]
        incl = param["incl"]
        pa = param["pa"]

        fh = param["fh"]
        fc = param["fc"]
        fs = 1 - fh - fc
        cond_flux = (fh >= 0) & (fc >= 0) & (fs >= 0)

        c1 = fs + fh + fc != 1
        c2 = (incl > 90.0) | (incl < 0)
        c3 = (pa < -180) | (pa > 180)
        c4 = np.invert(cond_flux)
        c5 = (la < -1) | (la > 1.5)
        c6 = (lk < -1) | (lk > 1.0)
        c7 = (cj < -1) | (cj > 1.0)
        c8 = (sj < -1) | (sj > 1.0)
        c9 = (kc < -1) | (kc > 0)
        c10 = (flor < 0) | (flor > 1)
        if c1 | c2 | c3 | c4 | c5 | c6 | c7 | c8 | c9 | c10:
            log = (
                "# fs + fh + fc = 1,\n"
                + "# 0 < incl < 90,\n"
                + "# 0 < pa < 180 deg,\n"
                + "# -1 < la < 1.5,\n"
                + "# -1 < lk < 1.\n"
                + "# -1 < cj, sj < 1.\n"
            )
            isValid = False
    elif param["model"] == "yso_line":
        la = param["la"]
        lk = param["lk"]

        incl = param["incl"]
        pa = param["pa"]
        fs = param["fs"]
        fc = param["fc"]
        fh = 1 - fs - fc
        cond_flux = (fs >= 0) & (fc >= 0)

        lincl = param["lincl"]
        lpa = param["lpa"]

        c1 = fs + fh + fc != 1
        c2 = (incl > 90.0) | (incl < 0)
        c3 = (pa < 0) | (pa > 180)
        c4 = np.invert(cond_flux)
        c5 = (la < -1) | (la > 1.5)
        c6 = (lk < -1) | (lk > 1.0)
        c7 = (lincl > 90) | (lincl < 0)
        c8 = (lpa < 0) | (lpa > 180)

        if c1 | c2 | c3 | c4 | c5 | c6 | c7 | c8:
            log = (
                "# fs + fh + fc = 1,\n"
                + "# 0 < incl < 90,\n"
                + "# 0 < pa < 180 deg,\n"
                + "# -1 < la < 1.5,\n"
                + "# -1 < lk < 1.\n"
            )
            isValid = False
    elif param["model"] == "yso":
        incl = param["incl"]
        hfr = param["hfr"]
        pa = param["pa"]
        fs = param["fs"]
        fc = param["fc"]
        flor = param["flor"]
        fh = 1 - fs - fc
        cond_flux = (fs >= 0) & (fc >= 0)

        c1 = fs + fh + fc != 1
        c2 = (incl > 90.0) | (incl < 0)
        c3 = (pa < 0) | (pa > 180)
        c4 = hfr < 0
        c5 = np.invert(cond_flux)
        c6 = (flor < 0) | (flor > 1)
        if c1 | c2 | c3 | c4 | c5 | c6:
            log = "# cosi <= 1,\n# hfr > 0 mas,\n# 0 < pa < 180 deg."
            isValid = False

    return isValid, log


def comput_V2(X, param, model):
    """ Compute squared visibility for a given model."""
    u = X[0]
    v = X[1]
    wl = X[2]

    isValid = check_params_model(param)[0]

    if not isValid:
        V2 = np.nan
    else:
        V = model(u, v, wl, param)
        V2 = np.abs(V) ** 2
    return V2


def comput_phi(X, param, model):
    """ Compute phase visibility for a given model."""
    u = X[0]
    v = X[1]
    wl = X[2]

    isValid = check_params_model(param)[0]

    if not isValid:
        phi = np.nan
    else:
        V = model(u, v, wl, param)
        phi = np.angle(V, deg=True)
    return phi


def comput_CP(X, param, model):
    """ Compute closure phases for a given model."""
    u1 = X[0]
    u2 = X[1]
    u3 = X[2]
    v1 = X[3]
    v2 = X[4]
    v3 = X[5]
    wl = X[6]

    V1 = model(u1, v1, wl, param)
    V2 = model(u2, v2, wl, param)
    V3 = model(u3, v3, wl, param)

    BS = V1 * V2 * V3
    CP = np.rad2deg(np.arctan2(BS.imag, BS.real))
    return CP


def engine_multi_proc(inputs):
    """
    Function used to parallelize (model_parallelized() function).
    """
    try:
        X = inputs[0]
        typ = inputs[1]
        param = inputs[-1]

        modelname = param["model"]

        model_target = select_model(modelname)

        if typ == "V2":
            mod = comput_V2(X, param, model_target)
        elif typ == "CP":
            mod = comput_CP(X, param, model_target)
        else:
            mod = np.nan
    except Exception:
        mod = 0

    return mod


def model_standard(obs, param):
    """
    Compute model for each data points in obs tuple.
    """
    modelname = param["model"]
    model_target = select_model(modelname)
    bunch = len(obs)
    res = np.zeros(bunch)
    for i in range(bunch):
        try:
            typ = obs[i][1]
            if typ == "V2":
                X = obs[i][0]
                mod = comput_V2(X, param, model_target)
            elif typ == "V":
                X = obs[i][0]
                mod = comput_V2(X, param, model_target) ** 0.5
            elif typ == "CP":
                X = obs[i][0]
                mod = comput_CP(X, param, model_target)
            elif typ == "phi":
                X = obs[i][0]
                mod = comput_phi(X, param, model_target)
            else:
                mod = np.nan
            res[i] = mod
        except TypeError:
            pass

    output = np.array(res)
    return output


def model_standard_fast(d, param):
    l_mod_cp, l_mod_cvis = [], []
    for data in d:
        nbl = len(data.u)
        ncp = len(data.cp)
        model_target = select_model(param["model"])
        mod_cvis = []
        for i in range(nbl):
            vis2 = data.vis2[i]
            if len(vis2[~np.isnan(vis2)]) != 0:
                flag = data.flag_vis2[i]
                u, v, wl = data.u[i], data.v[i], data.wl[~flag]
                mod_cvis.append(model_target(u, v, wl, param))
        mod_cp = []  # np.zeros_like(data.cp)
        for i in range(ncp):
            cp = data.cp[i]
            cond_bad_cp = len(cp[~np.isnan(cp)]) != 0
            if cond_bad_cp:
                flag = data.flag_cp[i]
                u1, u2, u3 = data.u1[i], data.u2[i], data.u3[i]
                v1, v2, v3 = data.v1[i], data.v2[i], data.v3[i]
                wl2 = data.wl[~flag]
                X = [u1, u2, u3, v1, v2, v3, wl2]
                tmp = comput_CP(X, param, model_target)
                mod_cp.append(tmp)
        l_mod_cp.append(mod_cp)
        l_mod_cvis.append(mod_cvis)
    l_mod_cp = np.array(l_mod_cp)
    l_mod_cvis = np.array(l_mod_cvis)

    fitted = param["fitted"]
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

    npts = len(model_fast)
    isValid = check_params_model(param)[0]
    if isValid:
        model_fast
    else:
        model_fast = [np.nan] * npts
    return model_fast


def model_parallelized(obs, param, ncore=12):
    """
    Compute model for each data points in obs tuple (multiprocess version).
    """
    pool = multiprocessing.Pool(ncore)

    b = np.array([param] * len(obs))
    c = b.reshape(len(obs), 1)
    d = np.append(obs, c, axis=1)

    res = pool.map(engine_multi_proc, d)

    pool.close()
    pool.join()
    return np.array(res)


def fits2obs(
    inputdata,
    use_flag=True,
    cond_wl=False,
    wl_min=None,
    wl_max=None,
    cond_uncer=False,
    rel_max=None,
    extra_error_v2=0,
    extra_error_cp=0,
    err_scale=1,
    input_rad=False,
    verbose=True,
):
    """
    Convert and select data from an oifits file.

    Parameters:
    -----------

    `inputdata`: {str}
        Oifits file,\n
    `use_flag`: {boolean}
        If True, use flag from the original oifits file,\n
    `cond_wl`: {boolean}
        If True, apply wavelenght restriction between wl_min and wl_max,\n
    `wl_min`, `wl_max`: {float}
        if cond_wl, limits of the wavelength domain [µm],\n
    `cond_uncer`: {boolean}
        If True, select the best data according their relative uncertainties (rel_max),\n
    `rel_max`: {float}
        if cond_uncer, maximum sigma uncertainties allowed [%],\n
    `extra_error_v2`: {float}
        Additional uncertainty of the V2 (added quadraticaly),\n
    `extra_error_cp`: {float}
        Additional uncertainty of the CP (added quadraticaly),\n
    `err_scale`: {float}
        Scaling factor applied on the CP uncertainties usualy used to
        include the non-independant CP correlation,\n
    `verbose`: {boolean}
        If True, display useful information about the data selection,\n


    Return:
    -------

    Obs: {tuple}
        Tuple containing all the selected data in an appropriate format to perform the fit.

    """

    data = oimalib.load(inputdata)
    nwl = len(data.wl)

    nbl = data.vis2.shape[0]
    ncp = data.cp.shape[0]

    vis2_data = data.vis2.flatten()  # * 0.97
    e_vis2_data = (data.e_vis2.flatten() ** 2 + extra_error_v2 ** 2) ** 0.5
    flag_V2 = data.flag_vis2.flatten()

    if input_rad:
        cp_data = np.rad2deg(data.cp.flatten())
        e_cp_data = np.rad2deg(
            np.sqrt(data.e_cp.flatten() ** 2 + extra_error_cp ** 2) * err_scale
        )
    else:
        cp_data = data.cp.flatten()
        e_cp_data = np.sqrt(data.e_cp.flatten() ** 2 + extra_error_cp ** 2) * err_scale

    flag_CP = data.flag_cp.flatten()

    if use_flag:
        pass
    else:
        flag_V2 = [False] * len(vis2_data)
        flag_CP = [False] * len(cp_data)

    u_data, v_data = [], []
    u1_data, v1_data, u2_data, v2_data = [], [], [], []

    for i in range(nbl):
        for j in range(nwl):
            u_data.append(data.u[i])
            v_data.append(data.v[i])

    for i in range(ncp):
        for j in range(nwl):
            u1_data.append(data.u1[i])
            v1_data.append(data.v1[i])
            u2_data.append(data.u2[i])
            v2_data.append(data.v2[i])

    u_data = np.array(u_data)
    v_data = np.array(v_data)

    u1_data = np.array(u1_data)
    v1_data = np.array(v1_data)
    u2_data = np.array(u2_data)
    v2_data = np.array(v2_data)

    wl_data = np.array(list(data.wl) * nbl)
    wl_data_cp = np.array(list(data.wl) * ncp)

    obs = []

    for i in range(nbl * nwl):
        if flag_V2[i] & use_flag:
            pass
        else:
            if not cond_wl:
                tmp = [u_data[i], v_data[i], wl_data[i]]
                typ = "V2"
                obser = vis2_data[i]
                err = e_vis2_data[i]
                if cond_uncer:
                    if err / obser <= rel_max * 1e-2:
                        obs.append([tmp, typ, obser, err])
                    else:
                        pass
                else:
                    obs.append([tmp, typ, obser, err])

            else:
                if (wl_data[i] >= wl_min * 1e-6) & (wl_data[i] <= wl_max * 1e-6):
                    tmp = [u_data[i], v_data[i], wl_data[i]]
                    typ = "V2"
                    obser = vis2_data[i]
                    err = e_vis2_data[i]
                    if cond_uncer:
                        if err / obser <= rel_max * 1e-2:
                            obs.append([tmp, typ, obser, err])
                        else:
                            pass
                    else:
                        obs.append([tmp, typ, obser, err])
                else:
                    pass
    N_v2_rest = len(obs)

    for i in range(ncp * nwl):
        if flag_CP[i]:
            pass
        else:
            if not cond_wl:
                tmp = [
                    u1_data[i],
                    u2_data[i],
                    (u1_data[i] + u2_data[i]),
                    v1_data[i],
                    v2_data[i],
                    (v1_data[i] + v2_data[i]),
                    wl_data_cp[i],
                ]
                typ = "CP"
                obser = cp_data[i]
                err = e_cp_data[i]
                if cond_uncer:
                    if err / obser <= rel_max * 1e-2:
                        obs.append([tmp, typ, obser, err])
                    else:
                        pass
                else:
                    obs.append([tmp, typ, obser, err])
            else:
                if (wl_data_cp[i] >= wl_min * 1e-6) & (wl_data_cp[i] <= wl_max * 1e-6):
                    tmp = [
                        u1_data[i],
                        u2_data[i],
                        (u1_data[i] + u2_data[i]),
                        v1_data[i],
                        v2_data[i],
                        (v1_data[i] + v2_data[i]),
                        wl_data_cp[i],
                    ]
                    typ = "CP"
                    obser = cp_data[i]
                    err = e_cp_data[i]
                    if cond_uncer:
                        if err / obser <= rel_max * 1e-2:
                            obs.append([tmp, typ, obser, err])
                        else:
                            pass
                    else:
                        obs.append([tmp, typ, obser, err])
                else:
                    pass

    N_cp_rest = len(obs) - N_v2_rest

    Obs = np.array(obs)

    if verbose:
        print(
            "\nTotal # of data points: %i (%i V2, %i CP)"
            % (len(Obs), N_v2_rest, N_cp_rest)
        )
        if use_flag:
            print("-> Flag in oifits files used.")
        if cond_wl:
            print(
                r"-> Restriction on wavelenght: %2.2f < %s < %2.2f µm"
                % (wl_min, chr(955), wl_max)
            )
        if cond_uncer:
            print(
                r"-> Restriction on uncertainties: %s < %2.1f %%" % (chr(949), rel_max)
            )

    return Obs


def _normalize_err_obs(obs, verbose=False):
    """ Normalize the errorbars to give the same weight for the V2 and CP data """

    errs = [o[-1] for o in obs]
    techs = [("V2"), ("CP")]
    n = [0, 0, 0, 0]
    for o in obs:
        for j, t in enumerate(techs):
            if any([x in o[1] for x in t]):
                n[j] += 1
    if verbose:
        print("-" * 50)
        print("error bar normalization by observable:")
        for j, t in enumerate(techs):
            print(t, n[j], np.sqrt(float(n[j]) / len(obs) * len(n)))
        print("-" * 50)

    for i, o in enumerate(obs):
        for j, t in enumerate(techs):
            if any([x in o[1] for x in t]):
                errs[i] *= np.sqrt(float(n[j]) / len(obs) * len(n))
    return errs


def compute_chi2_curve(
    obs,
    name_param,
    params,
    array_params,
    fitOnly,
    normalizeErrors=False,
    fitCP=True,
    onlyCP=False,
    ymin=0,
    ymax=3,
):
    """
    Compute a 1D reduced chi2 curve to determine the pessimistic (fully correlated)
    uncertainties on one parameter (name_param).

    Parameters:
    -----------

    `obs`: {tuple}
        Tuple containing all the selected data from format_obs function,\n
    `name_param` {str}:
        Name of the parameter to compute the chi2 curve,\n
    `params`: {dict}
        Parameters of the model,\n
    `array_params`: {array}
        List of parameters used to computed the chi2 curve,\n
    `fitOnly`: {list}
        fitOnly is a list of keywords to fit. By default, it fits all parameters in `param`,\n
    `normalizeErrors`: {str or boolean}
        If 'techniques', give the same weight for the V2 and CP data (even if only few CP compare to V2),\n
    `fitCP`: {boolean}
        If True, fit the CP data. If not fit only the V2 data.\n

    Returns:
    --------

    `fit` {dict}:
        Contains the results of the initial fit,\n
    `errors_chi2` {float}:
        Computed errors using the chi2 curve at the position of the chi2_r.min() + 1.
    """

    fit = smartfit(
        obs,
        params,
        normalizeErrors=normalizeErrors,
        fitCP=fitCP,
        onlyCP=onlyCP,
        fitOnly=fitOnly,
        ftol=1e-8,
        epsfcn=1e-6,
        multiproc=False,
        verbose=True,
    )

    # fit_chi2 = fit['chi2']
    fit_theta = fit["best"][name_param]
    fit_e_theta = fit["uncer"][name_param]

    fitOnly.remove(name_param)
    l_chi2r = []
    for pr in tqdm(array_params, desc="Chi2 curve (%s)" % name_param, ncols=100):
        params[name_param] = pr
        lfits = smartfit(
            obs,
            params,
            normalizeErrors=False,
            fitCP=fitCP,
            onlyCP=onlyCP,
            fitOnly=fitOnly,
            ftol=1e-8,
            epsfcn=1e-6,
            multiproc=False,
            verbose=False,
        )
        l_chi2r.append(lfits["chi2"])

    n_freedom = len(fitOnly)
    n_pts = len(obs)

    l_chi2r = np.array(l_chi2r)
    l_chi2 = np.array(l_chi2r) * (n_pts - (n_freedom - 1))

    plt.figure()
    plt.plot(array_params, l_chi2r)

    chi2r_m = l_chi2r.min()
    chi2_m = l_chi2.min()

    fitted_param = array_params[l_chi2 == chi2_m]

    c_left = array_params <= fitted_param
    c_right = array_params >= fitted_param

    left_curve = interp1d(l_chi2r[c_left], array_params[c_left])
    right_curve = interp1d(l_chi2r[c_right], array_params[c_right])

    try:
        left_res = left_curve(chi2r_m + 1)
        right_res = right_curve(chi2r_m + 1)
        dr1_r = abs(fitted_param - left_res)
        dr2_r = abs(fitted_param - right_res)
        # dr1_r, dig = round_sci_digit(dr1_r)
        # dr2_r = round_sci_digit(dr2_r)[0]
        fitted_param = float(np.round(fitted_param, 2))
        dr1_r = float(np.round(dr1_r, 2))
        dr2_r = float(np.round(dr2_r, 2))
        print("sig_chi2: %s = %s - %s + %s" % (name_param, fitted_param, dr1_r, dr2_r))
    except ValueError:
        print("Try to increase the parameters bounds (chi2_r).")
        return None

    dr1_r = round_sci_digit(dr1_r)[0]
    dr2_r = round_sci_digit(dr2_r)[0]

    errors_chi2 = np.mean([dr1_r, dr2_r])

    # fit_e_theta, scidigit = round_sci_digit(fit_e_theta)
    # fit_theta = float(np.round(fit_theta, 3))

    plt.figure()
    plt.plot(
        array_params[c_left], l_chi2r[c_left], color="tab:blue", lw=3, alpha=1, zorder=1
    )
    plt.plot(array_params[c_right], l_chi2r[c_right], color="tab:blue", lw=3, alpha=1)
    plt.plot(
        fit_theta,
        l_chi2r.min(),
        ".",
        color="#fc5185",
        ms=10,
        label="fit: %s=%2.2f±%2.2f" % (name_param, fit_theta, fit_e_theta),
    )
    plt.axvspan(
        fitted_param - dr1_r,
        fitted_param + dr2_r,
        ymin=ymin,
        ymax=ymax,
        color="#dbe4e8",
        label=r"$\sigma_{m1}=$-%s/+%s" % (dr1_r, dr2_r),
    )
    plt.axvspan(
        fitted_param - fit_e_theta,
        fitted_param + fit_e_theta,
        ymin=ymin,
        ymax=ymax,
        color="#359ccb",
        label=r"$\sigma_{m2}$",
        alpha=0.3,
    )
    plt.grid(alpha=0.1, color="grey")
    plt.legend(loc="best", fontsize=9)
    plt.ylabel(r"$\chi^2_{red}$")
    plt.xlim(array_params.min(), array_params.max())
    plt.tight_layout()
    plt.show(block=False)

    return fit, errors_chi2


def smartfit(
    data,
    first_guess,
    doNotFit=None,
    fitOnly=None,
    follow=None,
    ftol=1e-4,
    epsfcn=1e-7,
    normalizeErrors=False,
    scale_err=1,
    verbose=False,
    tobefit=["CP", "V2"],
    fast=True,
):
    """
    Perform the fit the observable in `tobefit` contained in data list (or class).

    Parameters:
    -----------

    data: {list}
        List containing all the selected data from `load()` function.\n
    first_guess: {dict}
        Parameters of the model.\n
    fitOnly: {list}
        fitOnly is a LIST of keywords to fit. By default, it fits all parameters in 'first_guess'.\n
    follow: {list}
        List of parameters to "follow" in the fit, i.e. to print in verbose mode.\n
    normalizeErrors: {boolean}
        If True, give the same weight for each observables (even if only few CP compare to V2).
    """
    first_guess["fitted"] = tobefit

    # -- avoid fitting string parameters
    tmp = list(filter(lambda x: isinstance(first_guess[x], str), first_guess.keys()))

    if type(data) is not list:
        data = [data]

    if len(tmp) > 0:
        if doNotFit is None:
            doNotFit = tmp
        else:
            doNotFit.extend(tmp)
        try:
            fitOnly = list(
                filter(lambda x: not isinstance(first_guess[x], str), fitOnly)
            )
        except Exception:
            pass

    if fast:
        M = model_standard_fast
        obs = np.concatenate([oimalib.format_obs(x) for x in data])
        save_obs = obs.copy()
        obs = []
        for o in save_obs:
            if o[1] in tobefit:
                obs.append(o)
        obs = np.array(obs)
    else:
        M = model_standard
        obs = np.array(data, dtype=object)

    errs = [o[-1] for o in obs]
    if normalizeErrors:
        errs = _normalize_err_obs(obs)

    Y = [o[2] for o in obs]

    lfit = leastsqFit(
        M,
        data,
        first_guess,
        Y,
        err=np.array(errs) * scale_err,
        doNotFit=doNotFit,
        fitOnly=fitOnly,
        follow=follow,
        normalizedUncer=False,
        verbose=verbose,
        ftol=ftol,
        epsfcn=epsfcn,
    )

    p = {}
    for k in fitOnly:
        err = lfit["uncer"][k]
        if err < 0:
            err = np.nan
        p[k] = ufloat(lfit["best"][k], err)
    lfit["p"] = p
    return lfit


def _compute_uvcoord(data):
    nwl = len(data.wl)
    nbl = data.vis2.shape[0]
    ncp = data.cp.shape[0]

    u_data = np.zeros([nbl, nwl])
    v_data = np.zeros_like(u_data)

    u1_data = np.zeros([ncp, nwl])
    u2_data = np.zeros_like(u1_data)
    u3_data = np.zeros_like(u1_data)
    v1_data = np.zeros_like(u1_data)
    v2_data = np.zeros_like(u1_data)
    v3_data = np.zeros_like(u1_data)

    for i in range(nbl):
        u_data[i] = np.array([data.u[i]] * nwl)
        v_data[i] = np.array([data.v[i]] * nwl)
    u_data, v_data = u_data.flatten(), v_data.flatten()

    for i in range(ncp):
        u1_data[i] = np.array([data.u1[i]] * nwl)
        u2_data[i] = np.array([data.u2[i]] * nwl)
        u3_data[i] = np.array([data.u3[i]] * nwl)
        v1_data[i] = np.array([data.v1[i]] * nwl)
        v2_data[i] = np.array([data.v2[i]] * nwl)
        v3_data[i] = np.array([data.v3[i]] * nwl)

    u1_data, v1_data = u1_data.flatten(), v1_data.flatten()
    u2_data, v2_data = u2_data.flatten(), v2_data.flatten()
    u3_data, v3_data = u3_data.flatten(), v3_data.flatten()

    output = dict2class(
        {
            "u": u_data,
            "v": v_data,
            "u1": u1_data,
            "v1": v1_data,
            "u2": u2_data,
            "v2": v2_data,
            "u3": u3_data,
            "v3": v3_data,
        }
    )
    return output


def format_obs(data, use_flag=True, input_rad=False, verbose=False):
    nwl = len(data.wl)
    nbl = data.vis2.shape[0]
    ncp = data.cp.shape[0]

    vis2_data = data.vis2.flatten()
    e_vis2_data = data.e_vis2.flatten()
    flag_V2 = data.flag_vis2.flatten()

    dvis_data = data.dvis.flatten()
    e_dvis_data = data.e_dvis.flatten()
    flag_dvis = data.flag_dvis.flatten()

    cp_data = data.cp.flatten()
    e_cp_data = data.e_cp.flatten()
    flag_CP = data.flag_cp.flatten()
    if input_rad:
        cp_data = np.rad2deg(cp_data)
        e_cp_data = np.rad2deg(e_cp_data)

    if not use_flag:
        flag_V2 = [False] * len(vis2_data)
        flag_CP = [False] * len(cp_data)
        flag_dvis = [False] * len(dvis_data)

    uv = _compute_uvcoord(data)

    wl_data = np.array(list(data.wl) * nbl)
    wl_data_cp = np.array(list(data.wl) * ncp)

    obs = []
    for i in range(nbl * nwl):
        if not flag_V2[i]:
            tmp = [uv.u[i], uv.v[i], wl_data[i]]
            typ = "V2"
            obser = vis2_data[i]
            err = e_vis2_data[i]
            if ~np.isnan(obser) | ~np.isnan(err):
                obs.append([tmp, typ, obser, err])
    N_v2 = len(obs)

    for i in range(nbl * nwl):
        if not flag_dvis[i]:
            tmp = [uv.u[i], uv.v[i], wl_data[i]]
            typ = "V"
            obser = dvis_data[i]
            err = e_dvis_data[i]
            obs.append([tmp, typ, obser, err])
    N_vis = len(obs) - N_v2

    for i in range(ncp * nwl):
        if not flag_CP[i]:
            tmp = [
                uv.u1[i],
                uv.u2[i],
                uv.u3[i],
                uv.v1[i],
                uv.v2[i],
                uv.v3[i],
                wl_data_cp[i],
            ]
            typ = "CP"
            obser = cp_data[i]
            err = e_cp_data[i]
            if ~np.isnan(obser) | ~np.isnan(err):
                obs.append([tmp, typ, obser, err])
    N_cp = len(obs) - N_v2 - N_vis

    Obs = np.array(obs, dtype=object)

    if verbose:
        print("\nTotal # of data points: %i (%i V2, %i CP)" % (len(Obs), N_v2, N_cp))
    return Obs


# MCMC estimation of uncertainties and refined parameter estimates


def _compute_model_mcmc(p_mcmc, data, param, fitOnly, tobefit, fast=False):
    """ Compute model for mcmc purpose changing only the parameters from `fitOnly`.
    
    Parameters:
    -----------
    `p_mcmc` {list, float}:
        List of parameter values formated for mcmc fitting process (the order is
        the same as `fitOnly` one),\n
    `obs` {array}: 
        List of data formated with oimalib.format_obs(). obs[0] is the u, v and
        wl position, obs[1] is the observable type ('V2', 'CP', 'VIS'), obs[2]
        is the observable value and obs[3] is the uncertainty,\n
    `param` {dict}:
        Dictionnary of parameters used with oimalib features,\n
    `fitOnly` {list, str}:
        List of parameters to be fit (smartfit LM or MCMC).
        
    Results:
    --------
    `model` {array}:
        Computed model values sorted as obs (model.shape = len(observables)).
    """
    for i, p in enumerate(fitOnly):
        param[p] = p_mcmc[i]

    if fast:
        param["fitted"] = tobefit
        M = model_standard_fast
    else:
        M = model_standard
    model = M(data, param)
    return np.array(model)


def log_prior(param, prior, fitOnly):
    """ Return -inf if param[p] is outside prior requirements."""
    for i, p in enumerate(fitOnly):
        if param[i] < prior[p][0] or (param[i] > prior[p][1]):
            return -np.inf
    return 0


def log_likelihood(p_mcmc, data, param, fitOnly, tobefit, obs=None, fast=False):
    """ Compute the likelihood estimation between the model (represented by param
    and only for fitOnly parameters) and the data (obs).
    
    Parameters:
    -----------
    `p_mcmc` {list, float}:
        List of parameter values formated for mcmc fitting process (the order is
        the same as `fitOnly` one),\n
    `obs` {array}: 
        List of data formated with oimalib.format_obs(). obs[0] is the u, v and
        wl position, obs[1] is the observable type ('V2', 'CP', 'VIS'), obs[2]
        is the observable value and obs[3] is the uncertainty,\n
    `param` {dict}:
        Dictionnary of parameters used with oimalib features,\n
    `fitOnly` {list, str}:
        List of parameters to be fit (smartfit LM or MCMC).
    """
    model = _compute_model_mcmc(p_mcmc, data, param, fitOnly, tobefit, fast=fast)

    y = obs[:, 2]
    e_y = obs[:, 3]

    inv_sigma2 = 1.0 / (e_y ** 2)
    res = -0.5 * (
        np.sum((y - model) ** 2 * inv_sigma2 - np.log(inv_sigma2.astype(float)))
    )

    # res = -0.5 * np.sum((y - model) ** 2 / e_y ** 2)

    if np.isnan(res):
        return -np.inf
    else:
        return res


def log_probability(p_mcmc, data, param, fitOnly, prior, tobefit, fast=False, obs=None):
    """ Similar to log_probability() but including the prior restrictions.
    
    Parameters:
    -----------
    `p_mcmc` {list, float}:
        List of parameter values formated for mcmc fitting process (the order is
        the same as `fitOnly` one),\n
    `obs` {array}: 
        List of data formated with oimalib.format_obs(). obs[0] is the u, v and
        wl position, obs[1] is the observable type ('V2', 'CP', 'VIS'), obs[2]
        is the observable value and obs[3] is the uncertainty,\n
    `param` {dict}:
        Dictionnary of parameters used with oimalib features,\n
    `fitOnly` {list, str}:
        List of parameters to be fit (smartfit LM or MCMC),\n
    `prior` {dict}:
        Dictionnary with the keys as fitOnly where the param[a] need to be
        between prior[a][0] <= param[a] <= prior[a][1]. 
    """
    lp = log_prior(p_mcmc, prior, fitOnly)
    if not np.isfinite(lp):
        return -np.inf
    return lp + log_likelihood(
        p_mcmc, data, param, fitOnly, tobefit, fast=fast, obs=obs
    )


def neg_like_prob(*args):
    """Minus log_probability computed to be used with minimize (scipy.optimize)."""
    return -log_probability(*args)


def _compute_initial_dist_mcmc(
    nwalkers, fitOnly, method="normal", w_walker=0.1, param=None, prior=None,
):
    nparam = len(fitOnly)
    pos = np.zeros([nwalkers, nparam])

    if method == "normal":
        for i in range(nparam):
            pp = param[fitOnly[i]]
            if pp == 0:
                i_p = prior[fitOnly[i]]
                m1, m2 = i_p[0], i_p[1]
                i_range = np.random.uniform(m1, m2, nwalkers)
            else:
                i_range = np.random.normal(pp, w_walker * abs(pp), nwalkers)
            pos[:, i] = i_range
    elif method == "prior":
        for i in range(nparam):
            pp = param[fitOnly[i]]
            i_p = prior[fitOnly[i]]
            m1, m2 = i_p[0], i_p[1]
            i_range = np.random.uniform(m1, m2, nwalkers)
            pos[:, i] = i_range
    elif method == "alex":
        p0 = np.array([param[fitOnly[i]] for i in range(nparam)])
        pos = p0 + 1e-1 * np.random.randn(nwalkers, nparam)
    return pos


def mcmcfit(
    data,
    first_guess,
    nwalkers=20,
    niter=150,
    prior=None,
    truths=None,
    fitOnly=None,
    guess_likehood=False,
    method="normal",
    threads=1,
    progress=True,
    plot_corner=False,
    burnin=50,
    tobefit=["V2", "CP"],
    fast=False,
):
    """ """
    ndim = len(fitOnly)

    obs = np.concatenate([oimalib.format_obs(x) for x in data])

    save_obs = obs.copy()
    obs = []
    for o in save_obs:
        if o[1] in tobefit:
            obs.append(o)
    obs = np.array(obs)

    if fast:
        args = (data, first_guess, fitOnly, prior, tobefit, fast, obs)
    else:
        args = (obs, first_guess, fitOnly, prior, tobefit, fast, obs)

    initial_mcmc = [first_guess[p] for p in fitOnly]
    if guess_likehood:
        initial_mcmc = [first_guess[p] for p in fitOnly]
        initial_like = np.array(initial_mcmc) + 0.1 * np.random.randn(len(initial_mcmc))
        soln = minimize(neg_like_prob, initial_like, args=args)
        initial_mcmc = soln.x
        print("\nMaximum likelihood estimates:")
        for i in range(ndim):
            print("%s = %2.3f" % (fitOnly[i], initial_mcmc[i]))

    pool = multiprocessing.Pool(threads)

    sampler = emcee.EnsembleSampler(
        nwalkers, ndim, log_probability, args=args, pool=pool,
    )

    pos = _compute_initial_dist_mcmc(
        nwalkers, fitOnly, method=method, param=first_guess, prior=prior,
    )

    cprint("------- MCMC is running ------", "cyan")
    sampler.run_mcmc(
        pos, niter, progress=progress, skip_initial_state_check=True,
    )

    pool.close()
    pool.join()

    if burnin > niter:
        cprint(
            "## Warning: Burnin = %i should be < %i iter (force to zero)."
            % (burnin, niter),
            "green",
        )
        cprint("-> You should increase the niter value (typically >100)", "green")
        burnin = 0

    flat_samples = sampler.get_chain(discard=burnin, thin=15, flat=True)
    for i in range(ndim):
        mcmc = np.percentile(flat_samples[:, i], [16, 50, 84])
        q = np.diff(mcmc)
        txt = "{3} = {0:.3f}_{{-{1:.3f}}}^{{{2:.3f}}}"
        txt = txt.format(mcmc[1], q[0], q[1], fitOnly[i])
        print(txt)

    if plot_corner:
        corner.corner(
            flat_samples,
            labels=fitOnly,
            bins=50,
            show_titles=True,
            title_fmt=".3f",
            color="k",
            truths=truths,
        )
    return sampler


def get_mcmc_results(sampler, param, fitOnly, burnin=200):
    flat_samples = sampler.get_chain(discard=burnin, thin=15, flat=True)

    ndim = len(fitOnly)

    fit_mcmc = {}
    fit_mcmc["best"] = param.copy()
    fit_mcmc["uncer"] = {}
    for i in range(ndim):
        mcmc = np.percentile(flat_samples[:, i], [16, 50, 84])
        q = np.diff(mcmc)
        txt = "{3} = {0:.3f}_{{-{1:.3f}}}^{{{2:.3f}}}"
        txt = txt.format(mcmc[1], q[0], q[1], fitOnly[i])
        print(txt)
        fit_mcmc["best"][fitOnly[i]] = mcmc[1]
        fit_mcmc["uncer"][fitOnly[i]] = max(q[0], q[1])
    return fit_mcmc
