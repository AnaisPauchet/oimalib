"""
@author: Anthony Soulain (University of Sydney)
-----------------------------------------------------------------
OIMALIB: Optical Interferometry Modelisation and Analysis Library
-----------------------------------------------------------------

Set of function to perform data selection.
-----------------------------------------------------------------
"""
from copy import deepcopy

import numpy as np
from munch import munchify

from oimalib.tools import binning_tab
from oimalib.tools import wtmn


def _check_bl_same(list_data):
    """Check if list_data contains only same VLTI configuration."""
    blname_ref = list_data[0].blname
    diff_bl = []
    for i in range(len(list_data) - 1):
        blname = list_data[i + 1].blname
        n_diff = len(set(blname_ref == blname))
        if n_diff != 1:
            diff_bl.append([i, blname])
    ckeck = True
    if len(diff_bl) != 0:
        print("Different BL found compared to the reference (%s)" % str(blname_ref))
        print(diff_bl)
        ckeck = False
    return ckeck


def _check_good_tel(list_data, verbose=True):
    """Check if one telescope if down in the list of dataset."""
    data_ref = list_data[0]
    blname = data_ref.blname
    nbl = len(blname)

    l_bad = []
    for i in range(nbl):
        dvis = data_ref.dvis[i]
        cond_nan = np.isnan(dvis)
        d_vis_sel = dvis[~cond_nan]
        if len(d_vis_sel) == 0:
            l_bad.append(blname[i])

    if len(l_bad) != 0:
        if verbose:
            print("\n## Warning: only nan detected in baselines:", l_bad)

    i_bl_bad = np.zeros(len(list_data[0].tel))
    l_tel = list(set(list_data[0].tel))

    if len(l_bad) != 0:
        for bad in l_bad:
            for i, tel in enumerate(l_tel):
                if tel in bad:
                    i_bl_bad[i] += 1

    nbad = len(i_bl_bad)
    max_bad = np.max(i_bl_bad)

    exclude_tel = []
    if len(l_bad) != 0:
        exclude_tel = [l_tel[i] for i in range(nbad) if i_bl_bad[i] == max_bad]

    if len(l_bad) != 0:
        if verbose:
            print(
                "-> so telescopes seem to be down and are automaticaly excluded:",
                exclude_tel,
            )
    return exclude_tel, l_tel


def _select_bl(list_data, blname, exclude_tel):
    """Compute the bl index and name if some need to be excluded (e.g.: down telescope)."""
    if len(exclude_tel) == 0:
        good_bl = range(len(list_data[0].blname))
        good_bl_name = blname
    else:
        good_bl, good_bl_name = [], []
        for i in range(len(blname)):
            for tel in exclude_tel:
                if tel not in blname[i]:
                    good_bl.append(i)
                    good_bl_name.append(blname[i])
    return good_bl, good_bl_name


def _select_data_v2(
    i,
    data,
    use_flag=False,
    cond_uncer=False,
    cond_wl=False,
    wl_bounds=None,
    rel_max=None,
):
    """Select data V2 using different criteria (errors, wavelenght, flag)."""
    nwl = len(data.wl)
    sel_flag = sel_err = sel_wl = np.array([False] * nwl)

    if use_flag:
        sel_flag = data.flag_vis2[i]

    vis2 = data.vis2[i, :]
    e_vis2 = data.e_vis2[i]
    if cond_uncer:
        rel_err = e_vis2 / vis2
        sel_err = np.invert(rel_err <= rel_max * 1e-2)

    if cond_wl:
        try:
            sel_wl = np.invert(
                (data.wl >= wl_bounds[0] * 1e-6) & (data.wl < wl_bounds[1] * 1e-6)
            )
        except TypeError:
            print("wl_bounds is None, please give wavelength limits (e.g.: [2, 3])")
    cond_v2 = sel_flag | sel_err | sel_wl
    return cond_v2


def _select_data_cp(
    i,
    data,
    use_flag=False,
    cond_uncer=False,
    cond_wl=False,
    wl_bounds=None,
):
    """Select data CP using different criteria (errors, wavelenght, flag)."""
    nwl = len(data.wl)

    sel_flag = sel_err = sel_wl = np.array([False] * nwl)

    if use_flag:
        sel_flag = data.flag_cp[i]

    e_cp = data.e_cp[i]
    if cond_uncer:
        sel_err = e_cp < 0

    if cond_wl:
        try:
            sel_wl = np.invert(
                (data.wl >= wl_bounds[0] * 1e-6) & (data.wl < wl_bounds[1] * 1e-6)
            )
        except TypeError:
            print("wl_bounds is None, please give wavelength limits (e.g.: [2, 3])")

    cond_cp = sel_flag | sel_err | sel_wl

    return cond_cp


def _find_max_freq(list_data):
    l_fmin, l_fmax = [], []
    for data in list_data:
        tfmax = data.freq_vis2.flatten().max()
        tfmin = data.freq_vis2.flatten().min()
        l_fmax.append(tfmax)
        l_fmin.append(tfmin)

    fmin = np.array(l_fmin).min()
    fmax = np.array(l_fmax).max()
    return fmin, fmax


def select_data(
    list_data,
    use_flag=True,
    cond_uncer=False,
    rel_max=None,
    cond_wl=False,
    wave_lim=None,
    extra_error_v2=0,
    extra_error_cp=0,
    err_scale_v2=1,
    err_scale_cp=1,
    replace_err=False,
    seuil_v2=None,
    seuil_cp=None,
):
    """
    Perform data selection base on uncertaintities (`cond_uncer`, `rel_max`),
    wavelength (`cond_wl`, `wl_bounds`) and data flagging (`use_flag`).
    Additionnal arguments are used to scale the uncertainties of the data (added
    and scaled).

    Parameters:
    -----------
    `list_data` {class/list}:
        Data or list of data from `oimalib.load()`,\n
    `use_flag` {boolean}:
        If True, use flag from the original oifits file,\n
    `cond_uncer` {boolean}:
        If True, select the best data according their relative
        uncertainties (`rel_max`),\n
    `rel_max` {float}:
        if `cond_uncer`=True, maximum sigma uncertainties allowed [%],\n
    `cond_wl` {boolean}:
        If True, apply wavelenght restriction between wl_min and wl_max,\n
    `wl_bounds` {array}:
        if `cond_wl`=True, limits of the wavelength domain [µm],\n
    `extra_error_v2`: {float}
        Additional uncertainty of the V2 (added quadraticaly),\n
    `extra_error_cp`: {float}
        Additional uncertainty of the CP (added quadraticaly),\n
    `err_scale_v2`: {float}
        Scaling factor applied on the V2 uncertainties,\n
    `err_scale_cp`: {float}
        Scaling factor applied on the CP uncertainties usualy used to
        include the non-independant CP correlation,\n
    """
    if type(list_data) != list:
        list_data = [list_data]
    nfile = len(list_data)
    fmin, fmax = _find_max_freq(list_data)

    param_select = {
        "use_flag": use_flag,
        "cond_uncer": cond_uncer,
        "cond_wl": cond_wl,
        "wl_bounds": wave_lim,
    }

    new_list = deepcopy(list_data)

    list_data_sel = []
    for data_i in new_list:
        data = data_i.copy()
        nbl = data.vis2.shape[0]
        ncp = data.cp.shape[0]
        for i in range(nbl):
            new_flag_v2 = _select_data_v2(i, data, rel_max=rel_max, **param_select)
            data.flag_vis2[i] = new_flag_v2
            old_err = data.e_vis2[i]
            if replace_err:
                e_vis2 = extra_error_v2
            else:
                e_vis2 = np.sqrt(old_err ** 2 + extra_error_v2 ** 2) * err_scale_v2
                if seuil_v2 is not None:
                    e_vis2[e_vis2 <= seuil_v2] = seuil_v2
            data.e_vis2[i] = e_vis2

        for j in range(ncp):
            new_flag_cp = _select_data_cp(j, data, **param_select)
            data.flag_cp[j] = new_flag_cp
            old_err = data.e_cp[j]
            if replace_err:
                e_cp = extra_error_cp
            else:
                e_cp = np.sqrt(old_err ** 2 + extra_error_cp ** 2) * err_scale_cp
                if seuil_cp is not None:
                    e_cp[e_cp <= seuil_cp] = seuil_cp
            data.e_cp[j] = e_cp

        data.info["fmax"] = fmax
        data.info["fmin"] = fmin
        list_data_sel.append(data)

    output = list_data_sel
    if nfile == 1:
        output = list_data_sel[0]
    return output


def spectral_bin_data(list_data, nbox=50, force=False, rel_err=0.01, wave_lim=None):
    """Compute spectrally binned observables using weigthed averages (based
    on squared uncertainties).

    Parameters:
    -----------

    `list_data` {list}:
        List of data class (see oimalib.load() for details),\n
    `nbox` {int}:
        Size of the box,\n
    `force` {bool}:
        If True, force the uncertainties as the relative error `rel_err`,\n
    `rel_err` {float}:
        If `force`, relative uncertainties to be used [%].

    Outputs:
    --------
    `output` {list}:
        Same as input `list_data` but spectrally binned.
    """

    if (type(list_data) != list) & (type(list_data) != np.ndarray):
        list_data = [list_data]
    nfile = len(list_data)

    list_data_bin = []
    for data in list_data:
        (
            l_wl,
            l_vis2,
            l_e_vis2,
            l_cp,
            l_e_cp,
            l_dvis,
            l_dphi,
            l_e_dvis,
            l_e_dphi,
        ) = binning_tab(data, nbox=nbox, force=force, rel_err=rel_err)
        data_bin = data.copy()
        data_bin.wl = l_wl
        data_bin.vis2 = l_vis2
        data_bin.e_vis2 = l_e_vis2
        data_bin.flag_vis2 = np.zeros_like(l_vis2) != 0
        data_bin.cp = l_cp
        data_bin.e_cp = l_e_cp
        data_bin.flag_cp = np.zeros_like(l_cp) != 0
        data_bin.dvis = l_dvis
        data_bin.e_dvis = l_e_dvis
        data_bin.dphi = l_dphi
        data_bin.e_dphi = l_e_dphi

        freq_cp, freq_vis2, bl_cp = [], [], []
        for i in range(len(data_bin.u1)):
            B1 = np.sqrt(data_bin.u1[i] ** 2 + data_bin.v1[i] ** 2)
            B2 = np.sqrt(data_bin.u2[i] ** 2 + data_bin.v2[i] ** 2)
            B3 = np.sqrt(data_bin.u3[i] ** 2 + data_bin.v3[i] ** 2)
            Bmax = np.max([B1, B2, B3])
            bl_cp.append(Bmax)
            freq_cp.append(Bmax / l_wl / 206264.806247)  # convert to arcsec-1
        freq_cp = np.array(freq_cp)
        for i in range(len(data_bin.u)):
            freq_vis2.append(
                data_bin.bl[i] / l_wl / 206264.806247
            )  # convert to arcsec-1
        freq_vis2 = np.array(freq_vis2)

        data_bin.freq_vis2 = freq_vis2
        data_bin.freq_cp = freq_cp

        if wave_lim is not None:
            data_bin = select_data(data_bin, cond_wl=True, wave_lim=wave_lim)

        list_data_bin.append(data_bin)

    output = list_data_bin
    if nfile == 1:
        output = list_data_bin[0]
    return output


def temporal_bin_data(list_data, wave_lim=None, time_lim=None, verbose=False):
    """Temporal bin between data observed during the same night. Can specify
    wavelength limits `wave_lim` (should be not used with spectrally binned data) and
    hour range `time_lim` to average the data according their observing time
    compared to the first obs (should be within an hour).

    Parameters:
    -----------
    `list_data` {list}:
        List of data class (see oimalib.load() for details),\n
    `wave_lim` {list, n=2}:
        Wavelength range to be exctracted [µm] (e.g.: around BrG line, [2.146, 2.186]),\n
    `time_lim` {list, n=2}:
        Time range to compute averaged obserbables [hour] (e.g.: [0, 1] for the first hour),\n
    """
    blname = list_data[0].blname
    cpname = list_data[0].cpname
    if not _check_bl_same(list_data):
        return None

    mjd0 = list_data[0].info.mjd
    l_hour = []
    for d in list_data:
        l_hour.append((d.info.mjd - mjd0) * 24)
    l_hour = np.array(l_hour)

    if wave_lim is None:
        wave_lim = [0, 20]

    exclude_tel, l_tel = _check_good_tel(list_data, verbose=verbose)

    good_bl, good_bl_name = _select_bl(list_data, blname, exclude_tel)

    if len(exclude_tel) != 0:
        good_cp = [i for i, x in enumerate(cpname) if exclude_tel[0] not in x]
        good_cp_name = [x for x in cpname if exclude_tel[0] not in x]
    else:
        good_cp = range(len(cpname))
        good_cp_name = cpname

    master_tel = list_data[0].tel
    n_bl = len(good_bl)
    n_data = len(list_data)

    if time_lim is None:
        n_data = len(list_data)
        file_to_be_combined = list(range(n_data))
    else:
        t0 = time_lim[0]
        t1 = time_lim[1]
        file_to_be_combined = [
            i for i in range(len(list_data)) if (l_hour[i] >= t0) & (l_hour[i] <= t1)
        ]
        n_data = len(file_to_be_combined)

    wave = list_data[0].wl * 1e6
    cond_wl = (wave >= wave_lim[0]) & (wave <= wave_lim[1])
    wave = wave[cond_wl]

    n_wave = len(wave)

    if n_wave == 0:
        wave = list_data[0].wl * 1e6
        n_wave = len(wave)
        cond_wl = [True] * n_wave

    tab_dvis = np.zeros([n_data, n_bl, n_wave])
    tab_e_dvis = np.zeros([n_data, n_bl, n_wave])
    tab_dphi = np.zeros([n_data, n_bl, n_wave])
    tab_e_dphi = np.zeros([n_data, n_bl, n_wave])

    tab_vis2 = np.zeros([n_data, n_bl, n_wave])
    tab_e_vis2 = np.zeros([n_data, n_bl, n_wave])

    n_cp = len(good_cp_name)
    tab_cp = np.zeros([n_data, n_cp, n_wave])
    tab_e_cp = np.zeros([n_data, n_cp, n_wave])
    tab_u1 = np.zeros([n_data, n_cp])
    tab_u2 = np.zeros([n_data, n_cp])
    tab_v1 = np.zeros([n_data, n_cp])
    tab_v2 = np.zeros([n_data, n_cp])

    all_u, all_v = [], []
    for i, ind_file in enumerate(file_to_be_combined):
        d = list_data[ind_file].copy()
        tmp_u, tmp_v = [], []

        for k, gdcp in enumerate(good_cp):
            cp = d.cp[gdcp][cond_wl]
            e_cp = d.e_cp[gdcp][cond_wl]
            e_cp[e_cp == 0] = np.nan
            tab_cp[i, k], tab_e_cp[i, k] = cp, e_cp
            tab_u1[i, k], tab_u2[i, k] = d.u1[gdcp], d.u2[gdcp]
            tab_v1[i, k], tab_v2[i, k] = d.v1[gdcp], d.v2[gdcp]

        for j, gd in enumerate(good_bl):
            tmp_u.append(d.u[gd])
            tmp_v.append(d.v[gd])

            vis2 = d.vis2[gd][cond_wl]
            e_vis2 = d.e_vis2[gd][cond_wl]
            e_vis2[e_vis2 == 0] = np.nan

            tab_vis2[i, j], tab_e_vis2[i, j] = vis2, e_vis2

            dvis = d.dvis[gd][cond_wl]
            dphi = d.dphi[gd][cond_wl]
            e_dvis = d.e_dvis[gd][cond_wl]
            e_dphi = d.e_dphi[gd][cond_wl]
            e_dvis[e_dvis == 0] = np.nan
            e_dphi[e_dphi == 0] = np.nan
            tab_dvis[i, j], tab_dphi[i, j] = dvis, dphi
            tab_e_dvis[i, j], tab_e_dphi[i, j] = e_dvis, e_dphi
        all_u.append(tmp_u)
        all_v.append(tmp_v)
    all_u, all_v = np.array(all_u), np.array(all_v)
    master_u = np.mean(all_u, axis=0)
    master_v = np.mean(all_v, axis=0)
    B = np.sqrt(master_u ** 2 + master_v ** 2)

    master_u1 = np.mean(tab_u1, axis=0)
    master_u2 = np.mean(tab_u2, axis=0)
    master_v1 = np.mean(tab_v1, axis=0)
    master_v2 = np.mean(tab_v2, axis=0)
    master_u3 = -(master_u1 + master_u2)
    master_v3 = -(master_v1 + master_v2)

    # Compute freq, blname
    freq_cp, freq_vis2, bl_cp = [], [], []

    wave /= 1e6
    for i in range(len(master_u1)):
        B1 = np.sqrt(master_u1[i] ** 2 + master_v1[i] ** 2)
        B2 = np.sqrt(master_u2[i] ** 2 + master_v2[i] ** 2)
        B3 = np.sqrt(master_u3[i] ** 2 + master_v3[i] ** 2)

        Bmax = np.max([B1, B2, B3])
        bl_cp.append(Bmax)
        freq_cp.append(Bmax / wave / 206264.806247)  # convert to arcsec-1

    for i in range(len(master_u)):
        freq_vis2.append(B[i] / wave / 206264.806247)  # convert to arcsec-1

    freq_cp = np.array(freq_cp)
    freq_vis2 = np.array(freq_vis2)
    bl_cp = np.array(bl_cp)

    weight_dvis, weight_dphi = 1.0 / tab_e_dvis ** 2, 1.0 / tab_e_dphi ** 2
    weight_vis2 = 1.0 / tab_e_vis2 ** 2
    weight_cp = 1.0 / tab_e_cp ** 2

    dvis_m, e_dvis_m = wtmn(tab_dvis, weights=weight_dvis)
    dphi_m, e_dphi_m = wtmn(tab_dphi, weights=weight_dphi)

    tab_vis2[np.isnan(tab_vis2)] = 0
    weight_vis2[np.isnan(weight_vis2)] = 1e-50

    vis2_m, e_vis2_m = wtmn(tab_vis2, weights=weight_vis2)

    tab_cp[np.isnan(tab_cp)] = 0
    weight_cp[np.isnan(weight_cp)] = 1e-50

    cp_m, e_cp_m = wtmn(tab_cp, weights=weight_cp)

    cond_flux = [True] * len(list_data[0].flux)
    if len(exclude_tel) != 0:
        cond_flux = ~(master_tel == exclude_tel[0])

    tab_flux = np.zeros([len(list_data), len(wave)])

    try:
        for i, d in enumerate(list_data):
            tab_flux[i] = np.mean(d.flux[cond_flux], axis=0)[cond_wl]
        master_flux = np.mean(tab_flux, axis=0)
    except IndexError:
        master_flux = np.array([[np.nan] * len(wave)] * 4)

    index_cp = []
    for i in good_cp:
        index_cp.append(list_data[0].index_cp[i])
    output = {
        "vis2": vis2_m,
        "e_vis2": e_vis2_m,
        "cp": cp_m,
        "e_cp": e_cp_m,
        "dvis": dvis_m,
        "e_dvis": e_dvis_m,
        "dphi": dphi_m,
        "e_dphi": e_dphi_m,
        "wl": wave,
        "blname": np.array(good_bl_name),
        "flux": master_flux,
        "u": master_u,
        "v": master_v,
        "info": list_data[0].info,
        "flag_vis2": np.zeros_like(vis2_m) != 0,
        "flag_dvis": np.zeros_like(dvis_m) != 0,
        "flag_cp": np.zeros_like(cp_m) != 0,
        "cpname": np.array(good_cp_name),
        "bl": B,
        "u1": master_u1,
        "v1": master_v1,
        "u2": master_u2,
        "v2": master_v2,
        "u3": master_u3,
        "v3": master_v3,
        "freq_cp": freq_cp,
        "freq_vis2": freq_vis2,
        "teles_ref": list_data[0].teles_ref,
        "index_ref": list_data[0].index_ref,
        "index_cp": index_cp,
    }

    return munchify(output)
