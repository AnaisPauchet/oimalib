# -*- coding: utf-8 -*-
"""
@author: Anthony Soulain (University of Sydney)
-----------------------------------------------------------------
OIMALIB: Optical Interferometry Modelisation and Analysis Library
-----------------------------------------------------------------

Set of function to perform data selection.
-----------------------------------------------------------------
"""


import numpy as np


def _select_data_v2(i, data, use_flag=False, cond_uncer=False, cond_wl=False,
                    wl_bounds=None, rel_max=None):
    """ Select data V2 using different criteria (errors, wavelenght, flag). """
    nwl = len(data.wl)
    sel_flag = sel_err = sel_wl = np.array([False]*nwl)

    if use_flag:
        sel_flag = data.flag_vis2[i]

    vis2 = data.vis2[i, :]
    e_vis2 = data.e_vis2[i]
    if cond_uncer:
        rel_err = e_vis2/vis2
        sel_err = np.invert((rel_err <= rel_max*1e-2))

    if cond_wl:
        try:
            sel_wl = np.invert((data.wl >= wl_bounds[0] *
                                1e-6) & (data.wl < wl_bounds[1]*1e-6))
        except TypeError:
            print(
                'wl_bounds is None, please give wavelength limits (e.g.: [2, 3])')

    cond_v2 = sel_flag | sel_err | sel_wl

    return cond_v2


def _select_data_cp(i, data, use_flag=False, cond_uncer=False, cond_wl=False,
                    wl_bounds=None, rel_max=None):
    """ Select data CP using different criteria (errors, wavelenght, flag). """
    nwl = len(data.wl)

    sel_flag = sel_err = sel_wl = np.array([False]*nwl)

    if use_flag:
        sel_flag = data.flag_cp[i]

    cp = data.cp[i, :]
    e_cp = data.e_cp[i]
    if cond_uncer:
        rel_err = e_cp/cp
        sel_err = np.invert((rel_err <= rel_max*1e-2))

    if cond_wl:
        try:
            sel_wl = np.invert((data.wl >= wl_bounds[0] *
                                1e-6) & (data.wl < wl_bounds[1]*1e-6))
        except TypeError:
            print(
                'wl_bounds is None, please give wavelength limits (e.g.: [2, 3])')

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


def select_data(list_data, use_flag=True, cond_uncer=False, rel_max=None,
                cond_wl=False, wl_bounds=None, extra_error_v2=0,
                extra_error_cp=0, err_scale_v2=1, err_scale_cp=1):
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

    param_select = {'use_flag': use_flag, 'cond_uncer': cond_uncer,
                    'rel_max': rel_max, 'cond_wl': cond_wl,
                    'wl_bounds': wl_bounds}

    for data in list_data:
        nbl = data.vis2.shape[0]
        ncp = data.cp.shape[0]
        for i in range(nbl):
            new_flag_v2 = _select_data_v2(i, data, **param_select)
            data.flag_vis2[i] = new_flag_v2
            old_err = data.e_vis2[i]
            e_vis2 = np.sqrt(old_err**2 + extra_error_v2**2) * err_scale_v2
            data.e_vis2[i] = e_vis2

        for j in range(ncp):
            new_flag_cp = _select_data_cp(j, data, **param_select)
            data.flag_cp[j] = new_flag_cp
            old_err = data.e_cp[j]
            e_cp = np.sqrt(old_err**2 + extra_error_cp**2) * err_scale_cp
            data.e_cp[j] = e_cp

        data.info['fmax'] = fmax
        data.info['fmin'] = fmin

        output = list_data
        if nfile == 1:
            output = list_data[0]
    return output
