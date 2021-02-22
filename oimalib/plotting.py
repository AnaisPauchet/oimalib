# -*- coding: utf-8 -*-
"""
@author: Anthony Soulain (University of Sydney)
-----------------------------------------------------------------
OIMALIB: Optical Interferometry Modelisation and Analysis Library
-----------------------------------------------------------------

Set of function to plot oi data, u-v plan, models, etc.
-----------------------------------------------------------------
"""

import numpy as np
from matplotlib import pyplot as plt
from termcolor import cprint

from .fitting import comput_CP, comput_V2, select_model

# from .tools import rad2mas

dic_color = {'A0-B2': '#928a97',  # SB
             'A0-D0': '#7131CC',
             'A0-C1': '#ffc93c',
             'B2-C1': 'indianred',
             'B2-D0': '#086972',
             'C1-D0': '#3ec1d3',
             'D0-G2': '#f37735',  # MB
             'D0-J3': '#4b86b4',
             'D0-K0': '#CC9E3D',
             'G2-J3': '#d11141',
             'G2-K0': '#A6DDFF',
             'J3-K0': '#00b159',
             'A0-G1': '#96d47c',  # LB
             'A0-J2': '#f38181',
             'A0-J3': '#1f5f8b',
             'G1-J2': '#a393eb',
             'G1-J3': '#eedf6b',
             'J2-J3': 'c',
             'J2-K0': 'c',
             'A0-K0': '#8d90a1',
             'G1-K0': '#ffd100',
             }

err_pts_style = {'linestyle': "None", 'capsize': 1,  # 'ecolor': '#364f6b', 'mec':'#364f6b',
                 'marker': '.', 'elinewidth': 0.5, 'alpha': 1, 'ms': 6}


def _peak_color_bl(base):
    if base in dic_color.keys():
        p_color = dic_color[base]
    else:
        station = base.split('-')
        base_new = '%s-%s' % (station[1], station[0])
        if base_new in dic_color.keys():
            p_color = dic_color[base_new]
        else:
            p_color = 'tab:blue'
    return p_color


def _index_2_tel(tab):
    """
    Make the match between index, telescope stations and color references.
    """
    dic_index = {}
    for data in tab:
        nbl = len(data.index)
        for i in range(len(data.index_ref)):
            ind = data.index_ref[i]
            tel = data.teles_ref[i]
            if ind not in dic_index.keys():
                dic_index[ind] = tel

    l_base = []
    for j in range(len(tab)):
        data = tab[j]
        for i in range(nbl):
            base = '%s-%s' % (dic_index[data.index[i][0]],
                              dic_index[data.index[i][1]])
            base2 = '%s-%s' % (dic_index[data.index[i][1]],
                               dic_index[data.index[i][0]])
            if (base2 in l_base):
                base = base2

            l_base.append(base)

    return dic_index, list(set(l_base))


def plot_oidata(tab, use_flag=True, cmax=200, v2min=0, v2max=1.2, model=False, param=None, fit=None,
                cond_uncer=False, rel_max=None, cond_wl=False, wl_min=None, wl_max=None, log=False,
                is_nrm=False, extra_error_v2=0, extra_error_cp=0, err_scale=1):
    """
    Plot the interferometric data (and the model if required), splitted in V2 and CP and restreined if different way.

    Parameters:
    -----------

    `tab`: {list}
        list containing of data from load() function (size corresponding to the number of files),\n
    `use_flag`: {boolean}
        If True, use flag from the original oifits file,\n
    `cp_born`: {float}
        Limit maximum along Y-axis of CP data plot,\n
    `v2max`: {float}
        Limit maximum along Y-axis of V2 data plot,\n
    `model`: {boolean}
        If True, display the model associated to the param dictionnary,\n
    `param`: {dict}
        Dictionnary containing model parameters,\n
    `fit`: {dict}
        Dictionnary containing the result of the fit (Smartfit function),\n
    `cond_uncer`: {boolean}
        If True, select the best data according their relative uncertainties (rel_max),\n
    `rel_max`: {float}
        if cond_uncer, maximum sigma uncertainties allowed [%],\n
    `cond_wl`: {boolean}
        If True, apply wavelenght restriction between wl_min and wl_max,\n
    `wl_min`, wl_max: {float}
        If cond_wl, limits of the wavelength domain [µm],\n
    `log`: {boolean}
        If True, display the Y-axis of the V2 plot in log scale.\n
    """
    global dic_color

    if type(tab) == list:
        data = tab[0]
    else:
        data = tab
        tab = [tab]

    nwl = len(data.wl)
    nbl = data.vis2.shape[0]
    ncp = data.cp.shape[0]

    dic_ind = _index_2_tel(tab)[0]

    l_fmin, l_fmax = [], []

    for data in tab:
        tfmax = data.freq_vis2.flatten().max()
        tfmin = data.freq_vis2.flatten().min()
        l_fmax.append(tfmax)
        l_fmin.append(tfmin)

    fmin = np.array(l_fmin).min()
    fmax = np.array(l_fmax).max()

    if model:
        model_target = select_model(param['model'])
        if fit is not None:
            chi2 = fit['chi2']
            label = 'Model %s ($\chi^2_{red}$ = %2.1f)' % (param['model'],
                                                           chi2)
        else:
            label = 'Model %s' % param['model']

    n_V2_rest = 0

    fig = plt.figure(figsize=(8, 6))
    ax1 = plt.subplot2grid((3, 1), (0, 0), rowspan=2)

    list_bl = []

    ncolor = 0

    for j in range(len(tab)):
        data = tab[j]
        for i in range(nbl):
            if use_flag:
                sel_flag = np.invert(data.flag_vis2[i])
            else:
                sel_flag = [True]*nwl

            if cond_uncer:
                vis2 = data.vis2[i, :]
                e_vis2 = data.e_vis2[i]
                rel_err = e_vis2/vis2
                sel_err = (rel_err <= rel_max*1e-2)
            else:
                sel_err = np.array([True]*nwl)

            if cond_wl:
                sel_wl = (data.wl >= wl_min*1e-6) & (data.wl < wl_max*1e-6)
            else:
                sel_wl = np.array([True]*nwl)

            sel_neg = data.vis2[i, :] > 0

            cond = sel_flag & sel_err & sel_wl & sel_neg

            freq_vis2 = data.freq_vis2[i][cond]
            vis2 = data.vis2[i][cond]  # * 0.97
            e_vis2 = data.e_vis2[i][cond]
            n_V2_rest += len(vis2)

            base = '%s-%s' % (dic_ind[data.index[i][0]],
                              dic_ind[data.index[i][1]])
            base2 = '%s-%s' % (dic_ind[data.index[i][1]],
                               dic_ind[data.index[i][0]])
            if not ((base or base2) in list_bl):
                bl1 = base
                list_bl.append(base)
                list_bl.append(base2)
            else:
                bl1 = ''

            e_vis2 = np.sqrt(e_vis2**2 + extra_error_v2**2) * err_scale

            ms_model = 5

            p_color = _peak_color_bl(base)

            if is_nrm:
                ax1.errorbar(freq_vis2, vis2, yerr=e_vis2, ecolor='lightgray',
                             color=p_color, marker='.', ms=6, elinewidth=1)
            else:
                ax1.plot(freq_vis2, vis2, color=p_color,
                         ls='-', lw=1, label=bl1)
                ax1.fill_between(freq_vis2, vis2-e_vis2, vis2+e_vis2,
                                 color=p_color, alpha=.3)

            if model:
                u, v, wl = data.u[i], data.v[i], data.wl[cond]
                mod = comput_V2([u, v, wl], param, model_target)
                # in_uncer = (abs(vis2 - mod) <= 1*e_vis2)
                if is_nrm:
                    ax1.plot(freq_vis2, mod, marker='x',
                             color='crimson', alpha=.7, zorder=100,
                             ms=ms_model)
                else:
                    ax1.plot(freq_vis2, mod, 'k-',
                             alpha=.7, zorder=100, lw=1, ms=ms_model)
                    # ax1.plot(freq_vis2[~in_uncer], mod[~in_uncer], 'r-',
                    #          alpha=.7, zorder=100, lw=1, ms=ms_model)

            ncolor += 1

    if model:
        if is_nrm:
            ax1.plot(-1, -1, 'x', color='crimson', label=label)
        else:
            ax1.plot(-1, -1, 'k-', color='gray', label=label)

    if log:
        ax1.set_yscale('log')
        ax1.set_ylim(1e-4, 1.5e0)
    else:
        ax1.set_ylim(v2min, v2max)
    ax1.set_xlim(fmin-2, fmax+2)
    ax1.legend(fontsize=7)
    ax1.set_ylabel(r'V$^2$', fontsize=12)
    ax1.grid(alpha=.3)

    ax2 = plt.subplot2grid((3, 1), (2, 0))

    N_cp_rest = 0
    for j in range(len(tab)):
        data = tab[j]
        for i in range(ncp):

            if use_flag:
                sel_flag = np.invert(data.flag_cp[i])
            else:
                sel_flag = np.array([True]*nwl)

            if cond_uncer:
                cp = data.cp[i]
                e_cp = data.e_cp[i]
                rel_err = e_cp/cp
                sel_err = (abs(rel_err) < rel_max*1e-2)
            else:
                sel_err = np.array([True]*nwl)

            if cond_wl:
                sel_wl = (data.wl >= wl_min*1e-6) & (data.wl < wl_max*1e-6)
            else:
                sel_wl = np.array([True]*nwl)

            cond = sel_flag & sel_err & sel_wl

            freq_cp = data.freq_cp[i][cond]
            cp = data.cp[i][cond]
            e_cp = data.e_cp[i][cond]
            e_cp = np.sqrt(e_cp**2 + extra_error_cp**2)

            N_cp_rest += len(cp)
            ax2.errorbar(freq_cp, cp, yerr=e_cp,
                         color='#0085ca', **err_pts_style)

            if model:
                u1, u2, u3 = data.u1[i], data.u2[i], data.u3[i]
                v1, v2, v3 = data.v1[i], data.v2[i], data.v3[i]
                wl2 = data.wl[cond]
                X = [u1, u2, u3, v1, v2, v3, wl2]
                mod_cp = comput_CP(X, param, model_target)
                # in_uncer = (abs(mod_cp - cp) < 1*e_cp)
                ax2.plot(freq_cp, mod_cp, marker='x',
                         color='crimson', ms=5, zorder=100, alpha=.7)
                # ax2.plot(freq_cp[~in_uncer], mod_cp[~in_uncer],
                #          'rx', ms=5, zorder=100, alpha=.7)
    ax2.set_ylabel(r'CP [deg]', fontsize=12)
    ax2.set_xlabel(r'Sp. Freq [arcsec$^{-1}$]', fontsize=12)
    ax2.set_ylim(-cmax, cmax)
    ax2.set_xlim(fmin-2, fmax+2)
    ax2.grid(alpha=.2)
    plt.tight_layout()
    plt.show(block=False)

    return fig


def plot_uv(tab, bmax=150, use_flag=False, cond_uncer=False, cond_wl=False,
            wl_min=None, wl_max=None, rel_max=None):
    """
    Plot the u-v coverage.

    Parameters:
    -----------

    tab: {list}
        list containing of data from OiFile2Class function (size corresponding to the number of files).\n
    bmax: {float}
        Limits of the plot [Mlambda].\n
    use_flag: {boolean}
        If True, use flag from the original oifits file.\n
    cond_uncer: {boolean}
        If True, select the best data according their relative uncertainties (rel_max).\n
    rel_max: {float}
        if cond_uncer, maximum sigma uncertainties allowed [%].\n
    cond_wl: {boolean}
        If True, apply wavelenght restriction between wl_min and wl_max.\n
    wl_min, wl_max: {float}
        If cond_wl, limits of the wavelength domain [µm].\n
    """
    global dic_color

    if type(tab) == list:
        data = tab[0]
    else:
        data = tab
        tab = [tab]

    nwl = len(data.wl)
    nbl = data.vis2.shape[0]

    # ntel = len(data.teles_ref)
    # nbl = int((ntel * (ntel - 1))/2)

    list_bl = []

    dic_ind = _index_2_tel(tab)[0]

    if cond_wl:
        try:
            float(wl_min)
        except TypeError:
            cprint('-'*38, 'red')
            cprint('Warnings: wavelengths limits not set!', 'red')
            cprint('-'*38, 'red')

    plt.figure(figsize=(6.5, 6))
    ax = plt.subplot(111)
    l_base2 = []
    for j in range(len(tab)):
        data = tab[j]
        for i in range(nbl):
            if use_flag:
                sel_flag = np.invert(data.flag_vis2[i])
            else:
                sel_flag = [True]*nwl

            if cond_uncer:
                vis2 = data.vis2[i, :]
                e_vis2 = data.e_vis2[i]
                rel_err = e_vis2/vis2
                sel_err = (rel_err <= rel_max*1e-2)
            else:
                sel_err = np.array([True]*nwl)

            if cond_wl:
                try:
                    sel_wl = (data.wl >= wl_min*1e-6) & (data.wl < wl_max*1e-6)
                except TypeError:
                    sel_wl = np.array([True]*nwl)

            else:
                sel_wl = np.array([True]*nwl)

            cond = sel_flag & sel_err & sel_wl

            U = data.u[i]/data.wl/1e6
            V = data.v[i]/data.wl/1e6

            u = U[cond]
            v = V[cond]

            base = '%s-%s' % (dic_ind[data.index[i][0]],
                              dic_ind[data.index[i][1]])
            base2 = '%s-%s' % (dic_ind[data.index[i][1]],
                               dic_ind[data.index[i][0]])
            l_base2.append(base)
            if not (base or base2) in list_bl:
                bl1 = base
                list_bl.append(base)
            else:
                bl1 = ''

            try:
                plt.scatter(
                    u, v, s=15, c=dic_color[base], label=bl1, marker='o')
                plt.scatter(-u, -v, s=15, c=dic_color[base], marker='o')
            except Exception:
                station = base.split('-')
                base_new = '%s-%s' % (station[1], station[0])
                plt.scatter(
                    u, v, s=15, c=dic_color[base_new], label=bl1, marker='o')
                plt.scatter(-u, -v, s=15, c=dic_color[base_new], marker='o')
            ax.patch.set_facecolor('#f7f9fc')
            plt.axis([-bmax, bmax, -bmax, bmax])
            plt.grid(alpha=.5, linestyle=':')
            plt.vlines(0, -bmax, bmax, linewidth=1, color='gray', alpha=0.05)
            plt.hlines(0, -bmax, bmax, linewidth=1, color='gray', alpha=0.05)
            plt.xlabel(r'U [M$\lambda$]')
            plt.ylabel(r'V [M$\lambda$]')
            plt.legend(fontsize=9)
            plt.subplots_adjust(top=0.97,
                                bottom=0.09,
                                left=0.11,
                                right=0.975,
                                hspace=0.2,
                                wspace=0.2)
    plt.show(block=False)
    return dic_color


# def plot_uvdata_im(data, rot=0, unit_vis='lambda', onecolor=False, color='r', ms=3, alpha=1):
#     """ """
#     if unit_vis == 'lambda':
#         f = 1e6
#     elif unit_vis == 'arcsec':
#         f = rad2mas(1)/1000.

#     try:
#         npts = len(data)
#         one = False
#     except TypeError:
#         one = True
#         npts = 1

#     if one:
#         l_bl_label, l_color = [], []
#         index2tel = _index_2_tel([data])[0]
#         for i in range(len(data.index)):
#             tel1, tel2 = index2tel[data.index[i]
#                                    [0]], index2tel[data.index[i][1]]
#             name_bl = '%s-%s' % (tel1, tel2)
#             name_bl_r = '%s-%s' % (tel2, tel1)
#             try:
#                 c = dic_color[name_bl]
#                 label_bl = name_bl
#             except KeyError:
#                 c = dic_color[name_bl_r]
#                 label_bl = name_bl_r
#             l_bl_label.append(label_bl)
#             if onecolor:
#                 c = color
#             l_color.append(c)

#         tab = data
#         for j in range(6):
#             angle = np.deg2rad(rot)

#             um0 = tab.u[j]/tab.wl/f
#             vm0 = tab.v[j]/tab.wl/f

#             um = um0*np.cos(angle) - vm0*np.sin(angle)
#             vm = um0*np.sin(angle) + vm0*np.cos(angle)

#             plt.scatter(um, vm, s=5, color=l_color[j], alpha=alpha)
#             plt.scatter(-um, -vm, s=5, color=l_color[j], alpha=alpha)

#     else:
#         for i in range(npts):
#             tab = data[i]
#             l_bl_label, l_color = [], []
#             index2tel = _index_2_tel([tab])[0]
#             for k in range(len(tab.index)):
#                 tel1, tel2 = index2tel[tab.index[k]
#                                        [0]], index2tel[tab.index[k][1]]
#                 name_bl = '%s-%s' % (tel1, tel2)
#                 name_bl_r = '%s-%s' % (tel2, tel1)
#                 try:
#                     c = dic_color[name_bl]
#                     label_bl = name_bl
#                 except KeyError:
#                     c = dic_color[name_bl_r]
#                     label_bl = name_bl_r
#                 l_bl_label.append(label_bl)
#                 if onecolor:
#                     c = color
#                 l_color.append(c)

#             for j in range(6):
#                 angle = np.deg2rad(rot)

#                 um0 = tab.u[j]/tab.wl/f
#                 vm0 = tab.v[j]/tab.wl/f

#                 um = um0*np.cos(angle) - vm0*np.sin(angle)
#                 vm = um0*np.sin(angle) + vm0*np.cos(angle)
#                 plt.scatter(um, vm, s=5, color=l_color[j], alpha=alpha)
#                 plt.scatter(-um, -vm, s=5, color=l_color[j], alpha=alpha)

#     return None


# def plot_uvdata_im_v2(tab, bmax=150, use_flag=False, cond_uncer=False, cond_wl=False,
#                       rot=0, wl_min=None, wl_max=None, rel_max=None):
#     if type(tab) == list:
#         data = tab[0]
#     else:
#         data = tab
#         tab = [tab]

#     nwl = len(data.wl)
#     nbl = data.vis2.shape[0]

#     list_bl = []

#     dic_ind = _index_2_tel(tab)[0]

#     if cond_wl:
#         try:
#             float(wl_min)
#         except TypeError:
#             cprint('-'*38, 'red')
#             cprint('Warnings: wavelengths limits not set!', 'red')
#             cprint('-'*38, 'red')

#     l_base2 = []
#     for j in range(len(tab)):
#         data = tab[j]
#         for i in range(nbl):
#             if use_flag:
#                 sel_flag = np.invert(data.flag_vis2[i])
#             else:
#                 sel_flag = [True]*nwl

#             if cond_uncer:
#                 vis2 = data.vis2[i, :]
#                 e_vis2 = data.e_vis2[i]
#                 rel_err = e_vis2/vis2
#                 sel_err = (rel_err <= rel_max*1e-2)
#             else:
#                 sel_err = np.array([True]*nwl)

#             if cond_wl:
#                 try:
#                     sel_wl = (data.wl >= wl_min*1e-6) & (data.wl < wl_max*1e-6)
#                 except TypeError:
#                     sel_wl = np.array([True]*nwl)

#             else:
#                 sel_wl = np.array([True]*nwl)

#             cond = sel_flag & sel_err & sel_wl

#             U = data.u[i]/data.wl/1e6
#             V = data.v[i]/data.wl/1e6

#             u = U[cond]
#             v = V[cond]

#             base = '%s-%s' % (dic_ind[data.index[i][0]],
#                               dic_ind[data.index[i][1]])
#             base2 = '%s-%s' % (dic_ind[data.index[i][1]],
#                                dic_ind[data.index[i][0]])
#             l_base2.append(base)
#             if not (base or base2) in list_bl:
#                 bl1 = base
#                 list_bl.append(base)
#             else:
#                 bl1 = ''

#             try:
#                 angle = np.deg2rad(rot)
#                 um = u*np.cos(angle) - v*np.sin(angle)
#                 vm = u*np.sin(angle) + v*np.cos(angle)

#                 plt.scatter(
#                     um, vm, s=15, c=dic_color[base], label=bl1, marker='o')
#                 plt.scatter(-um, -vm, s=15, c=dic_color[base], marker='o')
#             except Exception:
#                 station = base.split('-')
#                 base_new = '%s-%s' % (station[1], station[0])
#                 um = u*np.cos(angle) - v*np.sin(angle)
#                 vm = u*np.sin(angle) + v*np.cos(angle)
#                 plt.scatter(
#                     um, vm, s=15, c=dic_color[base_new], label=bl1, marker='o')
#                 plt.scatter(-um, -vm, s=15, c=dic_color[base_new], marker='o')
