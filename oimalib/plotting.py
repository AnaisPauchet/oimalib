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
import pkg_resources
from matplotlib import pyplot as plt
from matplotlib.colors import PowerNorm
from scipy.interpolate.interpolate import interp1d

from oimalib.complex_models import visGaussianDisk
from oimalib.fitting import check_params_model, select_model
from oimalib.fourier import UVGrid
from oimalib.tools import hide_xlabel, mas2rad, plot_vline, rad2mas, substract_run_med

dic_color = {
    "A0-B2": "#928a97",  # SB
    "A0-D0": "#7131CC",
    "A0-C1": "#ffc93c",
    "B2-C1": "indianred",
    "B2-D0": "#086972",
    "C1-D0": "#3ec1d3",
    "D0-G2": "#f37735",  # MB
    "D0-J3": "#4b86b4",
    "D0-K0": "#CC9E3D",
    "G2-J3": "#d11141",
    "G2-K0": "#A6DDFF",
    "J3-K0": "#00b159",
    "A0-G1": "#96d47c",  # LB
    "A0-J2": "#f38181",
    "A0-J3": "#1f5f8b",
    "G1-J2": "#a393eb",
    "G1-J3": "#eedf6b",
    "J2-J3": "c",
    "J2-K0": "c",
    "A0-K0": "#8d90a1",
    "G1-K0": "#ffd100",
    "U1-U2": "#82b4bb",
    "U2-U3": "#255e79",
    "U3-U4": "#267778",
    "U2-U4": "#ae3c60",
    "U1-U3": "#e35d5e",
    "U1-U4": "#f1ca7f",
}

err_pts_style = {
    "linestyle": "None",
    "capsize": 1,  # 'ecolor': '#9e978e',
    "marker": ".",
    "elinewidth": 0.5,
    "alpha": 1,
    "ms": 5,
}


def _update_color_bl(tab):
    data = tab[0]
    array_name = data.info["Array"]
    nbl_master = len(set(data.blname))

    if array_name == "CHARA":
        unknown_color = plt.cm.turbo(np.linspace(0, 1, nbl_master))
    else:
        unknown_color = plt.cm.Set2(np.linspace(0, 1, 8))

    i_cycle = 0
    for j in range(len(tab)):
        data = tab[j]
        nbl = data.vis2.shape[0]
        for i in range(nbl):
            base = data.blname[i]
            if base not in dic_color.keys():
                dic_color[base] = unknown_color[i_cycle]
                i_cycle += 1
    return dic_color


def _create_match_tel(data):
    dic_index = {}
    for i in range(len(data.index_ref)):
        ind = data.index_ref[i]
        tel = data.teles_ref[i]
        if ind not in dic_index.keys():
            dic_index[ind] = tel
    return dic_index


def check_closed_triplet(data, i=0):
    U = [data.u1[i], data.u2[i], data.u3[i], data.u1[i]]
    V = [data.v1[i], data.v2[i], data.v3[i], data.v1[i]]

    triplet = data.cpname[i]
    fig = plt.figure(figsize=[5, 4.5])
    plt.plot(U, V, label=triplet)
    plt.legend()
    plt.plot(0, 0, "r+")
    plt.grid(alpha=0.2)
    plt.xlabel("U [m]")
    plt.ylabel("V [m]")
    plt.tight_layout()
    return fig


def _flat_v2_data(data, use_flag=True):
    """ Flatten data V2 and apply flag (for plot_residuals()). """
    npts = len(data.freq_vis2.flatten())
    flag_vis2 = [True] * npts
    if use_flag:
        flag_vis2 = np.invert(data.flag_vis2.flatten())

    freq_vis2 = data.freq_vis2.flatten()[flag_vis2]
    vis2 = data.vis2.flatten()[flag_vis2]
    e_vis2 = data.e_vis2.flatten()[flag_vis2]
    wl = np.array([data.wl for i in range(len(data.vis2))]).flatten()[flag_vis2]
    return freq_vis2, vis2, e_vis2, wl, flag_vis2


def _flat_cp_data(data, use_flag=True):
    """ Flatten data CP and apply flag (for plot_residuals()). """
    npts = len(data.freq_cp.flatten())
    flag_cp = [True] * npts
    if use_flag:
        flag_cp = np.invert(data.flag_cp.flatten())
    freq_cp = data.freq_cp.flatten()[flag_cp]
    cp = data.cp.flatten()[flag_cp]
    e_cp = data.e_cp.flatten()[flag_cp]
    wl = np.array([data.wl for i in range(len(data.cp))]).flatten()[flag_cp]
    return freq_cp, cp, e_cp, wl, flag_cp


def plot_tellu(label=None, plot_ind=False, val=5000):
    file_tellu = pkg_resources.resource_stream(
        "oimalib", "internal_data/Telluric_lines.txt"
    )
    tellu = np.loadtxt(file_tellu, skiprows=1)
    plt.axvline(np.nan, lw=0.5, c="gray", alpha=0.5, label=label)
    for i in range(len(tellu)):
        plt.axvline(tellu[i], lw=0.5, c="crimson", ls="--", alpha=0.5)
        if plot_ind:
            plt.text(tellu[i], val, i, fontsize=7, c="crimson")


def _plot_uvdata_coord(tab, ax=None, rotation=0):
    """ Plot u-v coordinated of a bunch of data (see `plot_uv()`). """
    if type(tab) != list:
        tab = [tab]

    dic_color = _update_color_bl(tab)

    list_bl = []
    for data in tab:
        nbl = data.vis2.shape[0]
        for bl in range(nbl):
            flag = np.invert(data.flag_vis2[bl])
            u = data.u[bl] / data.wl[flag] / 1e6
            v = data.v[bl] / data.wl[flag] / 1e6
            base, label = data.blname[bl], ""
            if base not in list_bl:
                label = base
                list_bl.append(base)
            p_color = dic_color[base]
            angle = np.deg2rad(rotation)
            um = u * np.cos(angle) - v * np.sin(angle)
            vm = u * np.sin(angle) + v * np.cos(angle)
            ax.plot(um, vm, color=p_color, label=label, marker="o", ms=4)
            ax.plot(-um, -vm, ms=4, color=p_color, marker="o")
    return None


def plot_oidata(
    tab,
    use_flag=True,
    mod_v2=None,
    mod_cp=None,
    cp_max=200,
    v2min=0,
    v2max=1.2,
    log=False,
    is_nrm=False,
    ms_model=2,
    set_cp=1,
    color=False,
):
    """
    Plot the interferometric data (and the model if required).

    Parameters:
    -----------
    `tab` {list}:
        list containing of data from load() function (size corresponding to the
        number of files),\n
    `use_flag` {boolean}:
        If True, use flag from the oifits file (selected if select_data()
        was used before),\n
    `mod_v2`, `mod_cp` {array}:
        V2 and CP model computed from grid (`compute_grid_model()`) or
        the analytical model (`compute_geom_model()`),\n
    `cp_max` {float}:
        Limit maximum along Y-axis of CP data plot,\n
    `v2min`, `v2max` {float}:
        Limit maximum along Y-axis of V2 data plot,\n
    `log` {boolean}:
        If True, display the Y-axis of the V2 plot in log scale,\n
    `is_nrm` {boolean}:
        If True, data come from NRM data.
    """
    # global dic_color

    if type(tab) == list:
        data = tab[0]
    else:
        data = tab
        tab = [tab]

    dic_color = _update_color_bl(tab)

    array_name = data.info["Array"]
    l_fmin, l_fmax = [], []

    for data in tab:
        tfmax = data.freq_vis2.flatten().max()
        tfmin = data.freq_vis2.flatten().min()
        l_fmax.append(tfmax)
        l_fmin.append(tfmin)

    fmin = np.array(l_fmin).min()
    fmax = np.array(l_fmax).max()

    ncp_master = len(set(data.cpname))

    fig = plt.figure(figsize=(8, 6))
    ax1 = plt.subplot2grid((5, 1), (0, 0), rowspan=3)

    list_bl = []
    # PLOT VIS2 DATA AND MODEL IF ANY (mod_v2)
    # ----------------------------------------

    ndata = len(tab)
    for j in range(ndata):
        data = tab[j]
        nwl = len(data.wl)
        nbl = data.vis2.shape[0]
        for i in range(nbl):
            if use_flag:
                sel_flag = np.invert(data.flag_vis2[i])
            else:
                sel_flag = [True] * nwl

            freq_vis2 = data.freq_vis2[i][sel_flag]
            vis2 = data.vis2[i][sel_flag]
            e_vis2 = data.e_vis2[i][sel_flag]
            base, label = data.blname[i], ""
            wave = data.wl[sel_flag]

            if base not in list_bl:
                label = base
                list_bl.append(base)

            if is_nrm:
                p_color = "tab:blue"
                ax1.errorbar(
                    freq_vis2,
                    vis2,
                    yerr=e_vis2,
                    ecolor="lightgray",
                    color=p_color,
                    marker=".",
                    ms=6,
                    elinewidth=1,
                )
            else:
                p_color = dic_color[base]
                if len(vis2) == 1:
                    ax1.errorbar(
                        freq_vis2,
                        vis2,
                        yerr=e_vis2,
                        color=p_color,
                        label=label,
                        **err_pts_style
                    )
                else:
                    if color:
                        sc = ax1.scatter(freq_vis2, vis2, c=wave * 1e6, s=3)
                    else:
                        ebar = ax1.plot(
                            freq_vis2, vis2, color=p_color, ls="-", lw=1, label=label
                        )
                        ax1.fill_between(
                            freq_vis2,
                            vis2 - e_vis2,
                            vis2 + e_vis2,
                            color=p_color,
                            alpha=0.3,
                        )

            if mod_v2 is not None:
                mod = mod_v2[j][i][sel_flag]
                ax1.plot(
                    freq_vis2,
                    mod,
                    marker="x",
                    color="k",
                    alpha=0.7,
                    zorder=100,
                    lw=1,
                    ms=ms_model,
                    ls="",
                )

    if mod_v2 is not None:
        ax1.plot(
            -1,
            -1,
            marker="x",
            color="k",
            alpha=0.7,
            zorder=100,
            lw=1,
            ms=ms_model,
            ls="",
            label="model",
        )

    if log:
        ax1.set_yscale("log")
        ax1.set_ylim(v2min, v2max)
    else:
        ax1.set_ylim(v2min, v2max)

    offset = 0
    if data.info["Array"] == "CHARA":
        offset = 150
    ax1.set_xlim(fmin - 2, fmax + 2 + offset)
    if not color:
        handles, labels = ax1.get_legend_handles_labels()
        labels, handles = zip(*sorted(zip(labels, handles), key=lambda t: t[0]))
        ax1.legend(handles, labels, fontsize=7)
    else:
        cax = fig.add_axes(
            [
                ax1.get_position().x1 * 0.83,
                ax1.get_position().y1 * 1.03,
                0.2,
                ax1.get_position().height * 0.04,
            ]
        )
        cb = plt.colorbar(sc, cax=cax, orientation="horizontal")
        # cb.set_label(r"$\lambda$ [µm]")
        cb.ax.set_title(r"$\lambda$ [µm]", fontsize=9)
    # ax1.legend(fontsize=7)
    ax1.set_ylabel(r"V$^2$", fontsize=12)
    ax1.grid(alpha=0.3)

    # PLOT CP DATA AND MODEL IF ANY (mod_cp)
    # --------------------------------------
    ax2 = plt.subplot2grid((5, 1), (3, 0), rowspan=2)
    if not is_nrm:
        if array_name == "CHARA":
            fontsize = 5
            ax2.set_prop_cycle("color", plt.cm.turbo(np.linspace(0, 1, ncp_master)))
        elif array_name == "VLTI":
            fontsize = 7
            if set_cp == 0:
                ax2.set_prop_cycle(
                    "color", ["#f4dfcc", "#fa9583", "#2f4159", "#4097aa"]
                )
            elif set_cp == 1:
                ax2.set_prop_cycle(
                    "color", ["#eabd6f", "#fa9583", "#3a6091", "#79ab8e"]
                )
            elif set_cp == 2:
                ax2.set_prop_cycle(
                    "color", ["#79ab8e", "#5c95a8", "#fa9583", "#263a55"]
                )
        else:
            fontsize = 7
            ax2.set_prop_cycle("color", plt.cm.Set2(np.linspace(0, 1, 8)))
        color_cp = None
    else:
        color_cp = "tab:blue"

    color_cp_dic = {}
    list_triplet = []
    for j in range(len(tab)):
        data = tab[j]
        ncp = data.cp.shape[0]
        for i in range(ncp):

            if use_flag:
                sel_flag = np.invert(data.flag_cp[i])
            else:
                sel_flag = np.array([True] * nwl)

            freq_cp = data.freq_cp[i][sel_flag]
            cp = data.cp[i][sel_flag]
            e_cp = data.e_cp[i][sel_flag]
            wave = data.wl[sel_flag]

            dic_index = _create_match_tel(data)
            b1 = dic_index[data.index_cp[i][0]]
            b2 = dic_index[data.index_cp[i][1]]
            b3 = dic_index[data.index_cp[i][2]]
            triplet = "%s-%s-%s" % (b1, b2, b3)

            label = ""
            if triplet not in list_triplet:
                label = triplet
                list_triplet.append(triplet)

            if triplet in color_cp_dic.keys():
                color_cp = color_cp_dic[triplet]

            if not color:
                ebar = ax2.errorbar(
                    freq_cp, cp, yerr=e_cp, label=label, color=color_cp, **err_pts_style
                )
                if triplet not in color_cp_dic.keys():
                    color_cp_dic[triplet] = ebar[0].get_color()
            else:
                ax2.scatter(freq_cp, cp, c=wave, s=3)

            if mod_cp is not None:
                mod = mod_cp[j][i][sel_flag]
                ax2.plot(
                    freq_cp,
                    mod,
                    marker="x",
                    ls="",
                    color="k",
                    ms=ms_model,
                    zorder=100,
                    alpha=0.7,
                )

    if (is_nrm) | (not color):
        ax2.legend(fontsize=fontsize)
    ax2.set_ylabel(r"CP [deg]", fontsize=12)
    ax2.set_xlabel(r"Sp. Freq [arcsec$^{-1}$]", fontsize=12)
    ax2.set_ylim(-cp_max, cp_max)
    ax2.set_xlim(fmin - 2, fmax + 2 + offset)
    ax2.grid(alpha=0.2)
    plt.tight_layout()
    return fig


def plot_uv(tab, bmax=150, rotation=0):
    """
    Plot the u-v coverage.

    Parameters:
    -----------

    `tab` {list}:
        list containing of data from OiFile2Class function (size corresponding to the number of files),\n
    `bmax` {float}:
        Limits of the plot [Mlambda],\n
    """
    if type(tab) == list:
        wl_ref = np.mean(tab[0].wl) * 1e6
    else:
        wl_ref = np.mean(tab.wl) * 1e6

    bmax = bmax / wl_ref

    fig = plt.figure(figsize=(6.5, 6))
    ax = plt.subplot(111)

    ax2 = ax.twinx()
    ax3 = ax.twiny()

    _plot_uvdata_coord(tab, ax=ax, rotation=rotation)

    ax.patch.set_facecolor("#f7f9fc")
    ax.set_xlim([-bmax, bmax])
    ax.set_ylim([-bmax, bmax])
    ax2.set_ylim([-bmax * wl_ref, bmax * wl_ref])
    ax3.set_xlim([-bmax * wl_ref, bmax * wl_ref])
    plt.grid(alpha=0.5, linestyle=":")
    ax.axvline(0, linewidth=1, color="gray", alpha=0.2)
    ax.axhline(0, linewidth=1, color="gray", alpha=0.2)
    ax.set_xlabel(r"U [M$\lambda$]")
    ax.set_ylabel(r"V [M$\lambda$]")
    ax2.set_ylabel("V [m] - East", color="#007a59")
    ax3.set_xlabel("U [m] (%2.2f µm) - North" % wl_ref, color="#007a59")
    ax2.tick_params(axis="y", colors="#007a59")
    ax3.tick_params(axis="x", colors="#007a59")
    ax.legend(fontsize=7)
    plt.subplots_adjust(
        top=0.97, bottom=0.09, left=0.11, right=0.93, hspace=0.2, wspace=0.2
    )
    plt.tight_layout()
    plt.show(block=False)
    return fig


def plot_residuals(
    data,
    mod_v2,
    mod_cp,
    use_flag=True,
    modelname="model",
    d_freedom=0,
    cp_max=None,
    v2_min=None,
    v2_max=1.1,
    mega=False,
    color_wl=False,
):
    """ Plot the data with the model (param) and corresponding residuals.

    Parameters:
    -----------
    `data`: {class}
        Data from load() function,\n
    `param` {dict}:
        Dictionnary containing the model parameters,\n
    `use_flag`: {boolean}
        If True, use flag from the original oifits file,\n
    `d_freedom` {int}:
        Degree of freedom used during the model fitting.\n
    """
    data_v2 = _flat_v2_data(data, use_flag=use_flag)
    data_cp = _flat_cp_data(data, use_flag=use_flag)

    freq_vis2, vis2, e_vis2, wl_v2, cond_v2 = data_v2
    freq_cp, cp, e_cp, wl_cp, cond_cp = data_cp

    if mega:
        freq_vis2 = freq_vis2 * 206264.806247 / 1e6
        freq_cp = freq_cp * 206264.806247 / 1e6

    if (mod_v2 is None) or (mod_cp is None):
        return None

    mod_v2 = mod_v2.flatten()[cond_v2]
    mod_cp = mod_cp.flatten()[cond_cp]

    res_vis2 = (mod_v2 - vis2) / e_vis2
    res_cp = (mod_cp - cp) / e_cp

    chi2_cp = np.sum(((cp - mod_cp) ** 2 / (e_cp) ** 2)) / (len(e_cp) - (d_freedom - 1))
    chi2_vis2 = np.sum(((vis2 - mod_v2) ** 2 / (e_vis2) ** 2)) / (
        len(e_vis2) - (d_freedom - 1)
    )

    ms = 5
    fig = plt.figure(constrained_layout=True, figsize=(15, 6))
    axd = fig.subplot_mosaic(
        [["vis", "cp"], ["res_vis2", "res_cp"]],
        gridspec_kw={"width_ratios": [2, 2], "height_ratios": [3, 1]},
    )

    axd["res_vis2"].sharex(axd["vis"])
    axd["res_cp"].sharex(axd["cp"])

    if color_wl:
        sc = axd["vis"].scatter(
            freq_vis2, vis2, c=wl_v2 * 1e6, s=7, cmap="turbo", zorder=20
        )
        axd["vis"].errorbar(
            freq_vis2,
            vis2,
            yerr=e_vis2,
            marker="None",
            linestyle="None",
            elinewidth=1,
            color="grey",
            capsize=1,
        )
        cax = fig.add_axes(
            [
                axd["vis"].get_position().x1 * 0.93,
                axd["vis"].get_position().y1 * 0.8,
                0.01,
                axd["vis"].get_position().height * 0.4,
            ]
        )
        cb = plt.colorbar(sc, cax=cax)

        cb.set_label("$\lambda$ [µm]", fontsize=8)
    else:
        axd["vis"].errorbar(
            freq_vis2, vis2, yerr=e_vis2, **err_pts_style, color="#3d84a8"
        )
    axd["vis"].plot(
        freq_vis2,
        mod_v2,
        "x",
        color="#f6416c",
        zorder=100,
        ms=ms,
        label="%s ($\chi^2_r=%2.2f$)" % (modelname, chi2_vis2),
        alpha=0.7,
    )
    axd["vis"].legend()
    if v2_min is not None:
        axd["vis"].set_ylim(v2_min, v2_max)
    axd["vis"].grid(alpha=0.2)
    axd["vis"].set_ylabel(r"V$^2$")

    axd["res_vis2"].plot(freq_vis2, res_vis2, ".", color="#3d84a8")
    axd["res_vis2"].axhspan(-1, 1, alpha=0.2, color="#418fde")
    axd["res_vis2"].axhspan(-2, 2, alpha=0.2, color="#8bb8e8")
    axd["res_vis2"].axhspan(-3, 3, alpha=0.2, color="#c8d8eb")
    res_mas = 5
    try:
        if res_vis2.max() > 5:
            res_mas = res_vis2.max()
    except ValueError:
        axd["vis"].text(
            0.5,
            0.5,
            "ALL FLAGGED",
            color="r",
            ha="center",
            va="center",
            fontweight="bold",
            transform=axd["vis"].transAxes,
            fontsize=20,
        )

    axd["res_vis2"].set_ylim(-res_mas, res_mas)
    axd["res_vis2"].set_ylabel("Residual [$\sigma$]")
    axd["res_vis2"].set_xlabel("Sp. Freq. [arcsec$^{-1}$]")

    if color_wl:
        sc = axd["cp"].scatter(freq_cp, cp, c=wl_cp * 1e6, s=7, cmap="turbo", zorder=20)
        axd["cp"].errorbar(
            freq_cp,
            cp,
            yerr=e_cp,
            marker="None",
            linestyle="None",
            elinewidth=1,
            color="grey",
            capsize=1,
        )
    else:
        axd["cp"].errorbar(freq_cp, cp, yerr=e_cp, **err_pts_style, color="#289045")

    axd["cp"].plot(
        freq_cp,
        mod_cp,
        "x",
        color="#f6416c",
        zorder=100,
        ms=ms,
        label="%s ($\chi^2_r=%2.1f$)" % (modelname, chi2_cp),
        alpha=0.7,
    )

    if cp_max is not None:
        axd["cp"].set_ylim(-cp_max, cp_max)
    axd["cp"].grid(alpha=0.2)
    axd["cp"].set_ylabel("Closure phases [deg]")
    axd["cp"].legend()

    axd["res_cp"].plot(freq_cp, res_cp, ".", color="#1e7846")
    axd["res_cp"].axhspan(-1, 1, alpha=0.3, color="#28a16c")  # f5c893
    axd["res_cp"].axhspan(-2, 2, alpha=0.2, color="#28a16c")
    axd["res_cp"].axhspan(-3, 3, alpha=0.1, color="#28a16c")
    try:
        if res_cp.max() > 5:
            res_mas = res_cp.max()
        else:
            res_mas = 5
    except ValueError:
        axd["cp"].text(
            0.5,
            0.5,
            "ALL FLAGGED",
            color="r",
            ha="center",
            va="center",
            fontweight="bold",
            transform=axd["cp"].transAxes,
            fontsize=20,
        )
        pass
    axd["res_cp"].set_ylim(-res_mas, res_mas)
    axd["res_cp"].set_ylabel("Residual [$\sigma$]")
    axd["res_cp"].set_xlabel("Sp. Freq. [arcsec$^{-1}$]")
    return fig


def plot_complex_model(
    grid,
    data=None,
    i_sp=0,
    bmax=100,
    unit_im="mas",
    unit_vis="lambda",
    p=0.5,
    rotation=0,
):
    """ Plot model and corresponding visibility and phase plan. Additionallly, you
        can add data to show the u-v coverage compare to model.

    Parameters
    ----------
    `grid` : {class}
        Class generated using model2grid function,\n
    `i_sp` : {int}, optional
        Index number of the wavelength to display (in case of datacube), by default 0\n
    `bmax` : {int}, optional
        Maximum baseline to restrein the visibility field of view, by default 20\n
    `unit_im` : {str}, optional
        Unit of the spatial coordinates (model), by default 'arcsec'\n
    `unit_vis` : {str}, optional
        Unit of the complex coordinates (fft), by default 'lambda'\n
    `data` : {class}, optional
        Class containing oifits data (see OiFile2Class or dir2data function), by default None\n
    `p` : {float}, optional
        Normalization factor of the image, by default 0.5\n
    """
    fmin = grid.freq.min()
    fmax = grid.freq.max()
    fov = grid.fov
    if unit_im == "mas":
        f2 = 1
    else:
        unit_im = "arcsec"
        f2 = 1000.0

    if unit_vis == "lambda":
        f = 1e6
    elif unit_vis == "arcsec":
        f = rad2mas(1) / 1000.0

    extent_im = np.array([fov / 2.0, -fov / 2.0, -fov / 2.0, fov / 2.0]) / f2
    extent_vis = np.array([fmin, fmax, fmin, fmax]) / f

    fft2D = grid.fft
    cube = grid.cube

    im_phase = abs(np.angle(fft2D)[i_sp])[:, ::-1]
    im_amp = np.abs(fft2D)[i_sp][:, ::-1]
    im_model = cube[i_sp]

    wl_model = grid.wl[i_sp]

    umax = 2 * bmax / wl_model / f
    ax_vis = [-umax, umax, -umax, umax]
    modelname = grid.name

    # fig = plt.figure(figsize=(14, 5))
    fig, axs = plt.subplots(1, 3, figsize=(14, 5))
    # fig, ax1 = plt.subplot(1, 3, 1)
    axs[0].set_title(
        'Model "%s" ($\lambda$ = %2.2f $\mu$m)' % (modelname, wl_model * 1e6)
    )
    axs[0].imshow(
        im_model, norm=PowerNorm(p), origin="lower", extent=extent_im, cmap="afmhot"
    )
    axs[0].set_xlabel("$\Delta$RA [%s]" % (unit_im))
    axs[0].set_ylabel("$\Delta$DEC [%s]" % (unit_im))

    axs[1].set_title("Squared visibilities (V$^2$)")
    axs[1].imshow(
        im_amp ** 2,
        norm=PowerNorm(1),
        origin="lower",
        extent=extent_vis,
        cmap="gist_earth",
    )
    axs[1].axis(ax_vis)
    axs[1].plot(0, 0, "r+")

    if data is not None:
        _plot_uvdata_coord(data, ax=axs[1], rotation=rotation)

    if unit_vis == "lambda":
        plt.xlabel("U [M$\lambda$]")
        plt.ylabel("V [M$\lambda$]")
    else:
        plt.xlabel("U [arcsec$^{-1}$]")
        plt.ylabel("V [arcsec$^{-1}$]")

    # plt.subplot(1, 3, 3)
    axs[2].set_title("Phase [rad]")
    axs[2].imshow(
        im_phase,
        norm=PowerNorm(1),
        origin="lower",
        extent=extent_vis,
        cmap="gist_earth",
    )
    axs[2].plot(0, 0, "r+")

    if unit_vis == "lambda":
        axs[2].set_xlabel("U [M$\lambda$]")
        axs[2].set_ylabel("V [M$\lambda$]")
    else:
        axs[2].set_xlabel("U [arcsec$^{-1}$]")
        axs[2].set_ylabel("V [arcsec$^{-1}$]")
    axs[2].axis(ax_vis)
    plt.tight_layout()
    plt.show(block=False)
    return fig


def plot_image_model(
    wl,
    base_max,
    param,
    npts=128,
    fov=400,
    blm=130,
    fwhm_apod=1e2,
    hamming=False,
    cont=False,
    p=0.5,
):
    """
    Make the image of the `multiCompoPwhl` function.

    Parameters:
    -----------
    `wl` {float}:
        Wavelength of the observation [m],\n
    `mjd` {float}:
        Date of observation [mjd],\n
    `base_max` {float}:
        Maximum baseline (of single dish) used to convolved the model
        with a gaussian PSF [m],\n
    `blm` {float}:
        Maximum baseline to limit the spatial frequencies plot,\n
    `param` {dict}:
        Dictionnary of parameters of the `multiCompoPwhl` function,\n
    `npts` {int}:
        Number of pixels in the image,\n
    `fov` {float}:
        Field of view of the image [mas],\n
    `display` {boolean}:
        If True, plots are showed.

    Outputs:
    --------
    `image_orient` {array}:
        Image of the model (oriented north/up, east/left),\n
    `ima_conv_orient` {array}:
        Convolved model with a gaussian PSF,\n
    `xScales` {array}:
        Spatial coordinates of the images,\n
    `uv_scale` {array}:
        Fourier coordinated of the amplitude/phase image,\n
    `norm_amp` {array}:
        Normalized amplitudes of the visibility,\n
    `pixel_size` {float}:
        Pixel size of the image.
    """
    # Compute base max to get the good fov
    fov = mas2rad(fov)
    bmax = (wl / fov) * npts

    pixel_size = rad2mas(fov) / npts  # Pixel size of the image [mas]

    # Creat UV coord
    UVTable = UVGrid(bmax, npts) / 2.0  # Factor 2 due to the fft
    Utable = UVTable[:, 0]
    Vtable = UVTable[:, 1]

    uv_scale = np.reshape(Utable, (npts, npts))

    modelname = param["model"]
    model_target = select_model(modelname)

    isValid = check_params_model(param)[0]

    if not isValid:
        return None

    vis = model_target(Utable, Vtable, wl, param)

    param_psf = {"fwhm": wl / (2 * base_max), "x0": 0, "y0": 0}

    conv_psf = visGaussianDisk(Utable, Vtable, wl, param_psf)

    # Apodisation
    x, y = np.meshgrid(range(npts), range(npts))
    freq_max = bmax / wl / 206264.806247 / 2.0
    pix_vis = 2 * freq_max / npts
    freq_map = np.sqrt((x - (npts / 2.0)) ** 2 + (y - (npts / 2.0)) ** 2) * pix_vis

    x = np.squeeze(np.linspace(0, 1.5 * np.sqrt(freq_max ** 2 + freq_max ** 2), npts))
    y = np.squeeze(np.exp(-(x ** 2) / (2 * (fwhm_apod / 2.355) ** 2)))

    # Can use hamming window to apodise the visibility
    if hamming:
        y = np.hamming(2 * npts)[npts:]

    f = interp1d(x, y)
    img_apod = f(freq_map.flat).reshape(freq_map.shape)

    # Reshape because all visibililty are calculated in 1D array (faster computing)
    im_vis = vis.reshape(npts, -1) * img_apod
    fftVis = np.fft.ifft2(im_vis)

    amp = abs(vis)
    phi = np.arctan2(vis.imag, vis.real)

    x_u = Utable / wl
    freq_s_x = x_u[0:npts] / 206264.806247
    extent_vis = (freq_s_x.min(), freq_s_x.max(), freq_s_x.min(), freq_s_x.max())

    # Create an image
    im_amp = amp.reshape(npts, -1)
    im_phi = phi.reshape(npts, -1)

    image = np.fft.fftshift(abs(fftVis))
    maxX = rad2mas(wl * npts / bmax) / 2.0
    xScales = np.linspace(-maxX, maxX, npts)

    extent_ima = (
        np.array((xScales.max(), xScales.min(), xScales.min(), xScales.max()))
        - pixel_size / 2.0
    )

    vis = vis * conv_psf
    im_vis = vis.reshape(npts, -1)
    fftVis = np.fft.ifft2(im_vis)
    ima_conv = abs(np.fft.fftshift(fftVis))

    tmp = np.fliplr(image)
    image_orient = tmp / np.max(tmp)
    ima_conv_orient = np.fliplr(ima_conv)

    rb = (blm / wl) / 206264.806247

    plt.figure(figsize=(13, 3.5), dpi=120)
    plt.subplots_adjust(
        left=0.05, bottom=0.05, right=0.99, top=1, wspace=0.18, hspace=0.25
    )
    plt.subplot(1, 4, 1)
    plt.imshow(im_amp, origin="lower", extent=extent_vis)
    plt.axis([-rb, rb, -rb, rb])
    plt.xlabel("Sp. Freq [cycles/arcsec]")
    plt.ylabel("Sp. Freq [cycles/arcsec]")
    plt.title("Amplitude visibility", fontsize=12, color="grey", weight="bold")
    plt.subplot(1, 4, 2)
    plt.imshow(im_phi, origin="lower", extent=extent_vis)
    plt.title("Phase visibility", fontsize=12, color="grey", weight="bold")
    plt.xlabel("Sp. Freq [cycles/arcsec]")
    plt.ylabel("Sp. Freq [cycles/arcsec]")
    plt.axis([-rb, rb, -rb, rb])
    plt.subplot(1, 4, 3)
    from matplotlib.colors import LogNorm

    plt.imshow(
        image_orient,
        cmap="jet",
        norm=PowerNorm(p),
        interpolation=None,
        extent=np.array(extent_ima),
        origin="lower",
    )

    if cont:
        cs = plt.contour(
            image_orient,
            levels=[0.5],
            colors=["r"],
            extent=np.array(extent_ima),
            origin="lower",
        )

    plt.xlabel(r"Relative R.A. [mas]")
    plt.ylabel(r"Relative DEC [mas]")
    plt.title("Model image", fontsize=12, color="grey", weight="bold")

    # plt.ylabel(r'$\delta$ [mas]')
    plt.subplot(1, 4, 4)
    plt.imshow(
        ima_conv_orient,
        cmap="afmhot",
        norm=PowerNorm(p),
        interpolation=None,
        extent=np.array(extent_ima),
        origin="lower",
    )
    plt.xlabel(r"Relative R.A. [mas]")
    plt.ylabel(r"Relative DEC [mas]")
    plt.title(
        "Model convolved B=%im" % base_max, fontsize=12, color="grey", weight="bold"
    )
    plt.subplots_adjust(
        top=0.93, bottom=0.153, left=0.055, right=0.995, hspace=0.24, wspace=0.3
    )
    norm_amp = im_amp / np.max(im_amp)
    return image_orient, ima_conv_orient, xScales, uv_scale, norm_amp, pixel_size


def plot_spectra(
    data,
    aver=False,
    offset=0,
    wl_lim=[2.16612, 0.03],
    div=False,
    f_range=None,
    tellu=False,
    title=None,
):

    spectra = data.flux
    wave_cal = data.wl
    tel = data.tel

    array_name = data.info["Array"]
    nbl = spectra.shape[0]

    n_spec = spectra.shape[0]
    l_spec, l_wave = [], []
    for i in range(n_spec):
        flux, wave = substract_run_med(spectra[i], wave_cal, div=div)
        l_spec.append(flux)
        l_wave.append(wave * 1e6 - offset)

    spec = np.array(l_spec).T
    wave = np.array(l_wave)[0]

    if aver:
        spec = np.mean(spec, axis=1)

    plt.figure(figsize=[6, 4])
    ax = plt.subplot(111)

    if aver:
        plt.plot(
            wave,
            spec,
            lw=1.5,
            label="Averaged (%s+%s+%s+%s)" % (tel[0], tel[1], tel[2], tel[3]),
        )
        plt.legend(fontsize=7)
    else:
        if array_name == "VLTI":
            ax.set_prop_cycle("color", ["#ffc258", "#c65d7b", "#3a6091", "#79ab8e"])
        for i in range(nbl):
            plt.plot(wave, spec[:, i], lw=1.5, label=tel[i])

        handles, labels = ax.get_legend_handles_labels()
        labels, handles = zip(*sorted(zip(labels, handles), key=lambda t: t[0]))
        ax.legend(handles, labels, fontsize=7)

    plt.xlim(wl_lim[0] - wl_lim[1], wl_lim[0] + wl_lim[1])
    plt.grid(alpha=0.2)
    if tellu:
        plot_tellu()
    if f_range is not None:
        plt.ylim(f_range)
    plt.xlabel(r"Wavelength [$\mu$m]")
    plt.ylabel("Normalized flux [counts]")
    if title is not None:
        plt.title(title)
    plt.tight_layout()
    return wave, spec


def plot_dvis(data, bounds=None, line=None, dvis_range=0.08, dphi_range=9):
    """
    Plot differential observables (visibility amplitude and phase).

    Parameters:
    -----------

    `data` {class}:
        Interferometric data from load()\n
    `bounds` {list}:
        Wavelengths range (by default around Br Gamma line 2.166 µm, [2.14, 2.19]),\n
    `line` {float}:
        Vertical line reference to be plotted (by default, Br Gamma line 2.166 µm)\n
    """
    if bounds is None:
        bounds = [2.14, 2.19]

    spectrum = data.flux.mean(axis=0)

    wl = data.wl * 1e6

    flux, wave = substract_run_med(spectrum, wl, div=True)

    cond_wl = (wave >= bounds[0]) & (wave <= bounds[1])
    cond_wl2 = (wl >= bounds[0]) & (wl <= bounds[1])

    flux = flux[cond_wl]
    wave = wave[cond_wl]

    dphi = data.dphi
    dvis = data.dvis
    blname = data.blname

    linestyle = {"lw": 1}

    fig = plt.figure(figsize=(4, 8.5))

    # ------ PLOT AVERAGED SPECTRUM ------
    ax = plt.subplot(13, 1, 1)
    plt.plot(wave, flux, **linestyle)
    plt.ylabel("Spec.")
    hide_xlabel()

    if line is not None:
        plot_vline(line)
        plt.text(
            0.57,
            0.8,
            r"Br$\gamma$",
            color="#eab15d",
            fontsize=8,
            ha="center",
            va="center",
            transform=ax.transAxes,
        )
    ax.tick_params(axis="both", which="major", labelsize=8)

    # ------ PLOT VISIBILITY AMPLITUDE ------
    for i in range(dvis.shape[0]):
        ax = plt.subplot(13, 1, 2 + i)
        plt.step(wl[cond_wl2], dvis[i][cond_wl2], **linestyle)
        dvis_m = dvis[i][cond_wl2].mean()
        plt.text(
            0.92,
            0.8,
            blname[i],
            fontsize=8,
            ha="center",
            va="center",
            transform=ax.transAxes,
        )
        plt.ylabel("amp.")
        hide_xlabel()
        if line is not None:
            plot_vline(line)

        ax.tick_params(axis="both", which="major", labelsize=8)
        plt.ylim(dvis_m - dvis_range, dvis_m + dvis_range)

    # ------ PLOT VISIBILITY PHASE ------
    for i in range(dphi.shape[0]):
        ax = plt.subplot(13, 1, 8 + i)
        plt.step(wl[cond_wl2], dphi[i][cond_wl2], **linestyle)
        dphi_m = dphi[i][cond_wl2].mean()
        plt.text(
            0.92,
            0.8,
            blname[i],
            fontsize=8,
            ha="center",
            va="center",
            transform=ax.transAxes,
        )
        plt.ylabel(r"$\phi$ (deg)")
        if 8 + i != 13:
            hide_xlabel()
        else:
            plt.grid(lw=0.5, alpha=0.5)
            plt.xlabel(r"$\lambda$ ($\mu$m)")
        ax.tick_params(axis="both", which="major", labelsize=8)
        if line is not None:
            plot_vline(line)
        plt.ylim(dphi_m - dphi_range, dphi_m + dphi_range)

    plt.tight_layout()
    plt.subplots_adjust(hspace=0.15, bottom=0.05, top=0.99)
    return fig
