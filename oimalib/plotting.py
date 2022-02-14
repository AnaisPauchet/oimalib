"""
@author: Anthony Soulain (University of Sydney)
-----------------------------------------------------------------
OIMALIB: Optical Interferometry Modelisation and Analysis Library
-----------------------------------------------------------------

Set of function to plot oi data, u-v plan, models, etc.
-----------------------------------------------------------------
"""
import numpy as np
import pandas as pd
import pkg_resources
import seaborn as sns
from matplotlib import pyplot as plt
from matplotlib.colors import PowerNorm
from scipy.constants import c as c_light
from scipy.interpolate import interp1d
from termcolor import cprint

from oimalib.complex_models import visGaussianDisk
from oimalib.fitting import check_params_model
from oimalib.fitting import select_model
from oimalib.fourier import UVGrid
from oimalib.modelling import compute_geom_model
from oimalib.tools import hide_xlabel
from oimalib.tools import mas2rad
from oimalib.tools import plot_vline
from oimalib.tools import rad2arcsec
from oimalib.tools import rad2mas
from oimalib.tools import substract_run_med

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
    "capsize": 1,
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
        nbl = data.blname.shape[0]
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


def plot_tellu(label=None, plot_ind=False, val=5000, lw=0.5):
    file_tellu = pkg_resources.resource_stream(
        "oimalib", "internal_data/Telluric_lines.txt"
    )
    tellu = np.loadtxt(file_tellu, skiprows=1)
    file_tellu.close()
    plt.axvline(np.nan, lw=lw, c="gray", alpha=0.5, label=label)
    for i in range(len(tellu)):
        plt.axvline(tellu[i], lw=lw, c="crimson", ls="--", alpha=0.5)
        if plot_ind:
            plt.text(tellu[i], val, i, fontsize=7, c="crimson")


def _plot_uvdata_coord(tab, ax=None, rotation=0):
    """Plot u-v coordinated of a bunch of data (see `plot_uv()`)."""
    if (type(tab) != list) & (type(tab) != np.ndarray):
        tab = [tab]

    dic_color = _update_color_bl(tab)

    list_bl = []
    for data in tab:
        nbl = data.blname.shape[0]
        for bl in range(nbl):
            flag = np.invert(data.flag_vis2[bl])
            u = data.u[bl] / data.wl[flag] / 1e6
            v = data.v[bl] / data.wl[flag] / 1e6
            base, label = data.blname[bl], ""

            vis2 = data.vis2[bl]
            if len(vis2[~np.isnan(vis2)]) == 0:
                continue

            if base not in list_bl:
                label = base
                list_bl.append(base)

            p_color = dic_color[base]
            angle = np.deg2rad(rotation)
            um = np.squeeze(u * np.cos(angle) - v * np.sin(angle))
            vm = np.squeeze(u * np.sin(angle) + v * np.cos(angle))
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
    plot_vis2=True,
    force_freq=None,
    title="",
    mega=False,
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

    if (type(tab) == list) | (type(tab) == np.ndarray):
        data = tab[0]
    else:
        data = tab
        tab = [tab]

    dic_color = _update_color_bl(tab)

    array_name = data.info["Array"]
    l_fmin, l_fmax = [], []

    list_triplet = []
    for _ in tab:
        for i in range(len(_.cpname)):
            list_triplet.append(_.cpname[i])
    list_triplet = np.array(list_triplet)

    for _ in tab:
        tfmax = _.freq_vis2.flatten().max()
        tmp = _.freq_vis2.flatten()
        tfmin = tmp[tmp != 0].min()
        l_fmax.append(tfmax)
        l_fmin.append(tfmin)

    ff = 1
    if mega:
        ff = (1.0 / mas2rad(1000)) / 1e6
    l_fmin = np.array(l_fmin) * ff
    l_fmax = np.array(l_fmax) * ff

    if len(l_fmin) == 1:
        fmin = l_fmin[0]
    else:
        fmin = np.min(l_fmin)
    fmax = l_fmax.max()

    ncp_master = len(set(list_triplet))

    fig = plt.figure(figsize=(8, 6))
    ax1 = plt.subplot2grid((5, 1), (0, 0), rowspan=3)
    ax1.set_title(title, fontsize=14)
    list_bl = []
    # PLOT VIS2 DATA AND MODEL IF ANY (mod_v2)
    # ----------------------------------------

    if plot_vis2:
        ylabel = r"V$^2$"
    else:
        ylabel = r"Vis. Amp."

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
            if plot_vis2:
                vis2 = data.vis2[i][sel_flag]
                e_vis2 = data.e_vis2[i][sel_flag]
            else:
                vis2 = data.dvis[i][sel_flag]
                e_vis2 = data.e_dvis[i][sel_flag]

            base, label = data.blname[i], ""
            wave = data.wl[sel_flag]

            if len(vis2[~np.isnan(vis2)]) == 0:
                continue

            if base not in list_bl:
                label = base
                list_bl.append(base)

            if mega:
                freq_vis2 = freq_vis2 * (1.0 / mas2rad(1000)) / 1e6

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
                        **err_pts_style,
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
                if plot_vis2:
                    mod = mod_v2[j][i][sel_flag]
                else:
                    mod = mod_v2[j][i][sel_flag] ** 0.5
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

    if force_freq is not None:
        fmin, fmax = force_freq[0], force_freq[1]
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
        cb.ax.set_title(r"$\lambda$ [µm]", fontsize=9)
    ax1.set_ylabel(ylabel, fontsize=12)
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
                    "color",
                    [
                        "#f4dfcc",
                        "#fa9583",
                        "#2f4159",
                        "#4097aa",
                        "#82b4bb",
                        "#ae3c60",
                        "#eabd6f",
                        "#96d47c",
                    ],
                )
            elif set_cp == 1:
                ax2.set_prop_cycle(
                    "color",
                    [
                        "#eabd6f",
                        "#fa9583",
                        "#3a6091",
                        "#79ab8e",
                        "#82b4bb",
                        "#ae3c60",
                        "#eabd6f",
                        "#96d47c",
                    ],
                )
            elif set_cp == 2:
                ax2.set_prop_cycle(
                    "color", ["#79ab8e", "#5c95a8", "#fa9583", "#263a55"]
                )
            else:
                ax2.set_prop_cycle("color", plt.cm.turbo(np.linspace(0, 1, ncp_master)))

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
        color_cp = None
        for i in range(ncp):
            if use_flag:
                sel_flag = np.invert(data.flag_cp[i])
            else:
                sel_flag = np.array([True] * nwl)

            freq_cp = data.freq_cp[i][sel_flag]
            cp = data.cp[i][sel_flag]
            e_cp = data.e_cp[i][sel_flag]
            wave = data.wl[sel_flag]

            if mega:
                freq_cp = freq_cp * (1.0 / mas2rad(1000)) / 1e6

            if len(cp[~np.isnan(cp)]) == 0:
                continue

            dic_index = _create_match_tel(data)
            b1 = dic_index[data.index_cp[i][0]]
            b2 = dic_index[data.index_cp[i][1]]
            b3 = dic_index[data.index_cp[i][2]]
            triplet = f"{b1}-{b2}-{b3}"

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

    if (type(tab) == list) or (type(tab) == np.ndarray):
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
    data, param, fitOnly=None, hue=None, use_flag=True, save_dir=None, name=""
):
    sns.set_theme(color_codes=True)
    if fitOnly is None:
        print("Warning: FitOnly is None, the degree of freedom is set to 0.\n")
        fitOnly = []
    else:
        if len(fitOnly) == 0:
            print("Warning: FitOnly is empty, the degree of freedom is set to 0.\n")

    if type(data) is not list:
        data = [data]

    param_plot = {
        "data": data,
        "param": param,
        "fitOnly": fitOnly,
        "hue": hue,
        "use_flag": use_flag,
    }
    df_cp, chi2_cp, chi2_cp_full, mod_cp = plot_cp_residuals(**param_plot)
    if save_dir is not None:
        plt.savefig(save_dir + "residuals_CP_%sfit.png" % name, dpi=300)
    df_v2, chi2_vis2, chi2_vis2_full, mod_v2 = plot_v2_residuals(**param_plot)
    if save_dir is not None:
        plt.savefig(save_dir + "residuals_V2_%sfit.png" % name, dpi=300)

    d_freedom = len(fitOnly)

    nv2 = len(df_v2["vis2"])
    ncp = len(df_cp["cp"])
    nobs = nv2 + ncp
    obs = np.zeros(nobs)
    e_obs = np.zeros(nobs)
    all_mod = np.zeros(nobs)

    for i in range(len(df_v2["vis2"])):
        obs[i] = df_v2["vis2"][i]
        e_obs[i] = df_v2["e_vis2"][i]
        all_mod[i] = df_v2["mod"][i]
    for i in range(len(df_cp["cp"])):
        obs[i + nv2] = df_cp["cp"][i]
        e_obs[i + nv2] = df_cp["e_cp"][i]
        all_mod[i + nv2] = df_cp["mod"][i]

    chi2_global = np.sum((obs - all_mod) ** 2 / (e_obs) ** 2) / (nobs - (d_freedom - 1))
    title = "Statistic of the model %s" % param["model"]
    print(title)
    print("-" * len(title))
    print(f"χ² = {chi2_global:2.2f} (V² = {chi2_vis2:2.1f}, CP = {chi2_cp:2.1f})")
    return chi2_global, chi2_vis2, chi2_cp, mod_v2, mod_cp, chi2_vis2_full, chi2_cp_full


def plot_v2_residuals(data, param, fitOnly=None, hue=None, use_flag=True):
    if fitOnly is None:
        fitOnly = []
    mod_v2 = compute_geom_model(data, param)[0]

    input_keys = ["vis2", "e_vis2", "freq_vis2", "wl", "blname", "set", "flag_vis2"]

    dict_obs = {}
    for k in input_keys:
        dict_obs[k] = []

    nobs = 0
    for d in data:
        for k in input_keys:
            nbl = d.vis2.shape[0]
            nwl = d.vis2.shape[1]
            if k == "wl":
                for _ in range(nbl):
                    dict_obs[k].extend(np.round(d[k] * 1e6, 3))
            elif k == "blname":
                for j in range(nbl):
                    for _ in range(nwl):
                        dict_obs[k].append(d[k][j])
            elif k == "set":
                for _ in range(nbl):
                    for _ in range(nwl):
                        dict_obs[k].append(nobs)
            else:
                dict_obs[k].extend(d[k].flatten())
        nobs += 1

    dict_obs["mod"] = np.array(mod_v2).flatten()

    dict_obs["res"] = (dict_obs["vis2"] - dict_obs["mod"]) / dict_obs["e_vis2"]

    flag = np.array(dict_obs["flag_vis2"])
    flag_nan = np.isnan(np.array(dict_obs["vis2"]))

    if use_flag:
        for k in dict_obs.keys():
            dict_obs[k] = np.array(dict_obs[k])[~flag & ~flag_nan]

    df = pd.DataFrame(dict_obs)

    d_freedom = len(fitOnly)

    chi2_vis2_full = np.sum((df["vis2"] - df["mod"]) ** 2 / (df["e_vis2"]) ** 2)
    chi2_vis2 = chi2_vis2_full / (len(df["e_vis2"]) - (d_freedom - 1))

    label = "DATA"
    if hue == "wl":
        label = "Wavelenght [µm]"

    fig = plt.figure(constrained_layout=False, figsize=(7, 5))
    axd = fig.subplot_mosaic(
        [["vis2"], ["res_vis2"]],
        gridspec_kw={"height_ratios": [3, 1]},
    )
    ax = sns.scatterplot(
        x="freq_vis2",
        y="vis2",
        data=df,
        palette="crest",
        zorder=10,
        label=label,
        ax=axd["vis2"],
        style=None,
        hue=hue,
    )
    sns.scatterplot(
        x="freq_vis2",
        y="mod",
        data=df,
        color="#e19751",
        zorder=10,
        marker="^",
        label=r"MODEL ($\chi^2_r=%2.2f$)" % chi2_vis2,
        ax=axd["vis2"],
    )
    ax.errorbar(
        df.freq_vis2,
        df.vis2,
        yerr=df.e_vis2,
        fmt="None",
        zorder=1,
        color="gray",
        alpha=0.4,
        capsize=2,
    )
    sns.scatterplot(
        x="freq_vis2",
        y="res",
        data=df,
        zorder=10,
        ax=axd["res_vis2"],
    )
    axd["res_vis2"].sharex(axd["vis2"])
    plt.xlabel(r"Sp. Freq. [arcsec$^{-1}$]")
    axd["vis2"].set_ylabel("V$^{2}$")
    axd["vis2"].set_xlabel("")
    axd["res_vis2"].set_ylabel(r"Residuals [$\sigma$]")
    axd["res_vis2"].axhspan(-1, 1, alpha=0.6, color="#418fde")
    axd["res_vis2"].axhspan(-2, 2, alpha=0.6, color="#8bb8e8")
    axd["res_vis2"].axhspan(-3, 3, alpha=0.6, color="#c8d8eb")
    axd["res_vis2"].set_ylim(-5, 5)

    axd["vis2"].tick_params(
        axis="x",  # changes apply to the x-axis
        which="major",  # both major and minor ticks are affected
        bottom=False,  # ticks along the bottom edge are off
        top=False,  # ticks along the top edge are off
        labelbottom=False,  # labels along the bottom edge are off)
    )
    plt.subplots_adjust(hspace=0.1, top=0.98, right=0.98, left=0.11)
    return df, chi2_vis2, chi2_vis2_full, mod_v2


def plot_cp_residuals(data, param, fitOnly=None, hue=None, use_flag=True):
    if fitOnly is None:
        fitOnly = []
    mod_cp = compute_geom_model(data, param)[1]

    input_keys = ["cp", "e_cp", "freq_cp", "wl", "cpname", "set", "flag_cp"]

    dict_obs = {}
    for k in input_keys:
        dict_obs[k] = []

    nobs = 0
    for d in data:
        for k in input_keys:
            nbl = d.cp.shape[0]
            nwl = d.cp.shape[1]
            if k == "wl":
                for _ in range(nbl):
                    dict_obs[k].extend(np.round(d[k] * 1e6, 3))
            elif k == "cpname":
                for j in range(nbl):
                    for _ in range(nwl):
                        dict_obs[k].append(d[k][j])
            elif k == "set":
                for _ in range(nbl):
                    for _ in range(nwl):
                        dict_obs[k].append(nobs)
            else:
                dict_obs[k].extend(d[k].flatten())
        nobs += 1

    dict_obs["mod"] = np.array(mod_cp).flatten()
    dict_obs["res"] = (dict_obs["cp"] - dict_obs["mod"]) / dict_obs["e_cp"]

    if use_flag:
        flag = np.array(dict_obs["flag_cp"])
        flag_nan = np.isnan(np.array(dict_obs["cp"]))
        for k in dict_obs.keys():
            dict_obs[k] = np.array(dict_obs[k])[~flag & ~flag_nan]

    df = pd.DataFrame(dict_obs)

    d_freedom = len(fitOnly)

    chi2_cp_full = np.sum((df["cp"] - df["mod"]) ** 2 / (df["e_cp"]) ** 2)
    chi2_cp = chi2_cp_full / (len(df["e_cp"]) - (d_freedom - 1))

    res_max = 5
    if np.max(abs(df["res"])) >= 5:
        res_max = abs(df["res"]).max() * 1.2

    fig = plt.figure(constrained_layout=False, figsize=(7, 5))
    axd = fig.subplot_mosaic(
        [["cp"], ["res_cp"]],
        gridspec_kw={"height_ratios": [3, 1]},
    )
    label = "DATA"
    if hue == "wl":
        label = "Wavelenght [µm]"
    ax = sns.scatterplot(
        x="freq_cp",
        y="cp",
        data=df,
        palette="crest",
        zorder=10,
        label=label,
        ax=axd["cp"],
        style=None,
        hue=hue,
    )
    sns.scatterplot(
        x="freq_cp",
        y="mod",
        data=df,
        color="#e19751",
        zorder=10,
        marker="^",
        label=r"MODEL ($\chi^2_r=%2.2f$)" % chi2_cp,
        ax=axd["cp"],
    )
    ax.errorbar(
        df.freq_cp,
        df.cp,
        yerr=df.e_cp,
        fmt="None",
        zorder=1,
        color="gray",
        alpha=0.4,
        capsize=2,
    )
    sns.scatterplot(
        x="freq_cp",
        y="res",
        data=df,
        zorder=10,
        ax=axd["res_cp"],
    )
    axd["res_cp"].sharex(axd["cp"])
    plt.xlabel(r"Sp. Freq. [arcsec$^{-1}$]")
    axd["cp"].set_ylabel(r"Closure phase $\phi$ [deg]")
    axd["cp"].set_xlabel("")
    axd["res_cp"].set_ylabel(r"Residuals [$\sigma$]")
    axd["res_cp"].axhspan(-1, 1, alpha=0.6, color="#418fde")
    axd["res_cp"].axhspan(-2, 2, alpha=0.6, color="#8bb8e8")
    axd["res_cp"].axhspan(-3, 3, alpha=0.6, color="#c8d8eb")
    axd["res_cp"].set_ylim(-res_max, res_max)

    axd["cp"].tick_params(
        axis="x",  # changes apply to the x-axis
        which="major",  # both major and minor ticks are affected
        bottom=False,  # ticks along the bottom edge are off
        top=False,  # ticks along the top edge are off
        labelbottom=False,  # labels along the bottom edge are off)
    )
    plt.subplots_adjust(hspace=0.1, top=0.98, right=0.98, left=0.11)
    return df, chi2_cp, chi2_cp_full, mod_cp


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
    """Plot model and corresponding visibility and phase plan. Additionallly, you
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

    fig, axs = plt.subplots(1, 3, figsize=(14, 5))
    axs[0].set_title(fr'Model "{modelname}" ($\lambda$ = {wl_model * 1e6:2.2f} $\mu$m)')
    axs[0].imshow(
        im_model, norm=PowerNorm(p), origin="lower", extent=extent_im, cmap="afmhot"
    )
    axs[0].set_xlabel(r"$\Delta$RA [%s]" % (unit_im))
    axs[0].set_ylabel(r"$\Delta$DEC [%s]" % (unit_im))

    axs[1].set_title(r"Squared visibilities (V$^2$)")
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
        plt.xlabel(r"U [M$\lambda$]")
        plt.ylabel(r"V [M$\lambda$]")
    else:
        plt.xlabel(r"U [arcsec$^{-1}$]")
        plt.ylabel(r"V [arcsec$^{-1}$]")

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
        axs[2].set_xlabel(r"U [M$\lambda$]")
        axs[2].set_ylabel(r"V [M$\lambda$]")
    else:
        axs[2].set_xlabel(r"U [arcsec$^{-1}$]")
        axs[2].set_ylabel(r"V [arcsec$^{-1}$]")
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
    hamming=True,
    cont=False,
    p=0.5,
    obs=None,
    apod=False,
    corono=False,
    expert_plot=False,
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
    sns.reset_orig()
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

    isValid, log = check_params_model(param)
    if not isValid:
        cprint("Model %s not valid:" % param["model"], "cyan")
        cprint(log, "cyan")
        return None

    if param["model"] == "pwhl":
        vis = model_target(Utable, Vtable, wl, param, expert_plot=expert_plot)
    else:
        vis = model_target(Utable, Vtable, wl, param)

    param_psf = {"fwhm": rad2mas(wl / (2 * base_max)), "x0": 0, "y0": 0}

    conv_psf = visGaussianDisk(Utable, Vtable, wl, param_psf)

    # Apodisation
    x, y = np.meshgrid(range(npts), range(npts))
    freq_max = rad2arcsec(bmax / wl) / 2.0
    pix_vis = 2 * freq_max / npts
    freq_map = np.sqrt((x - (npts / 2.0)) ** 2 + (y - (npts / 2.0)) ** 2) * pix_vis

    x = np.squeeze(np.linspace(0, 1.5 * np.sqrt(freq_max ** 2 + freq_max ** 2), npts))
    y = np.squeeze(np.exp(-(x ** 2) / (2 * (fwhm_apod / 2.355) ** 2)))

    # Can use hamming window to apodise the visibility
    if hamming:
        y = np.hamming(2 * npts)[npts:]

    f = interp1d(x, y)
    img_apod = 1

    if apod:
        img_apod = f(freq_map.flat).reshape(freq_map.shape)

    # Reshape because all visibililty are calculated in 1D array (faster computing)
    im_vis = vis.reshape(npts, -1) * img_apod
    fftVis = np.fft.ifft2(im_vis)

    amp = abs(vis)
    phi = np.arctan2(vis.imag, vis.real)

    x_u = Utable / wl
    freq_s_x = rad2arcsec(x_u[0:npts])
    extent_vis = (freq_s_x.min(), freq_s_x.max(), freq_s_x.min(), freq_s_x.max())

    # Create an image
    im_amp = amp.reshape(npts, -1)
    im_phi = np.rad2deg(phi.reshape(npts, -1))

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
    ima_conv_orient /= np.max(ima_conv_orient)
    rb = rad2arcsec(2 * blm / wl)

    # image_orient[image_orient < 0.02] = 0
    # corono = True
    if corono:
        image_orient[npts // 2, npts // 2 - 1] = 0
        image_orient /= np.max(image_orient)

    plt.figure(figsize=(13, 3.5), dpi=120)
    plt.subplots_adjust(
        left=0.05, bottom=0.05, right=0.99, top=1, wspace=0.18, hspace=0.25
    )
    plt.subplot(1, 4, 1)
    plt.imshow(im_amp ** 2, origin="lower", extent=extent_vis)
    if obs is not None:
        save_obs = obs.copy()
        cond = save_obs[:, 1] == "V2"
        obs = save_obs[cond]
        for i in range(len(obs)):
            u = obs[i][0][0]
            v = obs[i][0][1]
            wl = obs[i][0][2]
            u_freq = rad2arcsec(u / wl)  # / (1/mas2rad(1000))
            v_freq = rad2arcsec(v / wl)  # / (1/mas2rad(1000))
            plt.scatter(u_freq, v_freq, s=4, marker="o", alpha=0.3, color="r")
            plt.scatter(-u_freq, -v_freq, s=4, marker="o", alpha=0.3, color="r")
    plt.axis([-rb, rb, -rb, rb])
    plt.xlabel("Sp. Freq [cycles/arcsec]")
    plt.ylabel("Sp. Freq [cycles/arcsec]")
    plt.title("Amplitude visibility", fontsize=12, color="grey", weight="bold")
    plt.subplot(1, 4, 2)
    plt.imshow(im_phi, origin="lower", extent=extent_vis)
    if obs is not None:
        save_obs = obs.copy()
        cond = save_obs[:, 1] == "V2"
        obs = save_obs[cond]
        for i in range(len(obs)):
            u = obs[i][0][0]
            v = obs[i][0][1]
            wl = obs[i][0][2]
            u_freq = rad2arcsec(u / wl)  # / (1/mas2rad(1000))
            v_freq = rad2arcsec(v / wl)  # / (1/mas2rad(1000))
            plt.scatter(u_freq, v_freq, s=4, marker="o", alpha=0.3, color="r")
            plt.scatter(-u_freq, -v_freq, s=4, marker="o", alpha=0.3, color="r")
    plt.title("Phase visibility", fontsize=12, color="grey", weight="bold")
    plt.xlabel("Sp. Freq [cycles/arcsec]")
    plt.ylabel("Sp. Freq [cycles/arcsec]")
    plt.axis([-rb, rb, -rb, rb])

    plt.subplot(1, 4, 3)

    plt.imshow(
        image_orient,
        cmap="turbo",
        norm=PowerNorm(p),
        interpolation=None,
        extent=np.array(extent_ima),
        origin="lower",
    )

    if cont:
        plt.contour(
            image_orient,
            levels=[0.5],
            colors=["r"],
            extent=np.array(extent_ima),
            origin="lower",
        )

    plt.xlabel(r"Relative R.A. [mas]")
    plt.ylabel(r"Relative DEC [mas]")
    plt.title("Model image", fontsize=12, color="grey", weight="bold")

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
    wl_lim=None,
    div=False,
    f_range=None,
    tellu=False,
    title=None,
    rest=0,
    speed=False,
    d_speed=1000,
    norm=True,
):
    if wl_lim is None:
        wl_lim = [2.16612, 0.03]

    spectra = data.flux
    wave_cal = data.wl
    tel = data.tel
    # ins = data.info.Ins

    array_name = data.info["Array"]
    nbl = spectra.shape[0]

    n_spec = spectra.shape[0]
    l_spec, l_wave = [], []
    for i in range(n_spec):
        if norm:
            flux, wave = substract_run_med(spectra[i], wave_cal, div=div)
        else:
            flux, wave = spectra[i], wave_cal
        l_spec.append(flux)
        l_wave.append(wave * 1e6 - offset)

    spec = np.array(l_spec).T

    if speed:
        wave = ((np.array(l_wave)[0] - rest) / rest) * c_light / 1e3
    else:
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
            label=f"Averaged ({tel[0]}+{tel[1]}+{tel[2]}+{tel[3]})",
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

    if not speed:
        plt.xlim(wl_lim[0] - wl_lim[1], wl_lim[0] + wl_lim[1])
    else:
        plt.xlim(-d_speed, d_speed)

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

    bounds2 = [bounds[0] - 0.001, bounds[1] + 0.001]

    if len(data.flux.shape) == 1:
        spectrum = data.flux
    else:
        spectrum = data.flux.mean(axis=0)

    wl = data.wl * 1e6

    try:
        flux, wave = substract_run_med(spectrum, wl, div=True)
    except IndexError:
        flux, wave = spectrum, wl

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
    hide_xlabel()
    plt.xlim(bounds2)
    # ------ PLOT VISIBILITY AMPLITUDE ------
    for i in range(dvis.shape[0]):
        ax = plt.subplot(13, 1, 2 + i, sharex=ax)

        data_dvis = dvis[i][cond_wl2]
        dvis_m = data_dvis[~np.isnan(data_dvis)].mean()

        if not np.isnan(dvis_m):
            plt.step(wl[cond_wl2], data_dvis, **linestyle)
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
            if line is not None:
                plot_vline(line)

            plt.ylim(dvis_m - dvis_range, dvis_m + dvis_range)
            ax.tick_params(axis="both", which="major", labelsize=8)
            hide_xlabel()
            plt.xlim(bounds2)
        else:
            # frame1 = plt.gca()
            # frame1.axes.get_xaxis().set_visible(False)
            # frame1.axes.get_yaxis().set_visible(False)
            plt.xlim(bounds2)
            plt.xticks([])
            plt.yticks([])
            plt.ylabel("amp.")
            plt.text(
                0.92,
                0.8,
                blname[i],
                fontsize=8,
                ha="center",
                va="center",
                transform=ax.transAxes,
            )

            plt.text(
                0.5,
                0.5,
                "Not available",
                color="red",
                fontsize=8,
                ha="center",
                va="center",
                transform=ax.transAxes,
            )

    # ------ PLOT VISIBILITY PHASE ------
    for i in range(dphi.shape[0]):
        ax = plt.subplot(13, 1, 8 + i, sharex=ax)

        if np.diff(dphi[i][cond_wl2]).mean() != 0:
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
            plt.xlim(bounds2)
        else:
            if 8 + i != 13:
                plt.xticks([])
                plt.yticks([])
            else:
                plt.xlabel(r"$\lambda$ ($\mu$m)")
                ax.tick_params(axis="both", which="major", labelsize=8)

            plt.ylabel(r"$\phi$ (deg)")
            plt.text(
                0.92,
                0.8,
                blname[i],
                fontsize=8,
                ha="center",
                va="center",
                transform=ax.transAxes,
            )

            plt.text(
                0.5,
                0.5,
                "Not available",
                color="red",
                fontsize=8,
                ha="center",
                va="center",
                transform=ax.transAxes,
            )
            plt.xlim(bounds2)

    plt.tight_layout()
    plt.subplots_adjust(hspace=0.15, bottom=0.05, top=0.99)
    return fig


def _summary_corner_sns(x, prec=2, color="#ee9068", **kwargs):
    t_unit = {
        "f$_c$": "%",
        "incl": "deg",
        "i": "deg",
        "a$_r*$": r"r$_{star}$",
        "a$_r$": "mas",
        "PA": "deg",
        "c$_j$": "",
        "s$_j$": "",
        "l$_a$": "",
        "$r$": "AU",
    }
    mcmc = np.percentile(x, [16, 50, 84])
    q = np.diff(mcmc)
    txt = r"{3} = {0:.%if}$_{{-{1:.%if}}}^{{+{2:.%if}}}$ {4}" % (prec, prec, prec)
    try:
        txt = txt.format(mcmc[1], q[0], q[1], x.name, t_unit[x.name])
    except KeyError:
        txt = txt.format(mcmc[1], q[0], q[1], x.name, "")
    ax = plt.gca()
    ax.set_axis_off()
    ax.axvline(mcmc[0], lw=1, color=color, alpha=0.8, ls="--")
    ax.axvline(mcmc[1], lw=1, color=color, alpha=0.8, ls="-")
    ax.axvline(mcmc[2], lw=1, color=color, alpha=0.8, ls="--")
    ax.set_title(txt, fontsize=9)


def _results_corner_sns(x, y, color="#ee9068", **kwargs):
    p1 = np.percentile(x, [16, 50, 84])
    p2 = np.percentile(y, [16, 50, 84])
    ax = plt.gca()
    ax.plot(p1[1], p2[1], "s", color=color, alpha=0.8)
    ax.axvline(p1[1], lw=1, color=color, alpha=0.8)
    ax.axhline(p2[1], lw=1, color=color, alpha=0.8)


def plot_mcmc_results(
    sampler,
    labels=None,
    burnin=200,
    compute_r=False,
    dpc=None,
    lk=None,
    prec=2,
    compute_w=False,
):
    """Plot modern corner plot using seaborn."""
    sns.set_theme(color_codes=True)
    flat_samples = sampler.get_chain(discard=burnin, thin=15, flat=True)

    dict_mcmc = {}
    for i in range(len(labels)):
        f = 1
        if (labels[i] == "f$_c$") or (labels[i] == "f$_h$"):
            f = 100
            dict_mcmc[labels[i]] = flat_samples[:-1, i] * f
        elif labels[i] == "l$_a$":
            if lk is not None:
                ar = 10 ** flat_samples[:-1, i]
                dict_mcmc["a"] = ar
        else:
            dict_mcmc[labels[i]] = flat_samples[:-1, i]

    try:
        if lk is None:
            lk = flat_samples[:-1, np.where(np.array(labels) == "l$_k$")[0][0]]
        la = flat_samples[:-1, np.where(np.array(labels) == "l$_a$")[0][0]]
        ar = 10 ** la / (np.sqrt(1 + 10 ** (2 * lk)))
        ak = ar * (10 ** lk)
        a = (ar ** 2 + ak ** 2) ** 0.5
        dict_mcmc["a"] = a
        w = ak / a
        if compute_w:
            dict_mcmc["w"] = w
    except IndexError:
        pass

    try:
        del dict_mcmc["l$_k$"]
    except KeyError:
        pass

    if compute_r:
        if dpc is None:
            raise TypeError("Distance (dpc) is required to compute the radius in AU.")
        ar = dict_mcmc["a"]
        dict_mcmc["$r$"] = ar * dpc  # * 215.0 / 2.0
        try:
            del dict_mcmc["l$_a$"]
        except KeyError:
            pass
        del dict_mcmc["a"]

    df = pd.DataFrame(dict_mcmc)

    color = "#ee9068"
    g = sns.PairGrid(df, corner=True, height=1.7)
    g.map_lower(sns.histplot, bins=40, pthresh=0.0)
    g.map_diag(sns.histplot, bins=20, element="step", linewidth=1, kde=True, alpha=0.5)
    g.map_diag(_summary_corner_sns, color=color, prec=prec)
    g.map_lower(_results_corner_sns, color=color)
    plt.subplots_adjust(wspace=0.05, hspace=0.05)
    return g
