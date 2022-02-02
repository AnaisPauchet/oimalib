"""
@author: Anthony Soulain (University of Sydney)
-----------------------------------------------------------------
OIMALIB: Optical Interferometry Modelisation and Analysis Library
-----------------------------------------------------------------

OIFITS related function.
-----------------------------------------------------------------
"""
import sys
from glob import glob

import numpy as np
from astropy.io import fits
from munch import munchify as dict2class

from oimalib.plotting import dic_color


def _compute_dic_index(index_ref, teles_ref):
    dic_index = {}
    for i in range(len(index_ref)):
        ind = index_ref[i]
        tel = teles_ref[i]
        if ind not in dic_index.keys():
            dic_index[ind] = tel
    return dic_index


def _compute_bl_name(index, index_ref, teles_ref):
    """Compute baseline name and check if the appropriate color
    is already associated (for the VLTI)."""
    dic_index = _compute_dic_index(index_ref, teles_ref)

    list_bl_name = []
    nbl = len(index)
    for i in range(nbl):
        base = f"{dic_index[index[i][0]]}-{dic_index[index[i][1]]}"
        base2 = f"{dic_index[index[i][1]]}-{dic_index[index[i][0]]}"
        if base in list(dic_color.keys()):
            baseline_name = base
        elif base2 in list(dic_color.keys()):
            baseline_name = base2
        else:
            baseline_name = base
        list_bl_name.append(baseline_name)
    list_bl_name = np.array(list_bl_name)
    return list_bl_name


def _compute_cp_name(index_cp, index_ref, teles_ref):
    """Compute triplet name and check if the appropriate color
    is already associated (for the VLTI)."""
    ncp = len(index_cp)
    dic_index = _compute_dic_index(index_ref, teles_ref)

    list_cp_name = []
    for i in range(ncp):
        b1 = dic_index[index_cp[i][0]]
        b2 = dic_index[index_cp[i][1]]
        b3 = dic_index[index_cp[i][2]]
        triplet = f"{b1}-{b2}-{b3}"
        list_cp_name.append(triplet)
    list_cp_name = np.array(list_cp_name)
    return list_cp_name


def oifits2dic(filename, rad=False):
    """
    Read an OiFits file and store observables (CP, V2, Vis, informations, etc,)
    in a dictionary format with keys corresponding to the standard oifits format.
    """
    fitsHandler = fits.open(filename)

    hdrr = fitsHandler[0].header

    for hdu in fitsHandler[1:]:
        if hdu.header["EXTNAME"] in ["OI_T3", "OI_VIS2"]:
            ins = hdu.header["INSNAME"]

    # -----------------------------------
    #            initOiData
    # -----------------------------------
    wavel = {}  # wave tables for each instrumental setup
    telArray = {}
    wlOffset = 0.0

    # -- load Wavelength and Array: ----------------------------------------------
    for hdu in fitsHandler[1:]:
        if hdu.header["EXTNAME"] == "OI_WAVELENGTH":
            ins = hdu.header["INSNAME"]
            wavel[ins] = wlOffset + hdu.data["EFF_WAVE"] * 1e6  # in um
        if hdu.header["EXTNAME"] == "OI_ARRAY":
            name = hdu.header["ARRNAME"]
            diam = hdu.data["DIAMETER"].mean()
            config = hdu.data["STA_NAME"]
            index = hdu.data["STA_INDEX"]
            if diam == 0:
                if "VLTI" in name:
                    if "AT" in hdu.data["TEL_NAME"][0]:
                        diam = 1.8
                    if "UT" in hdu.data["TEL_NAME"][0]:
                        diam = 8
            telArray[name] = diam
        if hdu.header["EXTNAME"] == "OI_TARGET":
            target = hdu.data["TARGET"]

    tab_data = {}
    for hdu in fitsHandler[1:]:
        if hdu.header["EXTNAME"] == "OI_T3":
            ins = hdu.header["INSNAME"]
            # if 'GRAVITY' in ins:
            #     if ('FT' not in ins) and ('SC' not in ins):
            #         ins = 'GRAVITY_simu'
            # elif'PIONIER' in ins:
            #     ins = 'PIONIER'

            # ----------------------------
            #       Closure phase
            # ----------------------------
            if rad:
                data = np.rad2deg(hdu.data["T3PHI"])
            else:
                data = hdu.data["T3PHI"]
            if len(data.shape) == 1:
                data = np.array([np.array([d]) for d in data])

            if rad:
                err = np.rad2deg(hdu.data["T3PHIERR"])
            else:
                err = hdu.data["T3PHIERR"]

            if len(err.shape) == 1:
                err = np.array([np.array([e]) for e in err])
            if np.sum(np.isnan(data)) < data.size:
                temp = {}
                temp["U1COORD"] = hdu.data["U1COORD"][:, None] + 0 * wavel[ins][None, :]
                temp["V1COORD"] = hdu.data["V1COORD"][:, None] + 0 * wavel[ins][None, :]
                temp["U2COORD"] = hdu.data["U2COORD"][:, None] + 0 * wavel[ins][None, :]
                temp["V2COORD"] = hdu.data["V2COORD"][:, None] + 0 * wavel[ins][None, :]
                temp["STA_INDEX"] = hdu.data["STA_INDEX"]
                temp["wavel"] = wavel[ins][None, :][0]
                temp["MJD"] = hdu.data["MJD"][0]
                temp["data"] = data
                ncp_master = len(data)
                temp["flag"] = hdu.data["FLAG"]
                temp["err"] = err
                tab_data["cp_phi; " + ins] = temp
            else:
                print(" | WARNING: no valid T3PHI values in this HDU")
            # ----------------------------
            #      Closure amplitude
            # ----------------------------
            data = hdu.data["T3AMP"]
            if len(data.shape) == 1:
                data = np.array([np.array([d]) for d in data])
            err = hdu.data["T3AMPERR"]
            if len(err.shape) == 1:
                err = np.array([np.array([e]) for e in err])
            if np.sum(np.isnan(data)) < data.size:
                temp = {}
                temp["U1COORD"] = hdu.data["U1COORD"][:, None] + 0 * wavel[ins][None, :]
                temp["V1COORD"] = hdu.data["V1COORD"][:, None] + 0 * wavel[ins][None, :]
                temp["U2COORD"] = hdu.data["U2COORD"][:, None] + 0 * wavel[ins][None, :]
                temp["V2COORD"] = hdu.data["V2COORD"][:, None] + 0 * wavel[ins][None, :]
                temp["wavel"] = wavel[ins][None, :][0]
                temp["MJD"] = hdu.data["MJD"][0]
                temp["err"] = err
                temp["data"] = data
                tab_data["cp_amp; " + ins] = temp
            else:
                print(" | WARNING: no valid T3AMP values in this HDU")
        if hdu.header["EXTNAME"] == "OI_VIS2":
            ins = hdu.header["INSNAME"]
            # ----------------------------
            #      Squared Vis. (V2)
            # ----------------------------
            data = hdu.data["VIS2DATA"]
            if len(data.shape) == 1:
                data = np.array([np.array([d]) for d in data])
            err = hdu.data["VIS2ERR"]
            if len(err.shape) == 1:
                err = np.array([np.array([e]) for e in err])
            temp = {}
            temp["UCOORD"] = hdu.data["UCOORD"][:, None] + 0 * wavel[ins][None, :]
            temp["VCOORD"] = hdu.data["VCOORD"][:, None] + 0 * wavel[ins][None, :]
            temp["STA_INDEX"] = hdu.data["STA_INDEX"]
            temp["wavel"] = wavel[ins][None, :][0]
            temp["MJD"] = hdu.data["MJD"][0]
            temp["err"] = err
            temp["data"] = data
            nbl_master = len(data)
            temp["flag"] = hdu.data["FLAG"]
            tab_data["Vis2; " + ins] = temp

            l_B = []
            for i in range(np.shape(temp["UCOORD"])[0]):
                U = temp["UCOORD"][i]
                V = temp["VCOORD"][i]
                l_B.append(np.sqrt(U ** 2 + V ** 2))

            if ins == "VEGA":
                b_name = []
                for x in np.unique(hdu.data["STA_INDEX"]):
                    b_name.append(config[x])
                config = b_name

            # -------- INFO ---------
            Bmax = np.sqrt(hdu.data["UCOORD"] ** 2 + hdu.data["VCOORD"] ** 2)
            Bmax = Bmax.max()
            temp2 = {}
            temp2["Bmax"] = Bmax
            temp2["hdr"] = hdrr
            temp2["Ins"] = ins
            temp2["Index"] = index
            temp2["Config"] = config
            temp2["Array"] = name
            temp2["Target"] = target
            temp2["L_base"] = l_B
            temp2["nbl"] = nbl_master

            try:
                temp2["Date"] = fitsHandler[0].header["MJD-OBS"]
            except KeyError:
                temp2["Date"] = None
                pass
            tab_data["info"] = temp2
        if hdu.header["EXTNAME"] == "OI_VIS":
            ins = hdu.header["INSNAME"]
            # ------------------------
            #          Vis.
            # ------------------------
            try:
                data = hdu.data["VISDATA"]
                if len(data.shape) == 1:
                    data = np.array([np.array([d]) for d in data])
                err = hdu.data["VISERR"]
                if len(err.shape) == 1:
                    err = np.array([np.array([e]) for e in err])
                temp = {}
                temp["UCOORD"] = hdu.data["UCOORD"][:, None]
                temp["VCOORD"] = hdu.data["VCOORD"][:, None]
                temp["wavel"] = wavel[ins][None, :][0]
                temp["MJD"] = hdu.data["MJD"][0]
                temp["err"] = err[0]
                temp["data"] = data
                tab_data["Vis_data; " + ins] = temp
            except Exception:
                pass

            # ------------------------
            #    Vis. amplitude
            # ------------------------
            data = hdu.data["VISAMP"]
            if len(data.shape) == 1:
                data = np.array([np.array([d]) for d in data])
            err = hdu.data["VISAMPERR"]
            if len(err.shape) == 1:
                err = np.array([np.array([e]) for e in err])
            temp = {}
            temp["UCOORD"] = hdu.data["UCOORD"][:, None] + 0 * wavel[ins][None, :]
            temp["VCOORD"] = hdu.data["VCOORD"][:, None] + 0 * wavel[ins][None, :]
            temp["wavel"] = wavel[ins][None, :][0]
            temp["MJD"] = hdu.data["MJD"][0]
            temp["err"] = err
            temp["data"] = data
            tab_data["Vis_amp; " + ins] = temp

            # ------------------------
            #       Vis. phase
            # ------------------------
            data = hdu.data["VISPHI"]
            if len(data.shape) == 1:
                data = np.array([np.array([d]) for d in data])
            err = hdu.data["VISPHIERR"]
            if len(err.shape) == 1:
                err = np.array([np.array([e]) for e in err])
            temp = {}
            temp["UCOORD"] = hdu.data["UCOORD"][:, None] + 0 * wavel[ins][None, :]
            temp["VCOORD"] = hdu.data["VCOORD"][:, None] + 0 * wavel[ins][None, :]
            temp["wavel"] = wavel[ins][None, :][0]
            temp["MJD"] = hdu.data["MJD"][0]
            temp["err"] = err
            temp["data"] = data
            tab_data["Vis_phi; " + ins] = temp

    fitsHandler.close()
    tab_data["info"]["ncp"] = ncp_master
    return tab_data


def data2obs(
    data,
    use_flag=True,
    cond_wl=False,
    wl_bounds=None,
    extra_error_v2=0,
    extra_error_cp=0,
    err_scale_cp=1,
    err_scale_v2=1,
    cond_uncer=False,
    rel_max=None,
    verbose=True,
    input_rad=False,
):
    """
    Convert and select data from the object format (load() function).

    Parameters:
    -----------

    `data` {class}:
        Object containing all the data (see `oimalib.load()`),\n
    `use_flag` {boolean}:
        If True, use flag from the original oifits file,\n
    `cond_wl` {boolean}:
        If True, apply wavelenght restriction between wl_min and wl_max,\n
    `wl_min`, `wl_max` {float}:
        if cond_wl, limits of the wavelength domain [µm],\n
    `extra_error_v2` {float}:
        Additonal error to apply on vis2 data (quadratically added),\n
    `extra_error_cp` {float}:
        Additonal error to apply on cp data (quadratically added),\n
    `err_scale_v2`, `err_scale_cp` {float}:
        Scaling factor to apply on uncertainties (multiplicative factor),\n
    `cond_uncer` {boolean}:
        If True, select the best data according their relative uncertainties (`rel_max`),\n
    `rel_max` {float}:
        if cond_uncer, maximum sigma uncertainties allowed [%],\n
    `input_rad` {bool}:
        If True, cp data are assumed in radian and so converted in degrees,\n
    `verbose`: {boolean}
        If True, display useful information about the data selection.\n

    Return:
    -------

    `Obs` {tuple}:
        Tuple containing all the selected data in an appropriate format to
        perform the fit.

    """
    nwl = len(data.wl)

    nbl = data.vis2.shape[0]
    ncp = data.cp.shape[0]

    vis2_data = data.vis2.flatten()  # * 0.97
    e_vis2_data = (
        (data.e_vis2.flatten() ** 2 + extra_error_v2 ** 2) ** 0.5
    ) * err_scale_v2
    flag_V2 = data.flag_vis2.flatten()

    if input_rad:
        cp_data = np.rad2deg(data.cp.flatten())
        e_cp_data = (
            np.rad2deg(np.sqrt(data.e_cp.flatten() ** 2 + extra_error_cp ** 2))
            * err_scale_cp
        )
    else:
        cp_data = data.cp.flatten()
        e_cp_data = (
            np.sqrt(data.e_cp.flatten() ** 2 + extra_error_cp ** 2) * err_scale_cp
        )

    flag_CP = data.flag_cp.flatten()

    if use_flag:
        pass
    else:
        flag_V2 = [False] * len(vis2_data)
        flag_CP = [False] * len(cp_data)

    u_data, v_data = [], []
    u1_data, v1_data, u2_data, v2_data = [], [], [], []

    for i in range(nbl):
        for _ in range(nwl):
            u_data.append(data.u[i])
            v_data.append(data.v[i])

    for i in range(ncp):
        for _ in range(nwl):
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
                if (wl_data[i] >= wl_bounds[0] * 1e-6) & (
                    wl_data[i] <= wl_bounds[1] * 1e-6
                ):
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
                if (wl_data_cp[i] >= wl_bounds[0] * 1e-6) & (
                    wl_data_cp[i] <= wl_bounds[1] * 1e-6
                ):
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
                % (wl_bounds[0], chr(955), wl_bounds[1])
            )
        if cond_uncer:
            print(fr"-> Restriction on uncertainties: {chr(949)} < {rel_max:2.1f} %")

    return Obs


def listfile2obs(
    tab,
    use_flag=False,
    cond_wl=False,
    wl_min=None,
    wl_max=None,
    cond_uncer=False,
    rel_max=None,
    verbose=False,
):
    """Add all oifits file in the Obs data array."""
    Obs = data2obs(
        tab[0],
        use_flag=use_flag,
        cond_wl=cond_wl,
        wl_min=wl_min,
        wl_max=wl_max,
        cond_uncer=cond_uncer,
        rel_max=rel_max,
        verbose=verbose,
    )

    for d in tab[1:]:
        o = data2obs(
            d,
            use_flag=use_flag,
            cond_wl=cond_wl,
            wl_min=wl_min,
            wl_max=wl_max,
            cond_uncer=cond_uncer,
            rel_max=rel_max,
            verbose=verbose,
        )
        Obs = np.concatenate([Obs, o])
    return Obs


def dir2data(filedir):
    """
    Format all data from different oifits files in filedir to the list usable by the other functions.
    """
    listfile = glob(filedir + "*.fits")

    tab = []
    for f in listfile:
        data = load(f, cam="SC")
        tab.append(data)
    return tab


def load(namefile, cam="SC", simu=False):
    fitsHandler = fits.open(namefile)

    # OI_TARGET table
    target = fitsHandler["OI_TARGET"].data.field("TARGET")

    # OI_WAVELENGTH table
    ins = fitsHandler["OI_WAVELENGTH"].header["INSNAME"]

    if "GRAVITY" in ins:
        index_cam = 10
        if cam == "FT":
            index_cam = 20
    else:
        index_cam = 1

    if simu:
        index_cam = None

    try:
        wave = fitsHandler["OI_WAVELENGTH", index_cam].data.field("EFF_WAVE")
    except KeyError:
        wave = np.zeros(1)

    # OI_FLUX table
    try:
        spectre = fitsHandler["OI_FLUX", index_cam].data.field("FLUXDATA")
        sta_index = fitsHandler["OI_FLUX", index_cam].data.field("STA_INDEX")
    except KeyError:
        try:
            spectre = fitsHandler["OI_FLUX", index_cam].data.field("FLUX")
            sta_index = fitsHandler["OI_FLUX", index_cam].data.field("STA_INDEX")
        except KeyError:
            spectre = np.zeros(1)
            sta_index = np.zeros(1)

    nspec = spectre.shape[0]

    # OI_ARRAY table
    index_ref = fitsHandler["OI_ARRAY"].data.field("STA_INDEX")
    teles_ref = fitsHandler["OI_ARRAY"].data.field("STA_NAME")
    array = fitsHandler["OI_ARRAY"].header["ARRNAME"]

    dic_index = _compute_dic_index(index_ref, teles_ref)

    tel = []
    for i in range(nspec):
        try:
            tel.append(dic_index[sta_index[i]])
        except KeyError:
            pass
    tel = np.array(tel)

    # OI_T3 table
    try:
        cp = fitsHandler["OI_T3", index_cam].data.field("T3PHI")
    except KeyError:
        test_cp = isinstance(fitsHandler["OI_T3", None].data.field("T3PHI"), np.ndarray)
        fitsHandler.close()
        if test_cp:
            print(
                "Your dataset seems to be a simulation (from aspro2), you should add simu=True.",
                file=sys.stderr,
            )
        else:
            print(
                "Your dataset have not OI_T3 with the supported index (%i), try another dataset."
                % index_cam,
                file=sys.stderr,
            )
        return None

    e_cp = fitsHandler["OI_T3", index_cam].data.field("T3PHIERR")
    index_cp = fitsHandler["OI_T3", index_cam].data.field("STA_INDEX")
    flag_cp = fitsHandler["OI_T3", index_cam].data.field("FLAG")
    u1 = fitsHandler["OI_T3", index_cam].data.field("U1COORD")
    u2 = fitsHandler["OI_T3", index_cam].data.field("U2COORD")
    v1 = fitsHandler["OI_T3", index_cam].data.field("V1COORD")
    v2 = fitsHandler["OI_T3", index_cam].data.field("V2COORD")
    u3 = -(u1 + u2)
    v3 = -(v1 + v2)

    # OI_VIS2 table
    vis2 = fitsHandler["OI_VIS2", index_cam].data.field("VIS2DATA")
    e_vis2 = fitsHandler["OI_VIS2", index_cam].data.field("VIS2ERR")
    index_vis2 = fitsHandler["OI_VIS2", index_cam].data.field("STA_INDEX")
    flag_vis2 = fitsHandler["OI_VIS2", index_cam].data.field("FLAG")
    u = fitsHandler["OI_VIS2", index_cam].data.field("UCOORD")
    v = fitsHandler["OI_VIS2", index_cam].data.field("VCOORD")
    B = np.sqrt(u ** 2 + v ** 2)

    # OI_VIS table
    dvis = fitsHandler["OI_VIS", index_cam].data.field("VISAMP")
    e_dvis = fitsHandler["OI_VIS", index_cam].data.field("VISAMPERR")
    dphi = fitsHandler["OI_VIS", index_cam].data.field("VISPHI")
    e_dphi = fitsHandler["OI_VIS", index_cam].data.field("VISPHIERR")
    flag_dvis = fitsHandler["OI_VIS", index_cam].data.field("FLAG")

    try:
        mjd = fitsHandler[0].header["MJD-OBS"]
    except KeyError:
        mjd = np.nan
    try:
        dat = fitsHandler[0].header["DATE-OBS"]
    except KeyError:
        dat = np.nan

    fitsHandler.close()

    info = {
        "Ins": ins,
        "Index": index_ref,
        "Config": teles_ref,
        "Target": target,
        "Bmax": B.max(),
        "Array": array,
        "nbl": len(u),
        "ncp": len(u1),
        "mjd": mjd,
        "Date": dat,
        "filename": namefile,
    }

    # Compute freq, blname
    freq_cp, freq_vis2, bl_cp = [], [], []

    for i in range(len(u1)):
        B1 = np.sqrt(u1[i] ** 2 + v1[i] ** 2)
        B2 = np.sqrt(u2[i] ** 2 + v2[i] ** 2)
        B3 = np.sqrt(u3[i] ** 2 + v3[i] ** 2)

        Bmax = np.max([B1, B2, B3])
        bl_cp.append(Bmax)
        freq_cp.append(Bmax / wave / 206264.806247)  # convert to arcsec-1

    for i in range(len(u)):
        freq_vis2.append(B[i] / wave / 206264.806247)  # convert to arcsec-1

    freq_cp = np.array(freq_cp)
    freq_vis2 = np.array(freq_vis2)
    bl_cp = np.array(bl_cp)

    blname = _compute_bl_name(index_vis2, index_ref, teles_ref)
    cpname = _compute_cp_name(index_cp, index_ref, teles_ref)

    dic_output = {
        "flux": spectre,
        "vis2": vis2,
        "e_vis2": e_vis2,
        "cp": cp,
        "e_cp": e_cp,
        "dvis": dvis,
        "e_dvis": e_dvis,
        "dphi": dphi,
        "e_dphi": e_dphi,
        "wl": wave,
        "u": u,
        "v": v,
        "u1": u1,
        "u2": u2,
        "u3": u3,
        "v1": v1,
        "v2": v2,
        "v3": v3,
        "cpname": cpname,
        "teles_ref": teles_ref,
        "index_ref": index_ref,
        "blname": blname,
        "bl": B,
        "bl_cp": bl_cp,
        "index": index_vis2,
        "index_cp": index_cp,
        "freq_cp": freq_cp,
        "freq_vis2": freq_vis2,
        "tel": tel,
        "flag_vis2": flag_vis2,
        "flag_cp": flag_cp,
        "flag_dvis": flag_dvis,
        "info": info,
    }

    output = dict2class(dic_output)
    return output
