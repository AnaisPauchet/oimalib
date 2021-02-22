# -*- coding: utf-8 -*-
"""
@author: Anthony Soulain (University of Sydney)
-----------------------------------------------------------------
OIMALIB: Optical Interferometry Modelisation and Analysis Library
-----------------------------------------------------------------

OIFITS related function.
-----------------------------------------------------------------
"""

from glob import glob

import numpy as np
from astropy.io import fits
from matplotlib import pyplot as plt
from munch import munchify as dict2class
from termcolor import cprint

plt.close('all')


def oifits2dic(filename, rad=False):
    """ 
    Read an OiFits file and store observables (CP, V2, Vis, informations, etc,) 
    in a dictionary format with keys corresponding to the standard oifits format.
    """
    fitsHandler = fits.open(filename)

    hdrr = fitsHandler[0].header

    for hdu in fitsHandler[1:]:
        if hdu.header['EXTNAME'] in ['OI_T3', 'OI_VIS2']:
            ins = hdu.header['INSNAME']

    # -----------------------------------
    #            initOiData
    # -----------------------------------
    wavel = {}  # wave tables for each instrumental setup
    telArray = {}
    wlOffset = 0.0

    # -- load Wavelength and Array: ----------------------------------------------
    for hdu in fitsHandler[1:]:
        if hdu.header['EXTNAME'] == 'OI_WAVELENGTH':
            if 'PIONIER' in hdu.header['INSNAME']:
                ins = 'PIONIER'
                wavel['PIONIER'] = wlOffset + hdu.data['EFF_WAVE']*1e6  # in um
            else:
                ins = hdu.header['INSNAME']
                if 'GRAVITY' in ins:
                    if ('FT' not in ins) and ('SC' not in ins):
                        ins = 'GRAVITY_simu'
                wavel[ins] = wlOffset + \
                    hdu.data['EFF_WAVE']*1e6  # in um
        if hdu.header['EXTNAME'] == 'OI_ARRAY':
            name = hdu.header['ARRNAME']
            diam = hdu.data['DIAMETER'].mean()
            config = hdu.data['STA_NAME']
            index = hdu.data['STA_INDEX']
            if diam == 0:
                if 'VLTI' in name:
                    if 'AT' in hdu.data['TEL_NAME'][0]:
                        diam = 1.8
                    if 'UT' in hdu.data['TEL_NAME'][0]:
                        diam = 8
            telArray[name] = diam
        if hdu.header['EXTNAME'] == 'OI_TARGET':
            target = hdu.data['TARGET']

    tab_data = {}
    for hdu in fitsHandler[1:]:
        if hdu.header['EXTNAME'] == 'OI_T3':
            ins = hdu.header['INSNAME']
            if 'GRAVITY' in ins:
                if ('FT' not in ins) and ('SC' not in ins):
                    ins = 'GRAVITY_simu'

            # ----------------------------
            #       Closure phase
            # ----------------------------
            if rad:
                data = np.rad2deg(hdu.data['T3PHI'])
            else:
                data = hdu.data['T3PHI']
            # print data, filename
            if len(data.shape) == 1:
                data = np.array([np.array([d]) for d in data])

            if rad:
                err = np.rad2deg(hdu.data['T3PHIERR'])
            else:
                err = hdu.data['T3PHIERR']

            if len(err.shape) == 1:
                err = np.array([np.array([e]) for e in err])
            if np.sum(np.isnan(data)) < data.size:
                temp = {}
                temp['U1COORD'] = hdu.data['U1COORD'][:, None] + \
                    0*wavel[ins][None, :]
                temp['V1COORD'] = hdu.data['V1COORD'][:, None] + \
                    0*wavel[ins][None, :]
                temp['U2COORD'] = hdu.data['U2COORD'][:, None] + \
                    0*wavel[ins][None, :]
                temp['V2COORD'] = hdu.data['V2COORD'][:, None] + \
                    0*wavel[ins][None, :]
                temp['STA_INDEX'] = hdu.data['STA_INDEX']
                temp['wavel'] = wavel[ins][None, :][0]
                temp['MJD'] = hdu.data['MJD'][0]
                temp['data'] = data
                temp['flag'] = hdu.data['FLAG']
                temp['err'] = err
                if 'PIONIER' in ins:
                    ins = 'PIONIER'
                tab_data['cp_phi; '+ins] = temp
            else:
                print(' | WARNING: no valid T3PHI values in this HDU')
            # ----------------------------
            #      Closure amplitude
            # ----------------------------
            data = hdu.data['T3AMP']
            if len(data.shape) == 1:
                data = np.array([np.array([d]) for d in data])
            err = hdu.data['T3AMPERR']
            if len(err.shape) == 1:
                err = np.array([np.array([e]) for e in err])
            if np.sum(np.isnan(data)) < data.size:
                temp = {}
                temp['U1COORD'] = hdu.data['U1COORD'][:, None] + \
                    0*wavel[ins][None, :]
                temp['V1COORD'] = hdu.data['V1COORD'][:, None] + \
                    0*wavel[ins][None, :]
                temp['U2COORD'] = hdu.data['U2COORD'][:, None] + \
                    0*wavel[ins][None, :]
                temp['V2COORD'] = hdu.data['V2COORD'][:, None] + \
                    0*wavel[ins][None, :]
                temp['wavel'] = wavel[ins][None, :][0]
                temp['MJD'] = hdu.data['MJD'][0]
                temp['err'] = err
                temp['data'] = data
                tab_data['cp_amp; '+ins] = temp
            else:
                print(' | WARNING: no valid T3AMP values in this HDU')
        if hdu.header['EXTNAME'] == 'OI_VIS2':
            ins = hdu.header['INSNAME']
            if 'GRAVITY' in ins:
                if ('FT' not in ins) and ('SC' not in ins):
                    ins = 'GRAVITY_simu'
            # ----------------------------
            #      Squared Vis. (V2)
            # ----------------------------
            data = hdu.data['VIS2DATA']
            if len(data.shape) == 1:
                data = np.array([np.array([d]) for d in data])
            err = hdu.data['VIS2ERR']
            if len(err.shape) == 1:
                err = np.array([np.array([e]) for e in err])
            temp = {}
            temp['UCOORD'] = hdu.data['UCOORD'][:, None]+0*wavel[ins][None, :]
            temp['VCOORD'] = hdu.data['VCOORD'][:, None]+0*wavel[ins][None, :]
            temp['STA_INDEX'] = hdu.data['STA_INDEX']
            temp['wavel'] = wavel[ins][None, :][0]
            temp['MJD'] = hdu.data['MJD'][0]
            temp['err'] = err
            temp['data'] = data
            temp['flag'] = hdu.data['FLAG']
            tab_data['Vis2; '+ins] = temp

            l_B = []
            for i in range(np.shape(temp['UCOORD'])[0]):
                U = temp['UCOORD'][i]
                V = temp['VCOORD'][i]
                l_B.append(np.sqrt(U**2+V**2))

            if ins == 'VEGA':
                b_name = []
                for x in np.unique(hdu.data['STA_INDEX']):
                    b_name.append(config[x])

                config = b_name

            # -------- INFO ---------
            Bmax = (hdu.data['UCOORD']**2+hdu.data['VCOORD']**2).max()
            Bmax = np.sqrt(Bmax)
            temp2 = {}
            temp2['Bmax'] = Bmax
            temp2['Date'] = Bmax
            temp2['hdr'] = hdrr
            if 'PIONIER' in ins:
                temp2['Ins'] = 'PIONIER'
            elif 'GRAVITY' in ins:
                if ('FT' not in ins) and ('SC' not in ins):
                    temp2['Ins'] = 'GRAVITY_simu'
                else:
                    temp2['Ins'] = ins
            else:
                temp2['Ins'] = ins
            temp2['Index'] = index
            temp2['Config'] = config
            temp2['Target'] = target
            temp2['L_base'] = l_B
            try:
                temp2['Date'] = fitsHandler[0].header['MJD-OBS']
            except KeyError:
                temp2['Date'] = None
                pass
            tab_data['info'] = temp2

        if hdu.header['EXTNAME'] == 'OI_VIS':
            ins = hdu.header['INSNAME']
            if 'GRAVITY' in ins:
                if ('FT' not in ins) and ('SC' not in ins):
                    ins = 'GRAVITY_simu'
            # ------------------------
            #          Vis.
            # ------------------------
            try:
                data = hdu.data['VISDATA']
                if len(data.shape) == 1:
                    data = np.array([np.array([d]) for d in data])
                err = hdu.data['VISERR']
                if len(err.shape) == 1:
                    err = np.array([np.array([e]) for e in err])
                temp = {}
                temp['UCOORD'] = hdu.data['UCOORD'][:, None]
                temp['VCOORD'] = hdu.data['VCOORD'][:, None]
                temp['wavel'] = wavel[ins][None, :][0]
                temp['MJD'] = hdu.data['MJD'][0]
                temp['err'] = err[0]
                temp['data'] = data
                tab_data['Vis_data; '+ins] = temp
            except Exception:
                pass

            # ------------------------
            #    Vis. amplitude
            # ------------------------
            data = hdu.data['VISAMP']
            if 'GRAVITY' in ins:
                if ('FT' not in ins) and ('SC' not in ins):
                    ins = 'GRAVITY_simu'
            if len(data.shape) == 1:
                data = np.array([np.array([d]) for d in data])
            err = hdu.data['VISAMPERR']
            if len(err.shape) == 1:
                err = np.array([np.array([e]) for e in err])
            temp = {}
            temp['UCOORD'] = hdu.data['UCOORD'][:, None]+0*wavel[ins][None, :]
            temp['VCOORD'] = hdu.data['VCOORD'][:, None]+0*wavel[ins][None, :]
            temp['wavel'] = wavel[ins][None, :][0]
            temp['MJD'] = hdu.data['MJD'][0]
            temp['err'] = err
            temp['data'] = data
            tab_data['Vis_amp; '+ins] = temp

            # ------------------------
            #       Vis. phase
            # ------------------------
            data = hdu.data['VISPHI']*np.pi/180
            if len(data.shape) == 1:
                data = np.array([np.array([d]) for d in data])
            err = hdu.data['VISPHIERR']*np.pi/180
            if len(err.shape) == 1:
                err = np.array([np.array([e]) for e in err])
            temp = {}
            temp['UCOORD'] = hdu.data['UCOORD'][:, None]+0*wavel[ins][None, :]
            temp['VCOORD'] = hdu.data['VCOORD'][:, None]+0*wavel[ins][None, :]
            temp['wavel'] = wavel[ins][None, :][0]
            temp['MJD'] = hdu.data['MJD'][0]
            temp['err'] = err
            temp['data'] = data
            tab_data['Vis_phi; '+ins] = temp

    return tab_data


def load(namefile, target=None, cam=None, AverPolar=True, pol=None,
         simu=False, verbose=False, rad=False):
    """
    Converts the oifits data to a usable object format (data).

    Parameters:
    -----------

    `namefile` {str}:
        Name of the oifits file,\n
    `target` {str}:
        Name of the target if not in the header,\n
    `cam` {int}:
        Only for GRAVITY data: corresponds to the used camera ('SC': science camera
        or 'FT': fringe tracker camera). Default = None,\n
    `AverPolar` {bool}:
        Only for GRAVITY data. AverPolar = True: the two polarisation are averaged 
        during the process,\n
    `pol` {int}:
        Only for GRAVITY data. Selected polarisation used during the process
        (1 or 2),\n
    `simu` {bool}:
        If True, the oifits come from ASPRO2.

    Return:
    -------

    `data` {obj}:
        Easy object format stored with all the interferometrical observables 
        (u=data.u, cp=data.cp, V2=data.vis2, etc.)\n
    """

    data = oifits2dic(namefile, rad=rad)

    hdr = data['info']['hdr']
    try:
        date = hdr['DATE-OBS']
    except KeyError:
        date = ''

    try:
        obj = hdr['OBJECT']
    except KeyError:
        obj = target

    ins = data['info']['Ins']
    index_ref = data['info']['Index']
    teles_ref = data['info']['Config']

    try:
        # Extract usable information of the OIfits file.
        seeing = hdr['HIERARCH ESO ISS AMBI FWHM START']
        tau0 = hdr['HIERARCH ESO ISS AMBI TAU0 START']
    except KeyError:
        print('Warning: header keyword format is not ESO standard.')

    try:
        relFT = hdr['HIERARCH ESO QC TRACKING_RATIO']
    except KeyError:
        relFT = np.nan

    if 'GRAVITY' in ins:
        ins = ins.split('_')[0]
        if (cam is None) and not (simu):
            cprint('-'*38, 'red')
            cprint('GRAVITY data file: please chose a camera (SC or FT)', 'red')
            cprint('-'*38, 'red')
            return None

    if verbose:
        if len(teles_ref) == 4:
            print('\n%s: %s (%s-%s-%s-%s), %s' % (ins, obj,
                                                  teles_ref[0], teles_ref[1],
                                                  teles_ref[2], teles_ref[3], date))
        elif len(teles_ref) == 3:
            print('\n%s: %s (%s-%s-%s), %s' %
                  (ins, obj, teles_ref[0], teles_ref[1], teles_ref[2], date))
        print('-'*50)

    if verbose:
        try:
            if ins == 'GRAVITY':
                print('Seeing = %2.2f, tau0 = %2.1f ms, FT = %2.1f %%' %
                      (seeing, tau0*1000., relFT))
            else:
                print('Seeing = %2.2f, tau0 = %2.1f ms' % (seeing, tau0*1000.))
        except Exception:
            pass

    dic_ind = {}
    for i in range(len(index_ref)):
        dic_ind[index_ref[i]] = teles_ref[i]

    if ins == 'GRAVITY':
        if simu:
            vis2 = data['Vis2; %s_simu' % ins]['data']
            e_vis2 = data['Vis2; %s_simu' % ins]['err']
            cp = data['cp_phi; %s_simu' % ins]['data']
            e_cp = data['cp_phi; %s_simu' % ins]['err']

            flag_vis2 = data['Vis2; %s_simu' % ins]['flag']
            flag_cp = data['cp_phi; %s_simu' % ins]['flag']

            wl = data['Vis2; %s_simu' % (ins)]['wavel'] * 1e-6
            u1 = data['cp_phi; %s_simu' % ins]['U1COORD'][:, 0]
            u2 = data['cp_phi; %s_simu' % ins]['U2COORD'][:, 0]
            v1 = data['cp_phi; %s_simu' % ins]['V1COORD'][:, 0]
            v2 = data['cp_phi; %s_simu' % ins]['V2COORD'][:, 0]
            u3 = (u1+u2)
            v3 = (v1+v2)

            u = data['Vis2; %s_simu' % ins]['UCOORD'][:, 0]
            v = data['Vis2; %s_simu' % ins]['VCOORD'][:, 0]

            B = np.sqrt(u**2 + v**2)

            index = data['Vis2; %s_simu' % (ins)]['STA_INDEX']
            index_cp = data['cp_phi; %s_simu' % (ins)]['STA_INDEX']
        else:
            if AverPolar:
                # V2
                # ---------------------
                vis2_1 = data['Vis2; GRAVITY_%s_P%s' % (cam, 1)]['data']
                vis2_2 = data['Vis2; GRAVITY_%s_P%s' % (cam, 2)]['data']

                vis2 = (vis2_1 + vis2_2)/2.  # Average polaristion of GRAVITY

                e_vis2_1 = data['Vis2; GRAVITY_%s_P%s' % (cam, 1)]['err']
                e_vis2_2 = data['Vis2; GRAVITY_%s_P%s' % (cam, 2)]['err']

                e_vis2 = (e_vis2_1**2 + e_vis2_2**2)**0.5

                cp_1 = data['cp_phi; GRAVITY_%s_P%s' % (cam, 1)]['data']
                cp_2 = data['cp_phi; GRAVITY_%s_P%s' % (cam, 2)]['data']

                cp = (cp_1 + cp_2)/2.

                e_cp_1 = data['cp_phi; GRAVITY_%s_P%s' % (cam, 1)]['err']
                e_cp_2 = data['cp_phi; GRAVITY_%s_P%s' % (cam, 2)]['err']

                e_cp = (e_cp_1**2 + e_cp_2**2)**0.5

                u1 = data['cp_phi; GRAVITY_%s_P%s' % (cam, 1)]['U1COORD'][:, 0]
                u2 = data['cp_phi; GRAVITY_%s_P%s' % (cam, 1)]['U2COORD'][:, 0]
                v1 = data['cp_phi; GRAVITY_%s_P%s' % (cam, 1)]['V1COORD'][:, 0]
                v2 = data['cp_phi; GRAVITY_%s_P%s' % (cam, 1)]['V2COORD'][:, 0]
                u3 = (u1+u2)
                v3 = (v1+v2)

                u = data['Vis2; GRAVITY_%s_P%s' % (cam, 1)]['UCOORD'][:, 0]
                v = data['Vis2; GRAVITY_%s_P%s' % (cam, 1)]['VCOORD'][:, 0]

                flag_vis2 = data['Vis2; GRAVITY_%s_P%s' % (cam, 1)]['flag']
                flag_cp = data['cp_phi; GRAVITY_%s_P%s' % (cam, 1)]['flag']

            else:
                vis2 = data['Vis2; GRAVITY_%s_P%s' % (cam, pol)]['data']
                e_vis2 = data['Vis2; GRAVITY_%s_P%s' % (cam, pol)]['err']
                cp = data['cp_phi; GRAVITY_%s_P%s' % (cam, pol)]['data']
                e_cp = data['cp_phi; GRAVITY_%s_P%s' % (cam, pol)]['err']

                u1 = data['cp_phi; GRAVITY_%s_P%s' %
                          (cam, pol)]['U1COORD'][:, 0]
                u2 = data['cp_phi; GRAVITY_%s_P%s' %
                          (cam, pol)]['U2COORD'][:, 0]
                v1 = data['cp_phi; GRAVITY_%s_P%s' %
                          (cam, pol)]['V1COORD'][:, 0]
                v2 = data['cp_phi; GRAVITY_%s_P%s' %
                          (cam, pol)]['V2COORD'][:, 0]
                u3 = (u1+u2)
                v3 = (v1+v2)

                u = data['Vis2; GRAVITY_%s_P%s' % (cam, pol)]['UCOORD'][:, 0]
                v = data['Vis2; GRAVITY_%s_P%s' % (cam, pol)]['VCOORD'][:, 0]

                flag_vis2 = data['Vis2; GRAVITY_%s_P%s' % (cam, pol)]['flag']
                flag_cp = data['cp_phi; GRAVITY_%s_P%s' % (cam, pol)]['flag']

            wl = data['Vis2; GRAVITY_%s_P%s' % (cam, 1)]['wavel'] * 1e-6
            index = data['Vis2; GRAVITY_%s_P%s' % (cam, 1)]['STA_INDEX']
            index_cp = data['cp_phi; GRAVITY_%s_P%s' % (cam, 1)]['STA_INDEX']
            B = np.sqrt(u**2 + v**2)

    else:
        vis2 = data['Vis2; %s' % ins]['data']
        e_vis2 = data['Vis2; %s' % ins]['err']
        cp = data['cp_phi; %s' % ins]['data']
        e_cp = data['cp_phi; %s' % ins]['err']

        flag_vis2 = data['Vis2; %s' % ins]['flag']
        flag_cp = data['cp_phi; %s' % ins]['flag']

        wl = data['Vis2; %s' % (ins)]['wavel'] * 1e-6
        u1 = data['cp_phi; %s' % ins]['U1COORD'][:, 0]
        u2 = data['cp_phi; %s' % ins]['U2COORD'][:, 0]
        v1 = data['cp_phi; %s' % ins]['V1COORD'][:, 0]
        v2 = data['cp_phi; %s' % ins]['V2COORD'][:, 0]
        u3 = (u1+u2)
        v3 = (v1+v2)

        u = data['Vis2; %s' % ins]['UCOORD'][:, 0]
        v = data['Vis2; %s' % ins]['VCOORD'][:, 0]

        B = np.sqrt(u**2 + v**2)

        index = data['Vis2; %s' % (ins)]['STA_INDEX']
        index_cp = data['cp_phi; %s' % (ins)]['STA_INDEX']

    freq_cp, freq_vis2 = [], []

    for i in range(len(u1)):
        B1 = np.sqrt(u1[i]**2+v1[i]**2)
        B2 = np.sqrt(u2[i]**2+v2[i]**2)
        B3 = np.sqrt(u3[i]**2+v3[i]**2)

        Bmax = np.max([B1, B2, B3])
        freq_cp.append(Bmax/wl/206264.806247)  # convert to arcsec-1

    for i in range(len(u)):
        freq_vis2.append(B[i]/wl/206264.806247)  # convert to arcsec-1

    freq_cp = np.array(freq_cp)
    freq_vis2 = np.array(freq_vis2)

    dic_output = {'vis2': vis2, 'e_vis2': e_vis2,
                  'cp': cp, 'e_cp': e_cp,
                  'wl': wl, 'u': u, 'v': v,
                  'u1': u1, 'u2': u2, 'u3': u3,
                  'v1': v1, 'v2': v2, 'v3': v3,
                  'teles_ref': teles_ref, 'index_ref': index_ref,
                  'B': B, 'index': index, 'index_cp': index_cp,
                  'freq_cp': freq_cp, 'freq_vis2': freq_vis2,
                  'flag_vis2': flag_vis2, 'flag_cp': flag_cp,
                  }

    data = dict2class(dic_output)
    return data


def data2obs(data, use_flag=True, cond_wl=False, wl_min=None, wl_max=None,
             extra_error_v2=0, extra_error_cp=0, err_scale=1, cond_uncer=False,
             rel_max=None, verbose=True, input_rad=False):
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
    `err_scale` {float}:
        Scaling factor to apply on all vis2 data (multiplicative factor),\n
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
    e_vis2_data = ((data.e_vis2.flatten()**2 + extra_error_v2**2)**0.5) * err_scale
    flag_V2 = data.flag_vis2.flatten()

    if input_rad:
        cp_data = np.rad2deg(data.cp.flatten())
        e_cp_data = np.rad2deg(
            np.sqrt(data.e_cp.flatten()**2+extra_error_cp**2))
    else:
        cp_data = data.cp.flatten()
        e_cp_data = np.sqrt(data.e_cp.flatten()**2 +
                            extra_error_cp**2)

    flag_CP = data.flag_cp.flatten()

    if use_flag:
        pass
    else:
        flag_V2 = [False]*len(vis2_data)
        flag_CP = [False]*len(cp_data)

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

    wl_data = np.array(list(data.wl)*nbl)
    wl_data_cp = np.array(list(data.wl)*ncp)

    obs = []

    for i in range(nbl*nwl):
        if flag_V2[i] & use_flag:
            pass
        else:
            if not cond_wl:
                tmp = [u_data[i], v_data[i], wl_data[i]]
                typ = 'V2'
                obser = vis2_data[i]
                err = e_vis2_data[i]
                if cond_uncer:
                    if (err/obser <= rel_max*1e-2):
                        obs.append([tmp, typ, obser, err])
                    else:
                        pass
                else:
                    obs.append([tmp, typ, obser, err])

            else:
                if (wl_data[i] >= wl_min*1e-6) & (wl_data[i] <= wl_max*1e-6):
                    tmp = [u_data[i], v_data[i], wl_data[i]]
                    typ = 'V2'
                    obser = vis2_data[i]
                    err = e_vis2_data[i]
                    if cond_uncer:
                        if (err/obser <= rel_max*1e-2):
                            obs.append([tmp, typ, obser, err])
                        else:
                            pass
                    else:
                        obs.append([tmp, typ, obser, err])
                else:
                    pass
    N_v2_rest = len(obs)

    for i in range(ncp*nwl):
        if flag_CP[i]:
            pass
        else:
            if not cond_wl:
                tmp = [u1_data[i], u2_data[i], (u1_data[i]+u2_data[i]), v1_data[i], v2_data[i],
                       (v1_data[i]+v2_data[i]), wl_data_cp[i]]
                typ = 'CP'
                obser = cp_data[i]
                err = e_cp_data[i]
                if cond_uncer:
                    if (err/obser <= rel_max*1e-2):
                        obs.append([tmp, typ, obser, err])
                    else:
                        pass
                else:
                    obs.append([tmp, typ, obser, err])
            else:
                if (wl_data_cp[i] >= wl_min*1e-6) & (wl_data_cp[i] <= wl_max*1e-6):
                    tmp = [u1_data[i], u2_data[i], (u1_data[i]+u2_data[i]), v1_data[i], v2_data[i],
                           (v1_data[i]+v2_data[i]), wl_data_cp[i]]
                    typ = 'CP'
                    obser = cp_data[i]
                    err = e_cp_data[i]
                    if cond_uncer:
                        if (err/obser <= rel_max*1e-2):
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
        print('\nTotal # of data points: %i (%i V2, %i CP)' %
              (len(Obs), N_v2_rest, N_cp_rest))
        if use_flag:
            print('-> Flag in oifits files used.')
        if cond_wl:
            print(r'-> Restriction on wavelenght: %2.2f < %s < %2.2f µm' %
                  (wl_min, chr(955), wl_max))
        if cond_uncer:
            print(r'-> Restriction on uncertainties: %s < %2.1f %%' %
                  (chr(949), rel_max))

    return Obs


def listfile2obs(tab, use_flag=False, cond_wl=False, wl_min=None, wl_max=None,
                 cond_uncer=False, rel_max=None, verbose=False):
    """ Add all oifits file in the Obs data array. """
    Obs = data2obs(tab[0], use_flag=use_flag, cond_wl=cond_wl, wl_min=wl_min, wl_max=wl_max,
                   cond_uncer=cond_uncer, rel_max=rel_max, verbose=verbose)

    for d in tab[1:]:
        o = data2obs(d, use_flag=use_flag, cond_wl=cond_wl, wl_min=wl_min, wl_max=wl_max,
                     cond_uncer=cond_uncer, rel_max=rel_max, verbose=verbose)
        Obs = np.concatenate([Obs, o])
    return Obs


def dir2data(filedir):
    """
    Format all data from different oifits files in filedir to the list usable by the other functions.
    """
    listfile = glob(filedir + '*.fits')

    tab = []
    for f in listfile:
        data = load(f, cam='SC')
        tab.append(data)
    return tab
