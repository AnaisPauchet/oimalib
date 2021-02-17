#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 10 15:30:57 2019

@author: asoulain

Different useful function to load, display and analyze interferometrical data.

Example of use to fit data: 
---------------------------

datadir = "Path of the folder containing dataset"

listfile = glob(datadir + '*.fits')

data = OiFile2Class(file, cam='SC') # Use a single file of data.
#/ or / 
data = dir2data(datadir) # Use a all files in the folder datadir

use_flag = False # Don't use flagged data
cond_wl = False # Fit only a spectral range
cond_uncer = False # Fit only a good snr data
wl_min, wl_max, rel_max = 2.1, 2.3, 3 #(band inf, band sup, snr min)

plotUV([data1, data2], use_flag=use_flag, cond_wl=cond_wl, wl_min=wl_min, wl_max=wl_max, cond_uncer=cond_uncer, rel_max=rel_max)

Obs1 = OiClass2Obs(data1, use_flag=use_flag, cond_wl=cond_wl, wl_min=wl_min, wl_max=wl_max, cond_uncer=cond_uncer, rel_max=rel_max)
Obs2 = OiClass2Obs(data2, use_flag=use_flag, cond_wl=cond_wl, wl_min=wl_min, wl_max=wl_max, cond_uncer=cond_uncer, rel_max=rel_max)
Obs = np.concatenate([Obs1, Obs2])

N_CP = len(Obs[(Obs[:,1] == 'CP')])
N_V2 = len(Obs[(Obs[:,1] == 'V2')])
N = len(Obs)

paramM = {'diam' : 1,
            'model' : 'disk',
            'x0' : 0,
            'y0' : 0}
#
fit = Smartfit(Obs, paramM, fitOnly=['diam'], multiproc=False)
#
plot_oidata([data1], model=True, param=fit['best'],
            use_flag=use_flag, fit=fit,
            cond_wl=cond_wl, wl_min=wl_min, wl_max=wl_max,
            cond_uncer=cond_uncer, rel_max=rel_max)

plt.show()

"""

from glob import glob

import numpy as np
from astropy.io import fits
from matplotlib import pyplot as plt
# from astools.all import AllMyFields, rad2mas
from munch import munchify as dict2class
from termcolor import cprint

from .fitting import comput_CP, comput_V2, select_model

plt.close('all')

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
             'A0-J2': '#f38181',  # violet
             'A0-J3': '#1f5f8b',
             'G1-J2': '#a393eb',
             'G1-J3': '#eedf6b',
             'J2-J3': 'c',
             'J2-K0': 'c',
             'A0-K0': '#8d90a1',
             'G1-K0': '#dcbc0e',
             }


def load(filename, rad=False):
    """ Read an OiFits file and store observables (CP, V2, Vis, informations, etc,) 
        in a dictionary format.
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


def OiFile2Class(namefile, target=None, cam=None, AverPolar=True, pol=None,
                 simu=False, verbose=False, rad=False):
    """
    Converts the oifits data to a usable object format (data).

    Parameters:
    -----------

    namefile: {str}
        Name of the oifits file.\n
    cam: {int}
        Only for GRAVITY data: corresponds to the used camera ('SC': science camera
        or 'FT': fringe tracker camera). Default = None.\n
    AverPolar: {bool}
        Only for GRAVITY data. AverPolar = True: the two polarisation are averaged 
        during the process, else pol need to be given (1 or 2).\n
    pol: {bool}
        Only for GRAVITY data. Selected polarisation used during the process.\n

    Return:
    -------

    data: {obj}
        Easy object format stored with all the interferometrical observables 
        (u=data.u, cp=data.cp, V2=data.vis2, etc.)\n
    """

    data = load(namefile, rad=rad)

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

    dic_output = {'vis2': vis2,
                  'e_vis2': e_vis2,
                  'cp': cp,
                  'e_cp': e_cp,
                  'wl': wl,
                  'u': u,
                  'v': v,
                  'u1': u1,
                  'u2': u2,
                  'u3': u3,
                  'v1': v1,
                  'v2': v2,
                  'v3': v3,
                  'teles_ref': teles_ref,
                  'index_ref': index_ref,
                  'B': B,
                  'index': index,
                  'index_cp': index_cp,
                  'freq_cp': freq_cp,
                  'freq_vis2': freq_vis2,
                  'flag_vis2': flag_vis2,
                  'flag_cp': flag_cp,
                  # 'hdr': data['info']
                  }

    data = dict2class(dic_output)
    return data


def OiClass2Obs(data, use_flag=True, cond_wl=False, wl_min=None, wl_max=None,
                extra_error_v2=0, extra_error_cp=0, err_scale=1, cond_uncer=False,
                rel_max=None, verbose=True, input_rad=False):
    """
    Convert and select data from the object format (OiFile2Class function).

    Parameters:
    -----------

    data: {obj}
        Object containing all the data.
    use_flag: {boolean}
        If True, use flag from the original oifits file.
    cond_wl: {boolean}
        If True, apply wavelenght restriction between wl_min and wl_max.
    wl_min, wl_max: {float}
        if cond_wl, limits of the wavelength domain [µm]
    cond_uncer: {boolean}
        If True, select the best data according their relative uncertainties (rel_max).
    rel_max: {float}
        if cond_uncer, maximum sigma uncertainties allowed [%].
    verbose: {boolean}
        If True, display useful information about the data selection.


    Return:
    -------

    Obs: {tuple}
        Tuple containing all the selected data in an appropriate format to perform the fit.

    """
    nwl = len(data.wl)

    nbl = data.vis2.shape[0]
    ncp = data.cp.shape[0]

    vis2_data = data.vis2.flatten()  # * 0.97
    e_vis2_data = (data.e_vis2.flatten()**2 + extra_error_v2**2)**0.5
    flag_V2 = data.flag_vis2.flatten()

    if input_rad:
        cp_data = np.rad2deg(data.cp.flatten())
        e_cp_data = np.rad2deg(
            np.sqrt(data.e_cp.flatten()**2+extra_error_cp**2) * err_scale)
    else:
        cp_data = data.cp.flatten()
        e_cp_data = np.sqrt(data.e_cp.flatten()**2 +
                            extra_error_cp**2) * err_scale

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


def AllFile2Obs(tab, use_flag=False, cond_wl=False, wl_min=None, wl_max=None,
                cond_uncer=False, rel_max=None, verbose=False):
    """ Add all oifits file in the Obs data array. """
    Obs = OiClass2Obs(tab[0], use_flag=use_flag, cond_wl=cond_wl, wl_min=wl_min, wl_max=wl_max,
                      cond_uncer=cond_uncer, rel_max=rel_max, verbose=verbose)

    for d in tab[1:]:
        o = OiClass2Obs(d, use_flag=use_flag, cond_wl=cond_wl, wl_min=wl_min, wl_max=wl_max,
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
        data = OiFile2Class(f, cam='SC')
        tab.append(data)

    return tab


def Index2Tel(tab):
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


def Index2TelBL(data):
    """
    Make the match between index, telescope stations and color references.
    """
    dic_index = {}

    nbl = len(data.index)
    nbs = len(data.index_cp)
    for i in range(len(data.index_ref)):
        ind = data.index_ref[i]
        tel = data.teles_ref[i]
        if ind not in dic_index.keys():
            dic_index[ind] = tel

    l_base = []
    for i in range(nbl):
        base = '%s-%s' % (dic_index[data.index[i][0]],
                          dic_index[data.index[i][1]])
        l_base.append(base)
    l_tri = []
    for i in range(nbs):
        tri = '%s-%s-%s' % (dic_index[data.index_cp[i][0]],
                            dic_index[data.index_cp[i][1]],
                            dic_index[data.index_cp[i][2]],
                            )
        l_tri.append(tri)

    return l_base, l_tri


def plot_oidata(tab, use_flag=True, cmax=200, v2min=0, v2max=1.2, model=False, param=None, fit=None,
                cond_uncer=False, rel_max=None, cond_wl=False, wl_min=None, wl_max=None, log=False,
                extra_error_v2=0):
    """
    Plot the interferometric data (and the model if required), splitted in V2 and CP and restreined if different way.

    Parameters:
    -----------

    `tab`: {list}
        list containing of data from OiFile2Class function (size corresponding to the number of files),\n
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

    # ntel = len(data.teles_ref)
    # nbl = int((ntel * (ntel - 1))/2)
    # ncp = int((ntel * (ntel - 1) * (ntel - 2))/6)

    dic_ind, l_base = Index2Tel(tab)

    l_fmin, l_fmax = [], []

    for data in tab:
        tfmax = data.freq_vis2.flatten().max()
        tfmin = data.freq_vis2.flatten().min()
        l_fmax.append(tfmax)
        l_fmin.append(tfmin)

    fmin = np.array(l_fmin).min()
    fmax = np.array(l_fmax).max()

    if model:
        # Models (optionnal)
        model_target = select_model(param['model'])
        # try:
        if fit is not None:
            chi2 = fit['chi2']
            label = 'Model ($\chi^2_{red}$ = %2.1f)' % chi2
        else:
            label = 'Model'

        # except Exception:
        #     cprint('-'*38, 'cyan')
        #     cprint('Warnings: plotted model is not a fit (chi2 = nan)', 'cyan')
        #     cprint('-'*38, 'cyan')
        #     label = 'Model'
        #     chi2 = np.nan

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
            if not (base or base2) in list_bl:
                bl1 = base
                list_bl.append(base)
                list_bl.append(base2)
            else:
                bl1 = ''

            e_vis2 = np.sqrt(e_vis2**2 + extra_error_v2**2)

            ms_data = 7
            ms_model = 5
            try:
                ax1.errorbar(freq_vis2, vis2, yerr=e_vis2, marker='.', linestyle='None', color=dic_color[base],
                             ecolor='lightgray', label=bl1, elinewidth=0.5, ms=ms_data)
            except Exception:
                station = base.split('-')
                base_new = '%s-%s' % (station[1], station[0])
                try:
                    ax1.errorbar(freq_vis2, vis2, yerr=e_vis2, marker='.', linestyle='None', color=dic_color[base_new],
                                 ecolor='lightgray', elinewidth=0.5, ms=ms_data, label=bl1)
                except KeyError:
                    ax1.errorbar(freq_vis2, vis2, yerr=e_vis2, marker='.', linestyle='None', color='tab:blue',
                                 ecolor='lightgray', elinewidth=0.5, ms=ms_data)  # , label=bl1)
            # ax1.scatter(freq_vis2, vis2, s=8, c = dic_color[base], label = bl1)

            if model:
                u, v, wl = data.u[i], data.v[i], data.wl[cond]
                mod = comput_V2([u, v, wl], param, model_target)
                ax1.plot(freq_vis2, mod, '-', color='k',
                         alpha=.5, zorder=100, ms=ms_model)
            ncolor += 1

    if model:
        ax1.plot(0, 0, 'k-', label=label)

    if log:
        ax1.set_yscale('log')
        ax1.set_ylim(1e-4, 1.5e0)
    else:
        ax1.set_ylim(v2min, v2max)
    ax1.set_xlim(fmin-2, fmax+2)
    ax1.legend(fontsize=7)
    ax1.set_ylabel(r'V$^2$', fontsize=12)
    # ax1.set_xticklabels([])
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

            N_cp_rest += len(cp)

            # ax2.scatter(freq_cp, cp, s=5, c = colors2[i])
            ax2.errorbar(freq_cp, cp, yerr=e_cp, marker='.', linestyle='None', color='tab:blue',
                         ecolor='lightgray', label=bl1, elinewidth=0.5, ms=ms_data)

            if model:
                u1, u2, u3 = data.u1[i], data.u2[i], data.u3[i]
                v1, v2, v3 = data.v1[i], data.v2[i], data.v3[i]
                wl2 = data.wl[cond]
                X = [u1, u2, u3, v1, v2, v3, wl2]
                mod_cp = comput_CP(X, param, model_target)
                ax2.plot(data.freq_cp[i][cond], mod_cp, 'k+', ms=3, zorder=100)
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

    dic_ind, l_base = Index2Tel(tab)

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


def plot_uvdata_im(data, rot=0, unit_vis='lambda', onecolor=False, color='r', ms=3, alpha=1):
    """ """
    global dic_color

    if unit_vis == 'lambda':
        f = 1e6
    elif unit_vis == 'arcsec':
        f = rad2mas(1)/1000.

    try:
        npts = len(data)
        one = False
    except TypeError:
        one = True
        npts = 1

    if one:
        l_bl_label, l_color = [], []
        index2tel = Index2Tel([data])[0]
        for i in range(len(data.index)):
            tel1, tel2 = index2tel[data.index[i]
                                   [0]], index2tel[data.index[i][1]]
            name_bl = '%s-%s' % (tel1, tel2)
            name_bl_r = '%s-%s' % (tel2, tel1)
            try:
                c = dic_color[name_bl]
                label_bl = name_bl
            except KeyError:
                c = dic_color[name_bl_r]
                label_bl = name_bl_r
            l_bl_label.append(label_bl)
            if onecolor:
                c = color
            l_color.append(c)

        tab = data
        for j in range(6):
            angle = np.deg2rad(rot)

            um0 = tab.u[j]/tab.wl/f
            vm0 = tab.v[j]/tab.wl/f

            um = um0*np.cos(angle) - vm0*np.sin(angle)
            vm = um0*np.sin(angle) + vm0*np.cos(angle)

            plt.scatter(um, vm, s=5, color=l_color[j], alpha=alpha)
            plt.scatter(-um, -vm, s=5, color=l_color[j], alpha=alpha)

    else:
        for i in range(npts):
            tab = data[i]
            l_bl_label, l_color = [], []
            index2tel = Index2Tel([tab])[0]
            for k in range(len(tab.index)):
                tel1, tel2 = index2tel[tab.index[k]
                                       [0]], index2tel[tab.index[k][1]]
                name_bl = '%s-%s' % (tel1, tel2)
                name_bl_r = '%s-%s' % (tel2, tel1)
                try:
                    c = dic_color[name_bl]
                    label_bl = name_bl
                except KeyError:
                    c = dic_color[name_bl_r]
                    label_bl = name_bl_r
                l_bl_label.append(label_bl)
                if onecolor:
                    c = color
                l_color.append(c)

            for j in range(6):
                angle = np.deg2rad(rot)

                um0 = tab.u[j]/tab.wl/f
                vm0 = tab.v[j]/tab.wl/f

                um = um0*np.cos(angle) - vm0*np.sin(angle)
                vm = um0*np.sin(angle) + vm0*np.cos(angle)
                plt.scatter(um, vm, s=5, color=l_color[j], alpha=alpha)
                plt.scatter(-um, -vm, s=5, color=l_color[j], alpha=alpha)

    return None


def plot_uvdata_im_v2(tab, bmax=150, use_flag=False, cond_uncer=False, cond_wl=False,
                      rot=0, wl_min=None, wl_max=None, rel_max=None):
    if type(tab) == list:
        data = tab[0]
    else:
        data = tab
        tab = [tab]

    nwl = len(data.wl)
    nbl = data.vis2.shape[0]

    list_bl = []

    dic_ind, l_base = Index2Tel(tab)

    if cond_wl:
        try:
            float(wl_min)
        except TypeError:
            cprint('-'*38, 'red')
            cprint('Warnings: wavelengths limits not set!', 'red')
            cprint('-'*38, 'red')

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
                angle = np.deg2rad(rot)
                um = u*np.cos(angle) - v*np.sin(angle)
                vm = u*np.sin(angle) + v*np.cos(angle)

                plt.scatter(
                    um, vm, s=15, c=dic_color[base], label=bl1, marker='o')
                plt.scatter(-um, -vm, s=15, c=dic_color[base], marker='o')
            except Exception:
                station = base.split('-')
                base_new = '%s-%s' % (station[1], station[0])
                um = u*np.cos(angle) - v*np.sin(angle)
                vm = u*np.sin(angle) + v*np.cos(angle)
                plt.scatter(
                    um, vm, s=15, c=dic_color[base_new], label=bl1, marker='o')
                plt.scatter(-um, -vm, s=15, c=dic_color[base_new], marker='o')
