# -*- coding: utf-8 -*-
"""
Created on Wed Nov  4 13:16:58 2019

@author: asoulain
"""

from math import atan2, degrees

import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import interp1d

from . import binary_integrator as bi
from astools.all import Orient_vector

M_sun = bi.M_sun
au = bi.AU
year = bi.year
G = bi.G


def kepler_solve(t, P, ecc):
    """ Compute deformated polar coordinates for a eccentric binary. """
    maxj = 50                       # Max number of iteration
    tol = 1e-8                      # Convergence tolerance
    M = 2*np.pi/P*t
    E = np.zeros(len(t))
    tj = 0
    for i in range(len(t)):
        E0 = M[i]
        # Newton's formula to solve for eccentric anomoly
        for j in range(1, maxj+1):
            E1 = E0 - (E0 - ecc*np.sin(E0)-M[i])/(1 - ecc*np.cos(E0))
            if abs(E1 - E0) < tol:
                E0 = E1
        E[i] = E1
        tj = tj+j

    # --- Compute 2-dimensional spiral angles & radii --- #
    theta = 2*np.arctan(np.sqrt((1+ecc)/(1-ecc))*np.tan(E/2))
    return theta, E


def AngleBtw2Points(pointA, pointB):
    """ Compute angle between 2 points in space. """
    changeInX = pointB[0] - pointA[0]
    changeInY = pointB[1] - pointA[1]
    theta = round(degrees(atan2(changeInX, changeInY)), 2)

    if theta < 0:
        theta = 360 + theta

    return theta


def getBinaryPos(mjd, param, mjd0=57590, revol=1, v=5, au=False, anim=False, display=False):
    """ Compute the spatial positon of a binary star. """
    P = param['P']
    e = param['e']
    M1 = param['M1']
    M2 = param['M2']
    dpc = param['dpc']
    i = param['incl']
    angleSky = param['angleSky']
    phi = param['angle_0']

    diff = mjd0 - mjd

    # set the masses
    M_star1 = M1*M_sun      # star 1's mass
    M_star2 = M2*M_sun      # star 2's mass

    P2 = P * 24 * 3600.
    a = ((G * (M_star1 + M_star2) * P2**2)/(4*np.pi**2))**(1/3.)

    a_au = a/bi.AU

    fact = diff/P

    pphase = -fact % 1  # + 0.5

    if pphase > 1:
        pphase = abs(1 - pphase)

    # set the eccentricity
    ecc = e
    theta = np.pi
    annotate = False

    b = bi.Binary(M_star1, M_star2, a,
                  ecc, theta, annotate=annotate)

    # set the timestep in terms of the orbital period
    dt = b.P/1000
    tmax = revol * b.P  # maximum integration time

    b.integrate(dt, tmax)
    s1 = b.orbit1
    s2 = b.orbit2

    if au:
        dpc = 1
        unit = 'AU'
    else:
        unit = 'mas'
        pass

    incl, angleSky, phi = np.deg2rad(i), np.deg2rad(angleSky), np.deg2rad(-phi)

    X1_b, Y1_b = -s1.x/bi.AU/dpc, s1.y/bi.AU/dpc
    X2_b, Y2_b = -s2.x/bi.AU/dpc, s2.y/bi.AU/dpc

    l_theta = []
    for i in range(len(X1_b)):
        theta = AngleBtw2Points([X1_b[i], Y1_b[i]], [X2_b[i], Y2_b[i]])
        l_theta.append(theta)

    l_theta = np.array(l_theta)

    X1_rot1 = X1_b*np.cos(phi)+Y1_b*np.sin(phi)
    Y1_rot1 = -X1_b*np.sin(phi)+Y1_b*np.cos(phi)

    X2_rot1 = X2_b*np.cos(phi)+Y2_b*np.sin(phi)
    Y2_rot1 = -X2_b*np.sin(phi)+Y2_b*np.cos(phi)

    X1_rot2, Y1_rot2 = X1_rot1 * np.cos(incl), Y1_rot1
    X2_rot2, Y2_rot2 = X2_rot1 * np.cos(incl), Y2_rot1

    X1 = X1_rot2*np.cos(angleSky)+Y1_rot2*np.sin(angleSky)
    Y1 = -X1_rot2*np.sin(angleSky)+Y1_rot2*np.cos(angleSky)

    X2 = X2_rot2*np.cos(angleSky)+Y2_rot2*np.sin(angleSky)
    Y2 = -X2_rot2*np.sin(angleSky)+Y2_rot2*np.cos(angleSky)

    phase = s1.t/b.P
    r = ((s1.x[:]**2 + s1.y[:]**2)**0.5 +
         (s2.x[:]**2 + s2.y[:]**2)**0.5)/bi.AU/dpc

    r2 = ((X1[:]**2 + Y1[:]**2)**0.5 + (X2[:]**2 + Y2[:]**2)**0.5)

    fx1 = interp1d(phase, X1)
    fy1 = interp1d(phase, Y1)
    fx2 = interp1d(phase, X2)
    fy2 = interp1d(phase, Y2)
    fr = interp1d(phase, r)
    ftheta = interp1d(phase, l_theta)

    xmod1, ymod1 = fx1(pphase), fy1(pphase)
    xmod2, ymod2 = fx2(pphase), fy2(pphase)

    r_act = fr(pphase)
    theta_act = ftheta(pphase)

    try:
        rs = param['s_prod']/dpc
        no_rs = False
    except KeyError:
        no_rs = True
        rs = np.nan

    # Now we fix the WR position to the center.
    x_star1, y_star1 = 0, 0
    x_star2, y_star2 = xmod2 - xmod1, ymod2 - ymod1

    X_star1, Y_star1 = X1-X1, Y1-Y1
    X_star2, Y_star2 = X2-X1, Y2-Y1

    if r.min() > rs:
        nodust = True
        days_prod = 0
    else:
        nodust = False
        if no_rs:
            cond_prod1 = (r <= 1e100) & (phase < 0.5)
            cond_prod2 = (r <= 1e100) & (phase > 0.5)
            days_prod = P
        else:
            cond_prod1 = (r <= rs) & (phase <= 0.5)
            cond_prod2 = (r <= rs) & (phase > 0.5)
            days_prod = 2*(phase[cond_prod1].max())*P

    tab = {'star1': {'x': x_star1/param['dpc'], 'y': y_star1/param['dpc']},
           'star2': {'x': x_star2/param['dpc'], 'y': y_star2/param['dpc']},
           'orbit1': {'x': X_star1, 'y': Y_star1},
           'orbit2': {'x': X_star2, 'y': Y_star2},
           'phase': phase,
           'r': r,
           'cond': r <= rs,
           'r_act': r_act,
           'theta_act': theta_act,
           'pphase': pphase,
           's1': s1,
           's2': s2,
           'l_theta': l_theta,
           'rs': rs,
           'a': a_au,
           'd_prod': days_prod,
           'f_prod': 100*(days_prod/P)
           }

    x1, y1 = tab['star1']['x'], tab['star1']['y']
    x2, y2 = tab['star2']['x'], tab['star2']['y']
    t = AngleBtw2Points([x1, y1], [x2, y2])

    tab['t'] = t

    d_post_pa = pphase * P

    v = 2*np.max([X_star2.max(), Y_star2.max(), abs(X_star2.min()), abs(Y_star2.min())])
    if display:
        xmin, xmax, ymin, ymax = -v, v, -v, v
        plt.figure(figsize=(10, 5))
        if anim:
            plt.clf()
        plt.subplot(1, 2, 1)
        plt.text(0.8*v, 0.8*v, r'$\theta$ = %2.1f $°$ (%2.1f d)' %
                 (t, d_post_pa))
        plt.plot(X_star2, Y_star2, '#008b8b', alpha=.2, linewidth=1)
        plt.plot(x_star1, y_star1, '*',
                 color='crimson', label='WR star')
        plt.plot(x_star2, y_star2, '*',
                 color='#008b8b', label='O star')
        plt.vlines(0, -v, v, linewidth=1, color='gray', alpha=.5)
        plt.hlines(0, -v, v, linewidth=1, color='gray', alpha=.5)
        plt.legend()
        plt.xlabel('X [%s]' % unit)
        plt.ylabel('Y [%s]' % unit)
        plt.axis([xmax, xmin, ymin, ymax])

        plt.subplot(1, 2, 2)
        plt.plot(phase, r, linewidth=1, linestyle='-', zorder=2,
                 label='$\phi_{prod}$ = %2.1f %% (%2.1f d)' % (tab['f_prod'], tab['d_prod']))
        plt.plot(pphase, r_act, 'o', color='#008b8b', zorder=3,
                 label='r = %2.2f %s' % (r_act, unit))
        if not nodust:
            plt.plot(phase[cond_prod1], r[cond_prod1],
                     '-', color='#a0522d', lw=4, alpha=.5)
            plt.plot(phase[cond_prod2], r[cond_prod2],
                     '-', color='#a0522d', lw=4, alpha=.5)
        plt.hlines(rs, 0, 1, linestyle='-.', color='#006400',
                   label=r'Threshold d$_{nuc}$ = %2.2f' % rs)
        plt.legend(loc='best')
        plt.grid(alpha=.2)
        plt.xlim(0, 1)
        plt.ylim(0, 2*r.mean())
        plt.xlabel('Orbital phase')
        plt.ylabel('r [%s]' % unit)
        plt.tight_layout()
        if anim:
            plt.pause(0.3)
            plt.draw()
    return tab
