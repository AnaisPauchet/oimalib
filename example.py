from matplotlib import pyplot as plt

import optimal

oifile = 'DATA/example_MATISSE_Betelgeuse.oifits'

d = optimal.loadc(oifile)

param = {'model': 'disk',
         'x0': 0,
         'y0': 0,
         'diam': 21}

obs = optimal.fitting.fits2obs(oifile, extra_error_v2=0.1)

fit = optimal.smartfit(obs, param, fitOnly=['diam'], fitCP=False)

optimal.plot_oidata(d, model=True, param=fit['best'], extra_error_v2=0.1)
optimal.plot_uv(d, bmax=20)

plt.show(block=True)
