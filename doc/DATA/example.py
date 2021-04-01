import oimalib
from matplotlib import pyplot as plt
from oimalib.oifits import data2obs

oifile = 'doc/DATA/example_MATISSE_Betelgeuse.oifits'

oifile = 'doc/DATA/example_GRAVITY_Binary_s3_pa45.oifits'

d = oimalib.load(oifile, cam='SC', simu=True)

# oifile_nrm = '/Users/asoulain/Documents/Postdoc_JWST/AMICAL/Saveoifits/example_fakebinary_NIRISS.oifits'
# d = oimalib.load(oifile_nrm)

param = {'model': 'binary',
         'x0': 0,
         'y0': 0,
         'sep': 3,
         'pa': 45,
         'dm': 3
         }

param2 = {'model': 'disk',
          'x0': 0,
          'y0': 0,
          'diam': 32
          }

obs = data2obs(d, err_scale_v2=1)

fit = oimalib.smartfit(obs, param, fitOnly=['dm', 'sep', 'pa'],
                       fitCP=True)

oimalib.plot_oidata(d, err_scale_v2=1, cmax=20)

oimalib.plot_uv(d)

oimalib.plot_residuals(d, param)
plt.show(block=True)
