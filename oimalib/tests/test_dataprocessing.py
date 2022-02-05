import numpy as np

import oimalib


def test_data_select_wl(example_oifits_rgrav):
    d = oimalib.load(example_oifits_rgrav)
    param_sel = {"cond_wl": True, "wave_lim": [2.14, 2.18]}
    # Expected with only wavelength selection
    cond_expected = (d.wl >= param_sel["wave_lim"][0] * 1e-6) & (
        d.wl <= param_sel["wave_lim"][1] * 1e-6
    )
    flag_vis2 = d.flag_vis2
    flag_cp = d.flag_cp
    n_expected1 = len(d.wl[cond_expected & ~flag_vis2[0]]) * len(d.vis2)
    n_expected2 = len(d.wl[cond_expected & ~flag_cp[0]]) * len(d.cp)
    n_expected = n_expected1 + n_expected2

    d_sel = oimalib.select_data(d, **param_sel)
    npts_sel = oimalib.get_stat_data([d_sel])
    assert isinstance(d, dict)
    assert n_expected == npts_sel


def test_data_select_err(example_oifits_rgrav):
    d = oimalib.load(example_oifits_rgrav)
    param_sel = {"cond_uncer": True, "rel_max": 1}
    # Expected with only uncertainty selection
    vis2 = d.vis2
    e_vis2 = d.e_vis2
    e_cp = d.e_cp
    cond_expected_v2 = (e_vis2 / vis2) <= (param_sel["rel_max"] / 100.0)
    cond_expected_cp = e_cp > 0
    flag_vis2 = d.flag_vis2
    flag_cp = d.flag_cp
    vis2_sel = vis2[cond_expected_v2 & ~flag_vis2]
    n_expected1 = len(np.hstack(vis2_sel.flatten()))
    n_expected2 = len(d.cp[cond_expected_cp & ~flag_cp].flatten())
    n_expected = n_expected1 + n_expected2

    d_sel = oimalib.select_data(d, **param_sel)
    npts_sel = oimalib.get_stat_data([d_sel])
    assert isinstance(d, dict)
    assert n_expected == npts_sel


def test_data_select_flag(example_oifits_rgrav):
    d = oimalib.load(example_oifits_rgrav)
    param_sel = {"use_flag": False}
    # Expected with only uncertainty selection
    vis2 = d.vis2
    cp = d.cp
    n_expected1 = len(np.hstack(vis2.flatten()))
    n_expected2 = len(np.hstack(cp.flatten()))
    n_expected = n_expected1 + n_expected2
    d_sel = oimalib.select_data(d, **param_sel)
    npts_sel = oimalib.get_stat_data([d_sel])
    assert isinstance(d, dict)
    assert n_expected == npts_sel
