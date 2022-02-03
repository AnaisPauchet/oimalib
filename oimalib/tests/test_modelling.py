import munch
import numpy as np
import pytest

import oimalib


@pytest.mark.parametrize("choice", [True, False])
def test_grid(example_model, choice):
    grid = oimalib.model2grid(example_model, fliplr=choice)

    wl0 = grid.wl
    tmp = 50.0 / wl0
    x = grid.sign * tmp
    y = tmp
    pts = (wl0, x, y)

    cv_real = grid.real(pts)
    cv_imag = grid.imag(pts)
    cond_all_inf_1 = False in (cv_imag <= 1)
    assert isinstance(grid, munch.Munch)
    assert isinstance(cv_real, np.ndarray)
    assert isinstance(cv_imag, np.ndarray)
    assert ~cond_all_inf_1


def test_grid_user(example_model_nochromatic):
    grid = oimalib.model2grid(
        example_model_nochromatic, wl_user=np.array([3]), pix_user=1.0
    )

    wl0 = grid.wl
    tmp = 50.0 / wl0
    x = grid.sign * tmp
    y = tmp

    cv_real = grid.real(x, y)
    cv_imag = grid.imag(x, y)
    cond_all_inf_1 = False in (cv_imag <= 1)
    assert isinstance(grid, munch.Munch)
    assert isinstance(cv_real, np.ndarray)
    assert isinstance(cv_imag, np.ndarray)
    assert ~cond_all_inf_1


def test_compute_model_grid(example_model, example_oifits_rmat):
    d = oimalib.load(example_oifits_rmat)
    grid = oimalib.model2grid(example_model)
    mod_v2_grid, mod_cp_grid = oimalib.compute_grid_model(d, grid, verbose=True)
    ncp = len(d.cp)
    nbl = len(d.vis2)
    nwl = len(d.wl)
    assert np.shape(mod_v2_grid)[1] == nbl
    assert np.shape(mod_cp_grid)[1] == ncp
    assert np.shape(mod_cp_grid)[2] == nwl


def test_compute_model_grid_user(example_model_nochromatic, example_oifits_rmat):
    d = oimalib.load(example_oifits_rmat)
    grid = oimalib.model2grid(
        example_model_nochromatic, wl_user=np.array([3e-6]), pix_user=1.0
    )
    mod_v2_grid, mod_cp_grid = oimalib.compute_grid_model(d, grid, verbose=True)
    ncp = len(d.cp)
    nbl = len(d.vis2)
    nwl = len(d.wl)
    assert np.shape(mod_v2_grid)[0] == nbl
    assert np.shape(mod_cp_grid)[0] == ncp
    assert np.shape(mod_cp_grid)[1] == nwl
