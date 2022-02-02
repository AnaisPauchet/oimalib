import pytest
from matplotlib import pyplot as plt

import oimalib


@pytest.mark.usefixtures("close_figures")
def test_plot_uv(example_oifits_grav):
    d = oimalib.load(example_oifits_grav, simu=True)
    oimalib.plot_uv(d)
    assert isinstance(d, dict)
    assert plt.gcf().number == 1


@pytest.mark.usefixtures("close_figures")
def test_plot_oidata(example_oifits_grav):
    d = oimalib.load(example_oifits_grav, simu=True)
    oimalib.plot_oidata(d)
    assert isinstance(d, dict)
    assert plt.gcf().number == 1


@pytest.mark.usefixtures("close_figures")
def test_plot_residual(example_oifits_grav):
    d = oimalib.load(example_oifits_grav, simu=True)
    param = {"model": "binary", "x0": 0, "y0": 0, "sep": 3, "pa": 45, "dm": 3}
    oimalib.plot_residuals(d, param)
    assert isinstance(d, dict)
    assert plt.gcf().number == 2


@pytest.mark.usefixtures("close_figures")
def test_plot_spectra(example_oifits_rmat):
    d = oimalib.load(example_oifits_rmat, simu=True)
    oimalib.plot_spectra(d, wl_lim=[3.5, 1], norm=False)
    assert isinstance(d, dict)
    assert plt.gcf().number == 1


@pytest.mark.usefixtures("close_figures")
@pytest.mark.parametrize("choice", [True, False])
def test_plot_spectra_line(example_oifits_rgrav, choice):
    d = oimalib.load(example_oifits_rgrav)
    oimalib.plot_spectra(d, tellu=False, div=choice, speed=choice, rest=2.166)
    assert isinstance(d, dict)
    assert plt.gcf().number == 1


@pytest.mark.usefixtures("close_figures")
def test_plot_spectra_tellu(example_oifits_rgrav):
    d = oimalib.load(example_oifits_rgrav)
    oimalib.plot_spectra(d, tellu=True)
    assert isinstance(d, dict)
    assert plt.gcf().number == 1


@pytest.mark.usefixtures("close_figures")
def test_plot_dvis(example_oifits_rgrav):
    d = oimalib.load(example_oifits_rgrav)
    oimalib.plot_dvis(d)
    assert isinstance(d, dict)
    assert plt.gcf().number == 1


def test_fail_load(example_oifits_grav, capsys):
    d = oimalib.load(example_oifits_grav, simu=False)
    captured = capsys.readouterr()
    txt_error = "Your dataset seems to be a simulation (from aspro2), you should add simu=True.\n"
    assert d is None
    assert captured.err == txt_error


@pytest.mark.usefixtures("close_figures")
@pytest.mark.parametrize("i", [0, 1])
def test_plot_triplet(example_oifits_grav, i):
    d = oimalib.load(example_oifits_grav, simu=True)
    oimalib.plotting.check_closed_triplet(d, i=i)
    assert isinstance(d, dict)
    assert plt.gcf().number == 1


@pytest.mark.usefixtures("close_figures")
@pytest.mark.parametrize("choice", [True, False])
def test_plot_image_model(example_oifits_grav, choice):
    d = oimalib.load(example_oifits_grav, simu=True)
    param = {"model": "binary", "x0": 0, "y0": 0, "sep": 3, "pa": 45, "dm": 3}
    wl = 2e-6
    oimalib.plot_image_model(
        wl,
        base_max=130,
        param=param,
        fov=50,
        npts=64,
        corono=choice,
        expert_plot=choice,
        apod=choice,
        hamming=choice,
        cont=choice,
    )
    assert isinstance(d, dict)
    assert plt.gcf().number == 1


@pytest.mark.usefixtures("close_figures")
def test_plot_model_with_obs(example_oifits_grav):
    d = oimalib.load(example_oifits_grav, simu=True)
    param = {"model": "binary", "x0": 0, "y0": 0, "sep": 3, "pa": 45, "dm": 3}
    wl = 2e-6
    obs = oimalib.format_obs(d)
    oimalib.plot_image_model(wl, base_max=130, param=param, fov=50, npts=64, obs=obs)
    assert isinstance(d, dict)
    assert plt.gcf().number == 1
