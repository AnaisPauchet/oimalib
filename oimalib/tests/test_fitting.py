import numpy as np
import pytest

import oimalib


@pytest.mark.filterwarnings(
    "ignore::matplotlib._api.deprecation.MatplotlibDeprecationWarning"
)
@pytest.mark.parametrize("method", ["normal", "prior", "alex"])
def test_mcmc(example_oifits_grav, method):
    data = [oimalib.load(example_oifits_grav, cam="SC", simu=True)]
    fitOnly = ["dm", "sep", "pa"]
    param = {"model": "binary", "x0": 0, "y0": 0, "sep": 3, "pa": 45, "dm": 3}
    prior = {"dm": [0, 6], "sep": [0, 10], "pa": [0, 90]}

    sampler = oimalib.mcmcfit(
        data,
        param,
        nwalkers=12,
        niter=100,
        prior=prior,
        fitOnly=fitOnly,
        threads=4,
        method=method,
    )

    fit = oimalib.get_mcmc_results(sampler, param, fitOnly, burnin=20)

    oimalib.plot_mcmc_results(sampler, labels=fitOnly, burnin=20)

    # Human checked values
    true_sep, true_pa, true_dm = 3.0, 45.0, 3.0

    sep = fit["best"]["sep"]
    pa = fit["best"]["pa"]
    dm = fit["best"]["dm"]

    prec = 0.1
    assert isinstance(fit, dict)
    # Check close true value
    assert sep == pytest.approx(true_sep, prec)
    assert pa == pytest.approx(true_pa, prec)
    assert dm == pytest.approx(true_dm, prec)


def test_uvline():
    npts = 10
    uvline = oimalib.fourier.UVLine(0, 10, 45, npts)
    assert isinstance(uvline, np.ndarray)
    assert len(uvline[0]) == npts


@pytest.mark.usefixtures("close_figures")
def test_smartfit(example_oifits_grav):
    data = [oimalib.load(example_oifits_grav, cam="SC", simu=True)]
    fitOnly = ["dm", "sep", "pa"]
    param = {"model": "binary", "x0": 0, "y0": 0, "sep": 3, "pa": 45, "dm": 3}

    oimalib.get_stat_data(data)

    fit = oimalib.smartfit(data, param, normalizeErrors=True, fitOnly=fitOnly)

    # Human checked values
    true_sep, true_pa, true_dm = 3.0, 45, 3.0

    sep, e_sep = fit["best"]["sep"], fit["uncer"]["sep"]
    pa, e_pa = fit["best"]["pa"], fit["uncer"]["pa"]
    dm, e_dm = fit["best"]["dm"], fit["uncer"]["dm"]

    assert isinstance(fit, dict)

    # Check close true value
    assert sep == pytest.approx(true_sep, 0.01)
    assert pa == pytest.approx(true_pa, 0.01)
    assert dm == pytest.approx(true_dm, 0.01)
    # Check small errors
    assert e_sep <= 0.01 * true_sep
    assert e_pa <= 0.01 * true_pa
    assert e_dm <= 0.01 * true_dm


@pytest.mark.parametrize("choice", [True, False])
def test_smartfit_withflag(example_oifits_rmat, choice):
    data = [oimalib.load(example_oifits_rmat)]
    fitOnly = ["dm", "sep", "pa"]
    param = {"model": "binary", "x0": 0, "y0": 0, "sep": 3, "pa": 45, "dm": 3}
    tobefit = ["V2", "CP"]  # , "dphi", "dvis"]
    fit = oimalib.smartfit(
        data,
        param,
        normalizeErrors=choice,
        fitOnly=fitOnly,
        verbose=choice,
        tobefit=tobefit,
    )
    assert isinstance(fit, dict)


def test_smartfit_hr(example_oifits_rgrav):
    data = [oimalib.load(example_oifits_rgrav)]
    fitOnly = ["dm", "sep", "pa"]
    param = {"model": "binary", "x0": 0, "y0": 0, "sep": 3, "pa": 45, "dm": 3}
    tobefit = ["V2", "CP", "dphi", "dvis"]
    fit = oimalib.smartfit(data, param, tobefit=tobefit, fitOnly=fitOnly)
    assert isinstance(fit, dict)


@pytest.mark.usefixtures("close_figures")
def test_chi2_curve(example_oifits_grav):
    data = [oimalib.load(example_oifits_grav, cam="SC", simu=True)]
    fitOnly = ["dm", "sep", "pa"]
    param = {"model": "binary", "x0": 0, "y0": 0, "sep": 3, "pa": 45, "dm": 3}

    fit = oimalib.smartfit(data, param, normalizeErrors=True, fitOnly=fitOnly)

    l_sep = np.linspace(2.5, 4.5, 20)

    pname = "sep"

    fitOnly = ["sep", "dm", "pa"]
    fit, error = oimalib.fitting.compute_chi2_curve(
        data, pname, param, l_sep, fitOnly=fitOnly
    )

    true_sep = 3
    sep = fit["best"][pname]

    rel_err = error / sep

    assert isinstance(fit, dict)
    assert sep == pytest.approx(true_sep, abs=error)
    assert rel_err <= 0.1
