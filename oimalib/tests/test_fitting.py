import pytest

import oimalib


@pytest.mark.filterwarnings(
    "ignore::matplotlib._api.deprecation.MatplotlibDeprecationWarning"
)
def test_mcmc(example_oifits_grav):
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
        fast=True,
    )

    fit = oimalib.get_mcmc_results(sampler, param, fitOnly, burnin=20)

    oimalib.plot_mcmc_results(sampler, labels=fitOnly, burnin=20)

    # Human checked values
    true_sep, true_pa, true_dm = 3.0, 45.0, 3.0

    sep, e_sep = fit["best"]["sep"], max(fit["uncer"]["sep_p"], fit["uncer"]["sep_m"])
    pa, e_pa = (
        fit["best"]["pa"],
        max(fit["uncer"]["pa_p"], fit["uncer"]["pa_m"]),
    )
    dm, e_dm = fit["best"]["dm"], max(fit["uncer"]["dm_p"], fit["uncer"]["dm_m"])

    prec = 0.02
    assert isinstance(fit, dict)
    # Check close true value
    assert sep == pytest.approx(true_sep, prec)
    assert pa == pytest.approx(true_pa, prec)
    assert dm == pytest.approx(true_dm, prec)
    # Check small errors
    assert e_sep <= prec * true_sep
    assert e_pa <= prec * true_pa
    assert e_dm <= prec * true_dm
