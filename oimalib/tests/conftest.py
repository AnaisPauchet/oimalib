import pathlib

import pytest
from matplotlib import pyplot as plt


@pytest.fixture()
def close_figures():
    plt.close("all")
    yield
    plt.close("all")


@pytest.fixture(scope="session")
def global_datadir():
    return pathlib.Path(__file__).parent / "data"


@pytest.fixture()
def example_oifits_mat(global_datadir):
    return global_datadir / "example_MATISSE.oifits"


@pytest.fixture()
def example_oifits_grav(global_datadir):
    return global_datadir / "example_GRAVITY.oifits"


@pytest.fixture()
def example_oifits_rmat(global_datadir):
    return global_datadir / "example_MATISSE_real.oifits"


@pytest.fixture()
def example_oifits_rgrav(global_datadir):
    return global_datadir / "example_GRAVITY_real.fits"


@pytest.fixture()
def example_model(global_datadir):
    return global_datadir / "example_model_chromatic.fits"


@pytest.fixture()
def example_model_nochromatic(global_datadir):
    return global_datadir / "example_model_nochromatic.fits"
