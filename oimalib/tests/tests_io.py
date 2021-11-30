import pathlib

import oimalib
import pytest


@pytest.fixture(scope="session")
def global_datadir():
    return pathlib.Path(__file__).parent / "data"


@pytest.fixture()
def example_oifits_mat(global_datadir):
    return global_datadir / "example_MATISSE.oifits"


@pytest.fixture()
def example_oifits_grav(global_datadir):
    return global_datadir / "example_GRAVITY.oifits"


def test_load_file(example_oifits_mat):
    d = oimalib.load(example_oifits_mat, simu=True)
    assert isinstance(d, dict)


def test_load_file_grav(example_oifits_grav):
    d = oimalib.load(example_oifits_grav, simu=True)
    assert isinstance(d, dict)
