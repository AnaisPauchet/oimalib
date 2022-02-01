import pytest
import pathlib


@pytest.fixture(scope="session")
def global_datadir():
    return pathlib.Path(__file__).parent / "data"


@pytest.fixture()
def example_oifits_mat(global_datadir):
    return global_datadir / "example_MATISSE.oifits"


@pytest.fixture()
def example_oifits_grav(global_datadir):
    return global_datadir / "example_GRAVITY.oifits"
