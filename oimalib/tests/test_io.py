import oimalib


def test_load_file(example_oifits_mat):
    d = oimalib.load(example_oifits_mat, simu=True)
    assert isinstance(d, dict)


def test_load_file_grav(example_oifits_grav):
    d = oimalib.load(example_oifits_grav, simu=True)
    assert isinstance(d, dict)


def test_load_repo(global_datadir):
    d = oimalib.oifits.dir2data(str(global_datadir), ext="oifits")
    assert isinstance(d, list)
    assert isinstance(d[0], dict)
