import oimalib


def test_plot_model(example_oifits_grav):
    d = oimalib.load(example_oifits_grav, simu=True)
    oimalib.plot_uv(d)
    assert isinstance(d, dict)
