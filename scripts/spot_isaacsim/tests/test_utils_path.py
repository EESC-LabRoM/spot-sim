from scripts.spot_isaacsim.utils.path import resolve_asset_path


def test_url_passthrough():
    assert resolve_asset_path("https://example.com/asset.usd") == "https://example.com/asset.usd"
    assert resolve_asset_path("omniverse://localhost/asset.usd") == "omniverse://localhost/asset.usd"


def test_absolute_passthrough(tmp_path):
    p = str(tmp_path / "asset.usd")
    assert resolve_asset_path(p) == p


def test_relative_resolves_to_absolute():
    result = resolve_asset_path("assets/test.usd")
    assert result.startswith("/")
    assert result.endswith("simulation/assets/test.usd")
