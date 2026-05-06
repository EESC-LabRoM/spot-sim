from pathlib import Path


def resolve_asset_path(usd_path: str) -> str:
    """Resolve a USD asset path to an absolute string.

    URLs (https://, http://, omniverse://) are returned as-is.
    Absolute filesystem paths are returned as-is.
    Relative paths are resolved from the simulation/ directory
    (4 parents up from utils/path.py: utils/ → spot_isaacsim/ → scripts/ → simulation/).
    """
    if usd_path.startswith(("https://", "http://", "omniverse://")):
        return usd_path
    p = Path(usd_path)
    if p.is_absolute():
        return str(p)
    simulation_root = Path(__file__).resolve().parent.parent.parent.parent
    return str((simulation_root / p).resolve())
