__version__ = "0.1.0"
__version_info__ = tuple(int(i) for i in __version__.split("."))

# Load user-registered plugins on package import
from cellmap_flow.utils.plugin_manager import load_plugins as _load_plugins
_load_plugins()
