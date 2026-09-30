__version__ = "0.2.3"
__version_info__ = tuple(int(i) for i in __version__.split("."))

# Plugins (~/.cellmap_flow/plugins/*.py) are not loaded here: the commands
# and the dashboard load them when they start (plugins.load_plugins), and a
# script that wants them calls load_plugins() itself.
