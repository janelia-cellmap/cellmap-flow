import importlib


class MissingDependencyError(ImportError):
    """Raised when one or more optional packages required by a feature aren't installed."""

    def __init__(self, missing_packages):
        self.missing_packages = list(missing_packages)
        super().__init__(
            f"Missing dependencies: {', '.join(self.missing_packages)}.\n"
            "Install them from the dashboard's install prompt, or manually with:\n\n"
            f"    pip install {' '.join(self.missing_packages)}"
        )


def get_missing_dependencies(dependencies):
    """Return the subset of `dependencies` (import names) that can't be imported."""
    missing_packages = []

    for dep in dependencies:
        try:
            importlib.import_module(dep)
        except ImportError:
            missing_packages.append(dep)

    return missing_packages


def check_dependencies(dependencies):
    missing_packages = get_missing_dependencies(dependencies)
    if missing_packages:
        raise MissingDependencyError(missing_packages)


# Allow-list of packages the dashboard's "install missing dependency" button may
# install. Keys are import names (as used in `check_dependencies`); values are the
# exact pip install spec. The install endpoint only accepts packages from this map -
# never an arbitrary string from the client - since it runs on the server.
INSTALLABLE_PACKAGES = {
    "mwatershed": "mwatershed @ git+https://github.com/pattonw/mwatershed",
    "funlib.math": "funlib.math @ git+https://github.com/funkelab/funlib.math.git",
    "fastmorph": "fastmorph",
    "fastremap": "fastremap",
    "pymorton": "pymorton",
    "edt": "edt",
    "neuroglancer": "neuroglancer",
    "dacapo": "dacapo-ml",
    "bioimageio.core": "bioimageio.core[onnx,pytorch]==0.7.0",
}
