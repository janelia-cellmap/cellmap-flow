"""
Plugin manager for cellmap_flow.

Handles registration, loading, and management of user plugins
(ModelConfig, InputNormalizer, PostProcessor subclasses).

Plugins are stored in ~/.cellmap_flow/plugins/. ``load_plugins`` runs them,
so that their subclasses appear in __subclasses__() calls; the commands,
the dashboard and the finetune job call it when they start, and importing
cellmap_flow does not. A script that uses a plugin's classes calls it.

``analyze_script`` is the safety check a plugin passes to be registered;
a script model's script passes it too, each time it is loaded
(``models.configs.script.load_safe_config``).
"""

import ast
import logging
import shutil
from pathlib import Path
from typing import List


logger = logging.getLogger(__name__)

PLUGINS_DIR = Path.home() / ".cellmap_flow" / "plugins"
_plugins_loaded = False
# Keep references to plugin namespaces so classes don't get garbage collected
_plugin_namespaces: List[dict] = []


# What a user's script may not import or call. analyze_script is copied from
# https://github.com/janelia-cellmap/cellmap-segmentation-challenge/blob/6e9d842b9a90b0df22aa07946a4d1deed5c27504/src/cellmap_segmentation_challenge/utils/security.py
DISALLOWED_IMPORTS = {"os", "subprocess", "sys"}
# DISALLOWED_FUNCTIONS = {"eval", "exec", "open", "compile", "__import__"}
DISALLOWED_FUNCTIONS = {"eval", "exec", "compile", "__import__"}


def analyze_script(filepath):
    """
    Analyzes the script at `filepath` using `ast` for potentially unsafe imports and function calls.
    Returns a boolean indicating whether the script is safe and a list of detected issues.
    """
    issues = []
    with open(filepath, "r") as file:
        source_code = file.read()

    # Parse the code into an AST
    tree = ast.parse(source_code, filename=filepath)

    # Traverse the AST and analyze nodes
    for node in ast.walk(tree):
        # Check for disallowed imports
        if isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name in DISALLOWED_IMPORTS:
                    issues.append(f"Disallowed import detected: {alias.name}")

        elif isinstance(node, ast.ImportFrom):
            if node.module in DISALLOWED_IMPORTS:
                issues.append(f"Disallowed import detected: {node.module}")

        # Check for disallowed function calls
        elif isinstance(node, ast.Call):
            # If function is a direct name (e.g., `eval()`)
            if isinstance(node.func, ast.Name) and node.func.id in DISALLOWED_FUNCTIONS:
                issues.append(f"Disallowed function call detected: {node.func.id}")
            # If function is an attribute call on a known-unsafe root
            # (e.g., `builtins.eval()` / `__builtins__.eval()`). Method calls
            # on user objects like `model.eval()` remain allowed.
            elif isinstance(node.func, ast.Attribute):
                base = node.func.value
                if (
                    node.func.attr in DISALLOWED_FUNCTIONS
                    and isinstance(base, ast.Name)
                    and base.id in {"builtins", "__builtins__"}
                ):
                    issues.append(
                        "Disallowed function call detected via attribute access: "
                        f"{base.id}.{node.func.attr}"
                    )

    # Return whether the script is safe (no issues found) and the list of issues
    is_safe = len(issues) == 0
    return is_safe, issues


def get_plugins_dir() -> Path:
    """Return the plugins directory, creating it if necessary."""
    PLUGINS_DIR.mkdir(parents=True, exist_ok=True)
    return PLUGINS_DIR


def _exec_plugin(filepath: str) -> None:
    """
    Execute a plugin file so its class definitions are registered.

    Unlike a script model's load_safe_config, this does not wrap the result
    in a Config object — we only need the side-effect of defining subclasses.

    The namespace is retained in _plugin_namespaces so class objects
    are not garbage-collected (which would remove them from __subclasses__).
    """
    with open(filepath, "r") as fh:
        code = fh.read()

    tree = ast.parse(code, filename=filepath)

    class ReplaceFileNode(ast.NodeTransformer):
        def visit_Name(self, node: ast.Name) -> ast.AST:
            if node.id == "__file__":
                return ast.Constant(value=str(filepath), kind=None)
            return node

    tree = ReplaceFileNode().visit(tree)
    code = ast.unparse(tree)
    namespace: dict = {"__file__": str(filepath), "__name__": Path(filepath).stem}
    exec(code, namespace)
    _plugin_namespaces.append(namespace)


def register_plugin(filepath: str, force: bool = False) -> Path:
    """
    Register a plugin by copying a Python file to ~/.cellmap_flow/plugins/.

    Args:
        filepath: Path to the Python file to register.
        force: Overwrite existing plugin with the same name.

    Returns:
        Path to the installed plugin file.

    Raises:
        FileNotFoundError: If the source file does not exist.
        FileExistsError: If a plugin with the same name already exists and force is False.
        ValueError: If the file is not a .py file or fails safety analysis.
    """
    source = Path(filepath).resolve()

    if not source.exists():
        raise FileNotFoundError(f"File not found: {source}")

    if source.suffix != ".py":
        raise ValueError(f"Only .py files can be registered, got: {source.suffix}")

    # Safety check
    is_safe, issues = analyze_script(str(source))
    if not is_safe:
        msg = "Plugin contains unsafe elements:\n" + "\n".join(f"  - {i}" for i in issues)
        raise ValueError(msg)

    dest = get_plugins_dir() / source.name

    if dest.exists() and not force:
        raise FileExistsError(
            f"Plugin '{source.name}' already registered. Use --force to overwrite."
        )

    shutil.copy2(str(source), str(dest))
    logger.info(f"Registered plugin: {source.name} -> {dest}")
    return dest


def unregister_plugin(name: str) -> None:
    """
    Remove a registered plugin by filename.

    Args:
        name: Filename of the plugin (e.g. 'my_normalizer.py').
              The .py extension is added automatically if missing.
    """
    if not name.endswith(".py"):
        name = f"{name}.py"

    target = PLUGINS_DIR / name
    if not target.exists():
        raise FileNotFoundError(f"Plugin not found: {name}")

    target.unlink()
    logger.info(f"Unregistered plugin: {name}")


def list_plugins() -> List[Path]:
    """Return a sorted list of all registered plugin file paths.

    Does not create the plugins directory: this runs on every package import,
    including inside each LSF job, and only registering a plugin needs it.
    """
    if not PLUGINS_DIR.is_dir():
        return []
    return sorted(PLUGINS_DIR.glob("*.py"))


def load_plugins() -> int:
    """
    Load all registered plugins from ~/.cellmap_flow/plugins/.

    Each plugin file is executed so that any subclasses defined in it
    (ModelConfig, InputNormalizer, PostProcessor) become available
    through __subclasses__().

    Safe to call multiple times — plugins are only loaded once.

    Returns:
        Number of plugins successfully loaded.
    """
    global _plugins_loaded
    if _plugins_loaded:
        return 0
    _plugins_loaded = True
    plugins = list_plugins()
    loaded = 0

    for plugin_path in plugins:
        try:
            _exec_plugin(str(plugin_path))
            loaded += 1
            logger.debug(f"Loaded plugin: {plugin_path.name}")
        except Exception as exc:
            logger.warning(f"Failed to load plugin {plugin_path.name}: {exc}")

    if loaded:
        logger.info(f"Loaded {loaded} plugin(s) from {PLUGINS_DIR}")

    return loaded
