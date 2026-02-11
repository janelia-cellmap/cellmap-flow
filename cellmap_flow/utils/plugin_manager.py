"""
Plugin manager for cellmap_flow.

Handles registration, loading, and management of user plugins
(ModelConfig, InputNormalizer, PostProcessor subclasses).

Plugins are stored in ~/.cellmap_flow/plugins/ and loaded automatically
at startup so that custom subclasses appear in __subclasses__() calls.
"""

import ast
import logging
import shutil
from pathlib import Path
from typing import List

from cellmap_flow.utils.load_py import analyze_script

logger = logging.getLogger(__name__)

PLUGINS_DIR = Path.home() / ".cellmap_flow" / "plugins"
_plugins_loaded = False
# Keep references to plugin namespaces so classes don't get garbage collected
_plugin_namespaces: List[dict] = []


def get_plugins_dir() -> Path:
    """Return the plugins directory, creating it if necessary."""
    PLUGINS_DIR.mkdir(parents=True, exist_ok=True)
    return PLUGINS_DIR


def _exec_plugin(filepath: str) -> None:
    """
    Execute a plugin file so its class definitions are registered.

    Unlike load_safe_config, this does not wrap the result in a Config
    object — we only need the side-effect of defining subclasses.

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

    target = get_plugins_dir() / name
    if not target.exists():
        raise FileNotFoundError(f"Plugin not found: {name}")

    target.unlink()
    logger.info(f"Unregistered plugin: {name}")


def list_plugins() -> List[Path]:
    """Return a sorted list of all registered plugin file paths."""
    plugins_dir = get_plugins_dir()
    return sorted(plugins_dir.glob("*.py"))


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
