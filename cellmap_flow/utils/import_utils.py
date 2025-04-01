import importlib

def check_dependencies(dependencies):
    missing_packages = []
    
    for dep in dependencies:
        try:
            importlib.import_module(dep)
        except ImportError:
            missing_packages.append(dep)
    
    if missing_packages:
        raise ImportError(
            f"Postprocessing dependencies not installed: {', '.join(missing_packages)}.\n"
            "Please install them using:\n\n"
            "    pip install cellmap-flow[postprocess]"
        )