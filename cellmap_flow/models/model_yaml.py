import yaml


def load_model_paths(yaml_file: str) -> dict:
    with open(yaml_file, "r") as f:
        data = yaml.safe_load(f)
    return data
