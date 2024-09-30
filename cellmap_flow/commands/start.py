import click
from cellmap_flow.core import start_flow
from cellmap_flow.data import generate_from_yaml
from cellmap_flow.settings import MODEL_YAML
import os
import zarr

from cellmap_flow.server import start, App


def check_zarr_data(container_path,dataset_path):
    """
    Checks if the given path is a valid Zarr dataset with a shape attribute.

    Args:
        path (str): The path to the Zarr dataset.

    Returns:
        bool: True if the path is a valid Zarr dataset with a shape attribute, False otherwise.
    """
    path = os.path.join(container_path, dataset_path)
    if not os.path.exists(path):
        return False

    try:
        zarr_data = zarr.open(path, mode="r")
        if isinstance(zarr_data, zarr.hierarchy.Group):
            print("Error: The provided path is a Zarr group, not a Zarr array.")
            return False
        if hasattr(zarr_data, "shape"):
            return True
    except Exception:
        print("Error: The provided path is not a valid Zarr dataset.")
        return False

    return False


def prompt_with_choices(prompt_text, choices, default_index=0):
    """
    Prompts the user with a list of choices and returns the selected choice.

    Args:
        prompt_text (str): The prompt text to display to the user.
        choices (list): The list of choices to present.
        default_index (int): The index of the default choice (0-based).

    Returns:
        str: The selected choice.
    """
    while True:
        click.echo(prompt_text)
        for i, choice in enumerate(choices, 1):
            click.echo(f"{i} - {choice}")

        # If the default_index is out of range, set to 0
        default_index = max(0, min(default_index, len(choices) - 1))

        try:
            # Prompt the user for input
            choice_num = click.prompt(
                f"Enter your choice (default: {choices[default_index]})",
                default=default_index + 1,
                type=int,
            )

            # Check if the provided number is valid
            if 1 <= choice_num <= len(choices):
                return choices[choice_num - 1]
            else:
                click.echo("Invalid choice number. Please try again.")
        except click.BadParameter:
            click.echo("Invalid input. Please enter a number.")


@click.command()
def start():
    """Start Cellmap Flow"""
    start_flow()
    models = generate_from_yaml(MODEL_YAML)
    model = prompt_with_choices("Enter number of model to start", list(models.keys()))
    checkpoints = models[model].checkpoints
    checkpoint = prompt_with_choices("Enter number of checkpoint to start", checkpoints)
    if not os.path.exists(checkpoint.path):
        raise FileNotFoundError(
            f"Checkpoint file {checkpoint.path} not found, check models.yaml"
        )
    valid = False
    while not valid:
        container_path = click.prompt("Enter the container .zarr path", type=str)
        dataset_path = click.prompt("Enter the path to the dataset", type=str)
        valid = check_zarr_data(container_path,dataset_path)
        if not valid:
            click.echo("Invalid data path. Please try again..")
    
    click.echo(f"Starting Cellmap Flow with model {model} and checkpoint {checkpoint}")


    app = App(checkpoint.path, container_path, dataset_path)
    start(app)
