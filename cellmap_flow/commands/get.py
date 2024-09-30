import click
from cellmap_flow.core import get_flow


@click.command()
@click.argument("id")
def get(id):
    """Get a flow by ID"""
    get_flow(id)
