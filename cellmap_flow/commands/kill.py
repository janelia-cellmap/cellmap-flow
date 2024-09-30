import click
from cellmap_flow.core import kill_flow


@click.command()
@click.argument("id")
def kill(id):
    """Kill a flow by ID"""
    kill_flow(id)
