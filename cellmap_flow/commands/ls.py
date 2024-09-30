import click
from cellmap_flow.core import list_flows
@click.command()
def ls():
    """List running flows"""

