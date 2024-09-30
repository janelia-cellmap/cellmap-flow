import click
from cellmap_flow.commands.start import start
from cellmap_flow.commands.ls import ls
from cellmap_flow.commands.kill import kill
from cellmap_flow.commands.get import get


@click.group()
def cli():
    """Cellmap Flow CLI"""
    pass


cli.add_command(start)
cli.add_command(ls)
cli.add_command(kill)
cli.add_command(get)

if __name__ == "__main__":
    cli()
