"""``cellmap_flow``: the one command, with a subcommand for each job.

\b
  infer <type>   start a model's inference server and open the viewer on it
  yaml           the same for the models a YAML file lists
  view           open a dataset in the viewer; pick models in the dashboard
  dashboard      serve the dashboard alone, for a viewer already running
  blockwise      run models over a whole volume, writing predictions to disk
  finetune       the finetune tools: train, export-merged, build-corrections
  models         list the model types and their arguments
  plugins        register, unregister and list plugins
  doctor         check the environment

Before 0.3.0 these were separate console scripts (``cellmap_flow_yaml`` and
the rest). Those names still work for one release: ``cli/aliases.py`` says
where each went and runs it. So do this command's own earlier subcommands,
hidden from ``--help``: ``cellmap_flow <type>``, ``list-models``,
``register``, ``unregister`` and ``list-plugins``.
"""

import sys

import click

from cellmap_flow.blockwise.cli import cli as blockwise
from cellmap_flow.cli import doctor, viewer_cli, yaml_cli
from cellmap_flow.cli.common import deprecated, log_level_option
from cellmap_flow.cli.infer import infer, run_generic
from cellmap_flow.models import registry
from cellmap_flow.plugins import list_plugins, load_plugins, register_plugin, unregister_plugin


class CellMapFlowGroup(click.Group):
    """The cellmap_flow group. ``cellmap_flow <type>``, its per-type commands
    before 0.3.0, are built on request as deprecated aliases of ``infer``."""

    def get_command(self, ctx, name):
        command = super().get_command(ctx, name)
        if command is None and name in registry.model_types():
            command = deprecated(
                infer.get_command(ctx, name), name, f"cellmap_flow {name}", f"cellmap_flow infer {name}"
            )
        return command


@click.group(cls=CellMapFlowGroup)
@log_level_option(default="INFO")
def cli():
    """CellMap Flow: real-time model inference on EM data, in Neuroglancer.

    Each job is a subcommand; `cellmap_flow <command> --help` shows its
    options.

    \b
      cellmap_flow view -d /path/to/data.zarr
      cellmap_flow yaml config.yaml
      cellmap_flow infer dacapo -r my_run -i 100 -d /path/to/data
      cellmap_flow blockwise config.yaml
    """


@click.command()
@click.option("-n", "--neuroglancer-url", default=None, help="The viewer the dashboard's page embeds.")
def dashboard(neuroglancer_url):
    """Serve the dashboard alone, for a viewer already running.

    `view`, `yaml` and `infer` start the dashboard with the viewer they
    open; this is the dashboard on its own, as `cellmap_flow_app` served it
    before 0.3.0. It prints its URL, and writes it to the file named by
    SERVICE_URL_PATH when that is set. Ctrl+C stops it and kills the models
    it launched.
    """
    # Imported here: the dashboard's routes pull in flask and the rest.
    from cellmap_flow.dashboard.app import create_and_run_app
    from cellmap_flow.jobs.launch import install_cleanup_handlers

    install_cleanup_handlers()
    create_and_run_app(neuroglancer_url=neuroglancer_url)


@click.command()
def models():
    """List the model types and the arguments each takes."""
    registry.print_available_models("cellmap_flow infer")


@click.group(name="plugins")
def plugins_group():
    """Register, unregister and list plugins.

    A plugin is a .py file defining subclasses of InputNormalizer,
    PostProcessor or ModelConfig; registering copies it to
    ~/.cellmap_flow/plugins/.
    """


@plugins_group.command(name="register")
@click.argument("filepath", type=click.Path(exists=True))
@click.option("--force", is_flag=True, help="Overwrite existing plugin with the same name.")
def register_cmd(filepath, force):
    """Register a custom plugin (normalizer, postprocessor, or model config).

    FILEPATH is the path to a .py file defining subclasses of
    InputNormalizer, PostProcessor, or ModelConfig.

    Example:
        cellmap_flow plugins register my_normalizer.py
    """
    try:
        dest = register_plugin(filepath, force=force)
        click.echo(f"Registered plugin: {dest.name}")
    except (FileNotFoundError, FileExistsError, ValueError) as exc:
        click.echo(f"Error: {exc}", err=True)
        sys.exit(1)


@plugins_group.command(name="unregister")
@click.argument("name")
def unregister_cmd(name):
    """Unregister a previously registered plugin by name.

    NAME is the plugin filename (with or without .py extension).

    Example:
        cellmap_flow plugins unregister my_normalizer
    """
    try:
        unregister_plugin(name)
        click.echo(f"Unregistered plugin: {name}")
    except FileNotFoundError as exc:
        click.echo(f"Error: {exc}", err=True)
        sys.exit(1)


@plugins_group.command(name="list")
def list_plugins_cmd():
    """List all registered plugins."""
    registered = list_plugins()
    if not registered:
        click.echo("No plugins registered.")
        return
    click.echo("Registered plugins:\n")
    for plugin_path in registered:
        click.echo(f"  {plugin_path.name}  ({plugin_path})")


def _run_argparse_main(main, prog, args):
    """Run an argparse ``main()`` as ``prog args``: argparse reads both from sys.argv."""
    saved = sys.argv
    sys.argv = [prog, *args]
    try:
        return main()
    finally:
        sys.argv = saved


def _finetune_tool(name, module, summary):
    """``cellmap_flow finetune <name>``: ``python -m <module>``, flags and all."""

    @click.command(
        name=name,
        help=f"{summary} Its flags are those of `python -m {module}`; --help lists them.",
        context_settings={"ignore_unknown_options": True, "allow_extra_args": True},
        add_help_option=False,
    )
    @click.argument("args", nargs=-1, type=click.UNPROCESSED)
    @click.pass_context
    def tool(ctx, args):
        import importlib

        sys.exit(_run_argparse_main(importlib.import_module(module).main, ctx.command_path, args))

    return tool


@click.group()
def finetune():
    """The finetune tools. The dashboard's finetune tab runs `train` for you."""


for _name, _module, _summary in (
    ("train", "cellmap_flow.finetune.finetune_cli",
     "Finetune a model on corrections painted in the dashboard, as its finetune jobs do."),
    ("export-merged", "cellmap_flow.finetune.export_merged",
     "Fold a finetune into the base model's weights and re-export it at a larger tile."),
    ("build-corrections", "cellmap_flow.finetune.build_corrections",
     "Build a finetuning corrections directory from a crops manifest, headlessly."),
):
    finetune.add_command(_finetune_tool(_name, _module, _summary))


cli.add_command(infer)
cli.add_command(yaml_cli.main, name="yaml")
cli.add_command(viewer_cli.main, name="view")
cli.add_command(dashboard)
cli.add_command(blockwise, name="blockwise")
cli.add_command(finetune)
cli.add_command(models)
cli.add_command(plugins_group)
cli.add_command(doctor.main, name="doctor")

# The subcommands before 0.3.0, hidden, until the release after it.
cli.add_command(run_generic, name="run")
cli.add_command(deprecated(models, "list-models", "cellmap_flow list-models", "cellmap_flow models"))
cli.add_command(deprecated(register_cmd, "register", "cellmap_flow register", "cellmap_flow plugins register"))
cli.add_command(deprecated(unregister_cmd, "unregister", "cellmap_flow unregister", "cellmap_flow plugins unregister"))
cli.add_command(deprecated(list_plugins_cmd, "list-plugins", "cellmap_flow list-plugins", "cellmap_flow plugins list"))


def main(args=None):
    """The ``cellmap_flow`` console script: load the plugins, then run the
    command ``args`` (by default, the process's arguments) names."""
    load_plugins()
    cli.main(args=args, prog_name="cellmap_flow")


if __name__ == "__main__":
    main()
