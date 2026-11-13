"""What the ``cellmap_flow`` subcommands share.

- ``log_level_option``: ``--log-level``, which sets the level as soon as it
  is parsed.
- ``ModelTypeGroup``: a group with one subcommand per model type, built
  from the registry when asked for.
- ``deprecated``: a command under the name it had before 0.3.0, which says
  what replaces it and then runs it.
- ``resample_option``: ``--resample``, of ``infer <type>`` and ``serve``.
- ``dataset_path``: the dataset a person gave as ``-d``, checked.
"""

import logging
import os

import click

from cellmap_flow.logging_setup import configure_logging
from cellmap_flow.models import registry

LOG_LEVELS = ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]

# When the aliases below, and the console scripts in cli/aliases.py, go.
REMOVED_IN = "the release after 0.3.0"


def _set_log_level(ctx, param, value):
    if value is None and ctx.parent is None:
        value = "INFO"  # run on its own, with no group to have set it
    if value is not None:
        configure_logging(getattr(logging, value.upper()))
    return value


def log_level_option(default=None, help="Set the logging level"):
    """``--log-level``, applied when it is parsed rather than passed on.

    The ``cellmap_flow`` group's defaults to INFO. A subcommand that had its
    own before 0.3.0 (yaml, view, blockwise) keeps it, defaulting to None, so
    that ``cellmap_flow --log-level DEBUG yaml c.yaml`` is not reset to INFO
    by the subcommand's default. Run on its own (``python -m``, as the
    dashboard's blockwise tab runs blockwise), such a subcommand logs at
    INFO unless told otherwise.
    """
    return click.option(
        "--log-level",
        type=click.Choice(LOG_LEVELS, case_sensitive=False),
        default=default,
        expose_value=False,
        callback=_set_log_level,
        help=help,
    )


def resample_option():
    """``--resample/--no-resample``, on by default: read the dataset resampled
    to the model's input voxel size when it has no level at that size
    (``open_raw``'s ``resample=True``, ``cellmap_flow.io.raw``). With
    ``--no-resample`` the level chosen is read as if it were at the model's
    voxel size, with a warning."""
    return click.option(
        "--resample/--no-resample",
        default=True,
        show_default=True,
        help="When the dataset has no level at the model's input voxel size, "
        "resample a level to it, axis by axis (the default), or, with --no-resample, "
        "read the nearest level as if it were at that size.",
    )


def dataset_path(text: str) -> str:
    """The dataset ``text``, given as ``-d`` to ``view``, ``infer <type>`` or
    ``add --run``, as cellmap-flow reads it (``portable.pasted.pasted_path``:
    a path copied from Neuroglancer without the format it names).

    A local path that is not there is a usage error that says so, before
    anything starts: otherwise it surfaces as worldarray's list of the
    metadata files it did not find, at the bottom of a traceback, which reads
    as if the data were there but malformed. A URL is left to the reader.
    """
    from cellmap_flow.io.paths import normalize_path
    from cellmap_flow.portable.pasted import pasted_path

    path = pasted_path(text)
    if "://" not in path and not os.path.exists(normalize_path(path)):
        raise click.BadParameter(
            f"{path} does not exist. Give a zarr or N5 directory, or an http(s), s3 or gs URL "
            "(https://janelia-cellmap.github.io/cellmap-flow/data_paths.html).",
            param_hint="'-d'",
        )
    return path


class ModelTypeGroup(click.Group):
    """A group with a subcommand for each model type, besides its own.

    ``make_command(name, config_class)`` builds a type's command when click
    asks for it, from the registry as it is at that moment: a type a plugin
    defines has its command, and nothing is built at import.
    """

    def __init__(self, *args, make_command, **kwargs):
        super().__init__(*args, **kwargs)
        self._make_command = make_command

    def list_commands(self, ctx):
        # Sorted, as click lists the commands it was given.
        return sorted({*super().list_commands(ctx), *registry.model_types()})

    def get_command(self, ctx, name):
        command = super().get_command(ctx, name)
        if command is not None:
            return command
        config_class = registry.model_types().get(name)
        return self._make_command(name, config_class) if config_class else None


def deprecation_notice(old: str, new: str) -> None:
    """Tell the user, on stderr, that ``old`` is going and ``new`` replaces it."""
    click.echo(f"`{old}` is deprecated and goes in {REMOVED_IN}; use `{new}`.", err=True)


def deprecated(command: click.Command, old_name: str, old: str, new: str) -> click.Command:
    """``command`` as ``old_name``, its name before 0.3.0: hidden from --help,
    and saying that ``new`` replaces ``old`` each time before it runs."""

    def callback(*args, **kwargs):
        deprecation_notice(old, new)
        return command.callback(*args, **kwargs)

    return click.Command(
        old_name,
        params=command.params,
        callback=callback,
        help=f"Deprecated: use `{new}`.",
        hidden=True,
        context_settings=command.context_settings,
    )
