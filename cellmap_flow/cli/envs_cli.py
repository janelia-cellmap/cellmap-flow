"""``cellmap_flow envs``: the environments models run in (``models.envs``).

\b
  envs [list]         each environment: where it is, whether it is
                      installed and can finetune, which types default to it
  envs install NAME   install a pixi environment now, not on its first job
  envs check NAME     import cellmap_flow in it, to prove it works

Only ``list`` imports the model types (numpy, funlib), to read their
defaults; ``--help`` imports none of them.
"""

import shlex
import subprocess
import sys

import click

from cellmap_flow.config.yaml import ConfigError
from cellmap_flow.models import envs

_CHECK_CODE = "import cellmap_flow; print(cellmap_flow.__file__)"


def _first_sentence(doc) -> str:
    """A docstring's first sentence on one line, without RST's double backquotes."""
    paragraph = " ".join((doc or "").strip().split("\n\n")[0].split())
    return paragraph.split(". ")[0].rstrip(".").replace("``", "")


def _type_defaults():
    """{environment: [the types defaulting to it]}, and [(type, how)] for the
    types that decide per model, which no list can say without a model."""
    from cellmap_flow.models import registry

    by_env, per_model = {}, []
    for name, cls in sorted(registry.model_types().items()):
        declared = envs.declared_default(cls)
        if envs.decides_per_model(declared):
            per_model.append((name, _first_sentence(getattr(declared, "__doc__", None)) or "decided per model"))
        elif declared:
            by_env.setdefault(declared, []).append(name)
    return by_env, per_model


def _row(name, pixi_names, alias_paths, defaulted_by):
    """NAME, KIND, INSTALLED, FINETUNE, DEFAULT FOR, WHERE for one environment."""
    default_for = ", ".join(defaulted_by.get(name, [])) or "-"
    if name in alias_paths:
        installed = "yes" if envs.is_installed(name) else "broken"
        # finetune_problem does not look inside a directory.
        return [name, "alias", installed, "unchecked", default_for, alias_paths[name]]
    if name in pixi_names:
        finetune = "no" if envs.finetune_problem(name) else "yes"
        installed = "yes" if envs.is_installed(name) else "no"
        return [name, "pixi", installed, finetune, default_for, envs.prefix(name)]
    return [name, "missing", "-", "-", default_for, "-"]


@click.group(name="envs", invoke_without_command=True)
@click.pass_context
def envs_group(ctx):
    """The environments models run in, besides this one.

    A model runs in its entry's `env`, else its type's default, else this
    environment; `env: current` keeps it here. An environment is one of
    cellmap-flow's pixi.toml, or an alias: a name mapped to a conda env or
    virtualenv in ~/.cellmap_flow/envs.yaml (CELLMAP_FLOW_ENVS_FILE).
    Without a subcommand, `list`.
    """
    if ctx.invoked_subcommand is None:
        ctx.invoke(list_cmd)


@envs_group.command(name="list")
def list_cmd():
    """List each environment, and the model types that default to it."""
    manifest = envs.pixi_manifest()
    alias_paths = _aliases()
    pixi_names = envs.pixi_environments(manifest) if manifest.is_file() else {}
    defaulted_by, per_model = _type_defaults()

    click.echo(f"pixi.toml: {manifest}" + ("" if manifest.is_file() else " (not found)"))
    click.echo(f"pixi:      {envs.pixi_program() if envs.has_pixi() else 'not found'}")
    click.echo(f"aliases:   {envs.aliases_file()}" + ("" if alias_paths else " (none)"))
    click.echo()
    names = list(pixi_names) + [n for n in alias_paths if n not in pixi_names]
    names += [n for n in sorted(defaulted_by) if n not in names]
    rows = [["NAME", "KIND", "INSTALLED", "FINETUNE", "DEFAULT FOR", "WHERE"]]
    rows += [_row(name, pixi_names, alias_paths, defaulted_by) for name in names]
    widths = [max(len(row[i]) for row in rows) for i in range(len(rows[0]) - 1)]
    for row in rows:
        click.echo("  ".join(cell.ljust(width) for cell, width in zip(row, widths)) + "  " + row[-1])
    if per_model:
        click.echo("\nDecided per model:")
        for name, how in per_model:
            click.echo(f"  {name}: {how}")


def _aliases():
    try:
        return envs.aliases()
    except ConfigError as e:
        raise click.ClickException(str(e))


def _checked(name) -> str:
    try:
        return envs.validate(name, name)
    except ConfigError as e:
        # validate speaks of a model's env; here there is no model.
        raise click.ClickException(str(e).removeprefix(f"Model '{name}': "))


def _run(argv) -> int:
    """``argv``'s exit status, its output not captured, so it shows as it goes."""
    click.echo(shlex.join(argv))
    try:
        return subprocess.run(argv).returncode
    except FileNotFoundError as e:
        raise click.ClickException(f"cannot run {argv[0]}: {e.strerror}")


@envs_group.command(name="install")
@click.argument("name")
def install_cmd(name):
    """Install the pixi environment NAME, as its first job otherwise would.

    Installed from pixi.lock as it is (--frozen), as `pixi run --frozen`
    runs it, so the lock in the checkout is never rewritten.
    """
    alias = _aliases().get(name)
    if alias is not None:
        raise click.ClickException(
            f"{name} is an alias of {alias} (in {envs.aliases_file()}): there is nothing to install"
        )
    if envs.is_path(name):
        raise click.ClickException(f"{name} is a path: install it with conda or pip")
    _checked(name)
    if not envs.has_pixi():
        raise click.ClickException("pixi is not installed: see https://pixi.sh, or set PIXI_EXE")
    sys.exit(_run([envs.pixi_program(), "install", "--frozen", "--manifest-path", str(envs.pixi_manifest()),
                   "-e", name]))


@envs_group.command(name="check")
@click.argument("name")
def check_cmd(name):
    """Import cellmap_flow in the environment NAME (a name or a path)."""
    env = _checked(name)
    if env == envs.CURRENT:
        python = [sys.executable]
    else:
        python = envs.python(env)
        if not envs.is_installed(env):
            click.echo(f"{name} is not installed; pixi installs it first (several minutes).")
    # -P keeps the working directory off sys.path: run inside a checkout, the
    # check finds the environment's own cellmap_flow, not the checkout's.
    returncode = _run([*python, "-P", "-c", _CHECK_CODE])
    if returncode:
        raise click.ClickException(f"{name} cannot import cellmap_flow (exit status {returncode})")
    click.echo(f"{name}: ok")
