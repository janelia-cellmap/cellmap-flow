"""``cellmap_flow add REF``: the model entry for a path, URL, repo or model name.

``models.resolve`` says what REF is; this prints its entry as YAML, ready to
paste under a config's ``models:``, with what it was taken to be, where it
runs and what it still needs as comments. ``--run -d DATA`` serves it right
away, as ``cellmap_flow yaml`` serves a config's models.

Everything past click is imported when the command runs, so ``--help`` is
as fast as the group's.
"""

import click

from cellmap_flow.cli.common import resample_option


def entry_yaml(resolved) -> str:
    """``resolved``'s entry as a ``models:`` mapping, after comments saying what it is.

    Its lines under ``models:`` are indented as they go in a config, so they
    paste under an existing ``models:`` as they are.
    """
    import yaml

    class Dumper(yaml.SafeDumper):
        """Mappings as blocks, a list of numbers or names on one line: [8, 8, 8]."""

    def flow_list(dumper, value):
        flat = all(not isinstance(v, (list, dict)) for v in value)
        return dumper.represent_sequence("tag:yaml.org,2002:seq", value, flow_style=flat)

    Dumper.add_representer(list, flow_list)
    params = {k: v for k, v in resolved.entry().items() if k != "name"}
    body = yaml.dump({"models": {resolved.name: params}}, Dumper=Dumper, sort_keys=False)
    where = f"the {resolved.env} environment (its type's default)" if resolved.env else "this environment"
    comments = [f"{resolved.type}: {resolved.how}", f"runs in {where}"]
    if resolved.needs:
        comments.append(f"still needs: {', '.join(resolved.needs)} (add to the entry)")
    comments += [f"note: {note}" for note in resolved.notes]
    return "".join(f"# {line}\n" for line in comments) + body


@click.command(name="add")
@click.argument("ref")
@click.option("-n", "--name", default=None, help="The model's name (default: one made from REF).")
@click.option("-v", "--voxel-size", default=None,
              help="nm per voxel, '8' or '16,8,8', for a model that does not say its own (Cellpose, bioimage.io).")
@click.option("--offline", is_flag=True,
              help="Do not look REF up on Hugging Face or in the BioImage Model Zoo.")
@click.option("--run", "run_now", is_flag=True,
              help="Serve the model on --data-path right away and open the viewer, as `cellmap_flow yaml` does.")
@click.option("-d", "--data-path", default=None, help="The dataset --run serves it on.")
@click.option("-q", "--queue", default=None, help="Queue for --run's job (default: the saved queue)")
@click.option("-P", "--project", default=None, help="Project/chargeback group for --run's job (default: the saved one)")
@resample_option()
def add(ref, name, voxel_size, offline, run_now, data_path, queue, project, resample):
    """Print the model entry for REF, as YAML for a config's `models:`.

    REF is what you have: a path (a .py script, a cellmap-models export
    folder, a fly checkpoint, a bioimage.io rdf.yaml or zip, a LoRA
    adapter), a Hugging Face repo (org/repo), a BioImage Model Zoo nickname,
    a URL, or a Cellpose model name. A prefix says outright what it is:
    hf:, bioimageio:, cellpose:, fly:, dacapo:RUN@ITERATION, script:.

    \b
      cellmap_flow add cellmap/fly_organelles_run07_432000
      cellmap_flow add cpsam -v 16,8,8
      cellmap_flow add /path/to/run/model_checkpoint_432000 --run -d /path/to/data.zarr
    """
    from cellmap_flow.models.resolve import resolve

    if run_now and not data_path:
        raise click.UsageError("--run needs --data-path (-d), the dataset to serve the model on")
    try:
        resolved = resolve(ref, name=name, voxel_size=voxel_size, online=not offline)
    except ValueError as e:
        raise click.ClickException(str(e))
    click.echo(entry_yaml(resolved), nl=False)
    if run_now:
        _run(resolved, data_path, queue, project, resample)


def _run(resolved, data_path, queue, project, resample):
    """Serve ``resolved``'s model on ``data_path``: ``cellmap_flow yaml``'s run, for one model."""
    from cellmap_flow.cli.yaml_cli import run_multiple
    from cellmap_flow.config.yaml import ConfigError
    from cellmap_flow.dashboard.state import get_session
    from cellmap_flow.jobs.launch import install_cleanup_handlers
    from cellmap_flow.jobs.settings import launcher_settings
    from cellmap_flow.jobs.spec import JobStartError
    from cellmap_flow.models.registry import build_model

    if resolved.needs:
        raise click.ClickException(
            f"{resolved.name} still needs {', '.join(resolved.needs)}: add it to the entry above and "
            "serve it with `cellmap_flow yaml`"
        )
    try:
        model = build_model(resolved.entry(), resolved.name)
    except ConfigError as e:
        raise click.ClickException(str(e))

    settings = launcher_settings()
    queue = queue or settings.queue
    project = project or settings.charge_group
    settings.queue = queue
    if project:
        settings.charge_group = project
    settings.save()

    session = get_session()
    session.models_config = [model]
    # Also for the models launched later from the dashboard, as the YAML's `resample`.
    session.resample = resample
    install_cleanup_handlers()
    try:
        run_multiple([model], data_path, project, queue, resample=resample)
    except JobStartError as e:
        raise click.ClickException(str(e))
