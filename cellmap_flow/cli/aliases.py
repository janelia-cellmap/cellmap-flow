"""The console scripts before 0.3.0, each now a ``cellmap_flow`` subcommand.

Each one says, on stderr, which subcommand replaces it, and runs that with
the arguments it was given. They go in the release after 0.3.0, with their
``[project.scripts]`` entries.

\b
  cellmap_flow_yaml                cellmap_flow yaml
  cellmap_flow_view                cellmap_flow view
  cellmap_flow_blockwise           cellmap_flow blockwise
  cellmap_flow_blockwise_multiple  cellmap_flow blockwise (it takes several YAMLs)
  cellmap_flow_app                 cellmap_flow dashboard
"""

import sys

from cellmap_flow.cli.common import deprecation_notice


def _alias(old: str, subcommand: str):
    def run(args=None):
        deprecation_notice(old, f"cellmap_flow {subcommand}")
        from cellmap_flow.cli.main import cli

        cli.main(args=[subcommand, *(sys.argv[1:] if args is None else args)], prog_name="cellmap_flow")

    run.__doc__ = f"``{old}``: ``cellmap_flow {subcommand}``, after a deprecation notice."
    return run


yaml = _alias("cellmap_flow_yaml", "yaml")
view = _alias("cellmap_flow_view", "view")
blockwise = _alias("cellmap_flow_blockwise", "blockwise")
blockwise_multiple = _alias("cellmap_flow_blockwise_multiple", "blockwise")
app = _alias("cellmap_flow_app", "dashboard")
