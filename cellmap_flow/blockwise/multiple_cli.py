"""``python -m cellmap_flow.blockwise.multiple_cli A.yaml B.yaml``, which the
dashboard's blockwise tab runs: ``cellmap_flow blockwise`` (blockwise/cli.py),
which takes several YAMLs since 0.3.0. Delete this module once the tab runs
that instead.
"""

from cellmap_flow.blockwise.cli import cli

if __name__ == "__main__":
    cli()
