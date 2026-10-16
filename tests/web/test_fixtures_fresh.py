"""The parity fixtures in web/test/fixtures describe the Python code beside them.

web/test/fixtures/make_fixtures.py writes what the normalizers, the
postprocessors and PipelineSpec produce for fixed inputs, and the vitest
suite in web/test/parity checks the TypeScript ports against those files.
When the Python changes and the fixtures are not regenerated, the
TypeScript is being checked against old behaviour; this test catches that.
"""

import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "web" / "test" / "fixtures" / "make_fixtures.py"
SEGMENT_SCRIPT = ROOT / "web" / "test" / "fixtures" / "make_segment_fixtures.py"


def test_fixtures_are_fresh():
    # The current interpreter (the one with cellmap_flow's dependencies), with
    # this checkout first on its path so the fixtures describe this code.
    pythonpath = os.pathsep.join(p for p in (str(ROOT), os.environ.get("PYTHONPATH", "")) if p)
    result = subprocess.run(
        [sys.executable, str(SCRIPT), "--check"],
        cwd=ROOT,
        env={**os.environ, "PYTHONPATH": pythonpath},
        capture_output=True,
        text=True,
        timeout=600,
    )
    assert result.returncode == 0, (
        "web/test/fixtures is stale: the Python normalizers, postprocessors or "
        "PipelineSpec changed since the fixtures were generated. Regenerate with\n"
        "    python web/test/fixtures/make_fixtures.py\n"
        "(the pixi default environment's python) and commit the result with the change.\n\n"
        f"{result.stdout}{result.stderr}"
    )


def test_segment_fixtures_are_fresh():
    # The same for the segmentation steps' fixtures (label.json, merger.json),
    # which make_segment_fixtures.py writes from LabelPostprocessor and
    # SimpleBlockwiseMerger.
    pythonpath = os.pathsep.join(p for p in (str(ROOT), os.environ.get("PYTHONPATH", "")) if p)
    result = subprocess.run(
        [sys.executable, str(SEGMENT_SCRIPT), "--check"],
        cwd=ROOT,
        env={**os.environ, "PYTHONPATH": pythonpath},
        capture_output=True,
        text=True,
        timeout=600,
    )
    assert result.returncode == 0, (
        "web/test/fixtures/label.json or merger.json is stale: LabelPostprocessor, "
        "SimpleBlockwiseMerger or segment.py changed since they were generated. Regenerate with\n"
        "    python web/test/fixtures/make_segment_fixtures.py\n"
        "(the pixi default environment's python) and commit the result with the change.\n\n"
        f"{result.stdout}{result.stderr}"
    )


AI_SCRIPT = ROOT / "web" / "test" / "fixtures" / "make_ai_fixtures.py"


def test_ai_annotate_fixture_is_fresh():
    # The same for the AI-assisted annotation fixture (ai_annotate.json),
    # which make_ai_fixtures.py writes from cellmap_flow.ai_annotate
    # (geometry, pipeline, mask_extraction, prompts, organelles) and
    # finetune/session/fill.py.
    pythonpath = os.pathsep.join(p for p in (str(ROOT), os.environ.get("PYTHONPATH", "")) if p)
    result = subprocess.run(
        [sys.executable, str(AI_SCRIPT), "--check"],
        cwd=ROOT,
        env={**os.environ, "PYTHONPATH": pythonpath},
        capture_output=True,
        text=True,
        timeout=600,
    )
    assert result.returncode == 0, (
        "web/test/fixtures/ai_annotate.json is stale: cellmap_flow.ai_annotate's geometry, pipeline, "
        "mask_extraction, prompts or organelles, or finetune/session/fill.py, changed since it was generated. Regenerate with\n"
        "    python web/test/fixtures/make_ai_fixtures.py\n"
        "(the pixi default environment's python) and commit the result with the change.\n\n"
        f"{result.stdout}{result.stderr}"
    )
