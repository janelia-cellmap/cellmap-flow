#!/usr/bin/env python3
"""
Coverage utilities for cellmap-flow tests.

Simple utilities for running tests with coverage and cleaning up coverage files.
"""

import sys
import subprocess
from pathlib import Path


def clean_coverage_files():
    """Remove any existing coverage files to avoid conflicts."""
    coverage_files = [".coverage", "coverage.json", "coverage.xml"]

    for file in coverage_files:
        path = Path(file)
        if path.exists():
            path.unlink()

    # Remove any temporary coverage files
    for file in Path(".").glob(".coverage.*"):
        file.unlink()


def run_tests_with_coverage(test_path=None):
    """Run tests with coverage reporting."""
    clean_coverage_files()

    cmd = [
        sys.executable,
        "-m",
        "pytest",
        test_path or "tests/",
        "--cov=cellmap_flow",
        "--cov-config=.coveragerc",
        "--cov-report=term-missing",
        "--cov-report=html:htmlcov",
        "-v",
    ]

    return subprocess.run(cmd)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run tests with coverage")
    parser.add_argument("test_path", nargs="?", help="Specific test file or directory")
    parser.add_argument(
        "--clean", action="store_true", help="Just clean coverage files"
    )

    args = parser.parse_args()

    if args.clean:
        clean_coverage_files()
        sys.exit(0)

    result = run_tests_with_coverage(args.test_path)
    sys.exit(result.returncode)
