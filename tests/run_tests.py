#!/usr/bin/env python3
"""
Test runner script for cellmap-flow unit tests with comprehensive coverage tracking.

This script demonstrates how to run the comprehensive test suite with detailed
coverage reporting and analysis.
"""

import sys
import subprocess
import os
import webbrowser
from pathlib import Path


def run_tests():
    """Run the test suite using pytest."""

    # Change to the project root directory
    project_root = os.path.dirname(os.path.abspath(__file__))
    os.chdir(project_root)

    print("Running cellmap-flow unit tests...")
    print(f"Project root: {project_root}")
    print("-" * 50)

    # Basic test run
    cmd = [
        sys.executable,
        "-m",
        "pytest",
        "tests/",
        "-v",  # verbose output
        "--tb=short",  # shorter traceback format
        "--durations=10",  # show 10 slowest tests
    ]

    try:
        result = subprocess.run(cmd, cwd=project_root, check=False)
        return result.returncode
    except Exception as e:
        print(f"Error running tests: {e}")
        return 1


def run_tests_with_coverage():
    """Run tests with comprehensive coverage reporting."""

    project_root = os.path.dirname(os.path.abspath(__file__))
    os.chdir(project_root)

    print("Running tests with comprehensive coverage...")
    print("-" * 50)

    # Clean previous coverage data
    cleanup_coverage()

    cmd = [
        sys.executable,
        "-m",
        "pytest",
        "tests/",
        "--cov=cellmap_flow",
        "--cov-report=html:htmlcov",
        "--cov-report=term-missing",
        "--cov-report=xml:coverage.xml",
        "--cov-report=json:coverage.json",
        "--cov-branch",  # Enable branch coverage
        "--cov-fail-under=70",  # Fail if coverage below 70%
        "-v",
    ]

    try:
        result = subprocess.run(cmd, check=False)
        if result.returncode == 0:
            print("\n" + "=" * 50)
            print("Coverage reports generated:")
            print("  - HTML: htmlcov/index.html")
            print("  - XML: coverage.xml")
            print("  - JSON: coverage.json")
            print("=" * 50)

            # Optionally open HTML coverage report
            html_report = Path("htmlcov/index.html")
            if html_report.exists():
                try:
                    response = input("\nOpen HTML coverage report in browser? (y/N): ")
                    if response.lower() in ["y", "yes"]:
                        webbrowser.open(f"file://{html_report.absolute()}")
                except KeyboardInterrupt:
                    pass
        return result.returncode
    except Exception as e:
        print(f"Error running tests with coverage: {e}")
        return 1


def run_coverage_analysis():
    """Run detailed coverage analysis and reporting."""

    project_root = os.path.dirname(os.path.abspath(__file__))
    os.chdir(project_root)

    print("Running detailed coverage analysis...")
    print("-" * 50)

    # First run tests with coverage
    test_result = run_tests_with_coverage()
    if test_result != 0:
        print("Tests failed, coverage analysis incomplete")
        return test_result

    # Generate coverage reports in multiple formats
    reports = [
        (["coverage", "report", "--show-missing"], "Terminal coverage report"),
        (["coverage", "html"], "HTML coverage report"),
        (["coverage", "xml"], "XML coverage report"),
        (["coverage", "json"], "JSON coverage report"),
    ]

    for cmd, description in reports:
        print(f"\nGenerating {description}...")
        try:
            subprocess.run(cmd, check=True)
        except subprocess.CalledProcessError as e:
            print(f"Warning: Failed to generate {description}: {e}")
        except FileNotFoundError:
            print(f"Warning: Coverage tool not found for {description}")

    # Show coverage summary
    print("\n" + "=" * 60)
    print("COVERAGE SUMMARY")
    print("=" * 60)

    try:
        subprocess.run(["coverage", "report", "--precision=2"], check=True)
    except (subprocess.CalledProcessError, FileNotFoundError):
        print("Could not generate coverage summary")

    return 0


def cleanup_coverage():
    """Clean up previous coverage data."""

    files_to_remove = [
        ".coverage",
        "coverage.xml",
        "coverage.json",
    ]

    dirs_to_remove = [
        "htmlcov",
        ".pytest_cache",
    ]

    for file_path in files_to_remove:
        try:
            os.remove(file_path)
        except FileNotFoundError:
            pass

    for dir_path in dirs_to_remove:
        try:
            import shutil

            shutil.rmtree(dir_path)
        except FileNotFoundError:
            pass


def run_specific_tests():
    """Run specific test categories with coverage."""

    test_categories = {
        "globals": "tests/test_globals.py",
        "processing": "tests/test_processing.py",
        "data_utils": "tests/test_data_utils.py",
        "normalization": "tests/test_norm/",
    }

    print("Available test categories:")
    for i, (name, path) in enumerate(test_categories.items(), 1):
        print(f"  {i}. {name} ({path})")

    try:
        choice = input("\nEnter category number (or 'all' for all tests): ")

        if choice.lower() == "all":
            return run_tests_with_coverage()

        choice_num = int(choice) - 1
        if 0 <= choice_num < len(test_categories):
            category_name, test_path = list(test_categories.items())[choice_num]
            print(f"\nRunning {category_name} tests with coverage...")

            cmd = [
                sys.executable,
                "-m",
                "pytest",
                test_path,
                "--cov=cellmap_flow",
                "--cov-report=term-missing",
                "--cov-report=html:htmlcov",
                "-v",
            ]
            result = subprocess.run(cmd, check=False)
            return result.returncode
        else:
            print("Invalid choice")
            return 1

    except (ValueError, KeyboardInterrupt):
        print("\nCancelled")
        return 1


def run_parallel_tests():
    """Run tests in parallel for faster execution."""

    project_root = os.path.dirname(os.path.abspath(__file__))
    os.chdir(project_root)

    print("Running tests in parallel with coverage...")
    print("-" * 50)

    cmd = [
        sys.executable,
        "-m",
        "pytest",
        "tests/",
        "--cov=cellmap_flow",
        "--cov-report=html:htmlcov",
        "--cov-report=term-missing",
        "-n",
        "auto",  # Use all available CPUs
        "-v",
    ]

    try:
        result = subprocess.run(cmd, check=False)
        return result.returncode
    except Exception as e:
        print(f"Error running parallel tests: {e}")
        return 1


def main():
    """Main entry point."""

    if len(sys.argv) > 1:
        if sys.argv[1] == "--coverage":
            return run_tests_with_coverage()
        elif sys.argv[1] == "--analysis":
            return run_coverage_analysis()
        elif sys.argv[1] == "--specific":
            return run_specific_tests()
        elif sys.argv[1] == "--parallel":
            return run_parallel_tests()
        elif sys.argv[1] == "--cleanup":
            cleanup_coverage()
            print("Coverage data cleaned up")
            return 0
        elif sys.argv[1] == "--help":
            print("Usage:")
            print("  python run_tests.py              # Run all tests")
            print("  python run_tests.py --coverage   # Run with coverage")
            print("  python run_tests.py --analysis   # Detailed coverage analysis")
            print("  python run_tests.py --specific   # Choose specific tests")
            print("  python run_tests.py --parallel   # Run tests in parallel")
            print("  python run_tests.py --cleanup    # Clean coverage data")
            print("  python run_tests.py --help       # Show this help")
            return 0

    return run_tests()


if __name__ == "__main__":
    sys.exit(main())
