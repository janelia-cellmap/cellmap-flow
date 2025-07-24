# Simple Makefile for cellmap-flow project

.PHONY: test test-cov clean install install-dev

test:
	python -m pytest tests/ -v

test-cov:
	python tests/coverage_utils.py

clean:
	python tests/coverage_utils.py --clean
	rm -rf .pytest_cache __pycache__ htmlcov

install:
	pip install -e .

install-dev:
	pip install -e ".[dev]"
