# Test Coverage Tracking Guide

This document explains how to use the test coverage tracking system for cellmap-flow.

## Quick Start

### Run Tests with Coverage
```bash
# Run all tests with coverage
make test-cov

# View coverage summary in terminal
make coverage-show

# Generate HTML coverage report
make coverage-report
```

### View Coverage Reports
```bash
# Open HTML coverage report
open htmlcov/index.html

# Show coverage summary in terminal
make coverage-show
```

## Coverage Tools Overview

### 1. pytest-cov Integration
- **Command**: `pytest --cov=cellmap_flow`
- **Output**: Terminal summary, HTML reports, XML for CI/CD
- **Configuration**: Defined in `.coveragerc`

### 2. Coverage Utilities
- **Location**: `tests/coverage_utils.py`
- **Features**: Simple coverage test runner with cleanup

### 3. CI/CD Integration
- **File**: `.github/workflows/test-coverage.yml`
- **Features**: Automated testing, coverage reporting

## Basic Coverage Commands

| Command | Description |
|---------|-------------|
| `make test-cov` | Run tests with HTML and terminal coverage reports |
| `make coverage-show` | Display coverage summary with missing lines |
| `make coverage-report` | Generate HTML coverage report |
| `make test-cov-clean` | Clean coverage files |

## Usage

### For Coverage Testing
```bash
# Run tests with coverage
make test-cov

# Clean coverage files if needed
make test-cov-clean

# View detailed coverage report
make coverage-show
```

### For Development
```bash
# Setup development environment
make install-dev

# Run specific tests with coverage
python tests/coverage_utils.py tests/test_specific.py
```

## Coverage Configuration

### Minimum Coverage Threshold
- **Current**: 70%
- **Location**: `.coveragerc`
- **Enforcement**: CI/CD pipeline

### Coverage Exclusions
The following are excluded from coverage calculations:
- Test files (`tests/`)
- Migration scripts
- Debug and development utilities
- External interface stubs

### Branch Coverage
- **Enabled**: Yes
- **Purpose**: Ensure all code paths are tested
- **Command**: `--cov-branch` flag is used

## Understanding Coverage Reports

### HTML Report Structure
```
htmlcov/
├── index.html          # Overview with module summaries
├── cellmap_flow_*.html # Individual file reports
└── static/             # CSS and JavaScript assets
```

### Coverage Metrics
- **Line Coverage**: Percentage of executable lines tested
- **Branch Coverage**: Percentage of conditional branches tested
- **Function Coverage**: Percentage of functions called in tests

### Reading the Reports
- **Green**: Well-covered code
- **Red**: Uncovered code that needs tests
- **Yellow**: Partially covered branches

## Coverage Improvement Workflow

1. **Run Coverage**: `make test-cov`
2. **Review Report**: Open `htmlcov/index.html` 
3. **Write Tests**: Focus on uncovered functions and methods
4. **Verify**: Run coverage again to confirm improvements

## Best Practices

- **Aim for 80%+ coverage** in new modules
- **Focus on critical paths** first  
- **Test error conditions** and edge cases
- **Use appropriate mocks** for external dependencies

## Troubleshooting

### Coverage Data Not Found
**Solution**: Run tests with coverage first:
```bash
make test-cov
```

### Coverage File Conflicts
**Solution**: Clean coverage files:
```bash
make test-cov-clean
```
