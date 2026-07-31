import pytest

from cellmap_flow.utils.import_utils import (
    check_dependencies,
    get_missing_dependencies,
    MissingDependencyError,
    INSTALLABLE_PACKAGES,
)


def test_get_missing_dependencies_reports_only_missing():
    missing = get_missing_dependencies(["numpy", "definitely_not_a_real_package_xyz"])
    assert missing == ["definitely_not_a_real_package_xyz"]


def test_get_missing_dependencies_empty_when_all_present():
    assert get_missing_dependencies(["numpy", "os"]) == []


def test_check_dependencies_raises_missing_dependency_error():
    with pytest.raises(MissingDependencyError) as exc_info:
        check_dependencies(["definitely_not_a_real_package_xyz"])
    assert exc_info.value.missing_packages == ["definitely_not_a_real_package_xyz"]


def test_check_dependencies_passes_when_installed():
    check_dependencies(["numpy"])  # should not raise


def test_installable_packages_allow_list_only_contains_known_keys():
    # Every entry should be a plain (import_name -> pip spec) mapping usable
    # by the /api/dependencies/install allow-list check.
    for import_name, pip_spec in INSTALLABLE_PACKAGES.items():
        assert isinstance(import_name, str) and import_name
        assert isinstance(pip_spec, str) and pip_spec
