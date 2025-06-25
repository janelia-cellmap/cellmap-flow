import pytest
from cellmap_flow.globals import Flow


def test_singleton_instance():
    f1 = Flow()
    f2 = Flow()
    assert f1 is f2, "Flow should implement the singleton pattern"
