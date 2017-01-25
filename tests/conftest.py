# -*- coding: utf-8 -*-
import pytest


def pytest_addoption(parser):
    parser.addoption("--profile", action="store_true",
                     help="Run marked tests with line-profiler")
    parser.addoption("--class-verbose", action="store_true", default=False,
                     help="Print verbose information")

@pytest.fixture
def verbose(request):
    return request.config.getoption("--class-verbose")
