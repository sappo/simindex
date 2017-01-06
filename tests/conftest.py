# -*- coding: utf-8 -*-
import pytest


def pytest_addoption(parser):
    parser.addoption("--profile", action="store_true",
                     help="Run marked tests with line-profiler")
