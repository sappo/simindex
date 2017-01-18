#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
test_simindex
----------------------------------

Tests for `simindex` module.
"""

# import pytest
# from .testprofiler import profile
# from .testdata import restaurant_records
# from .testhelper import has_common_token

from simindex import SimEngine

engine = SimEngine()
engine.fit_csv("restaurant.csv")
