#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
test_simindex
----------------------------------

Tests for `simindex` module.
"""

import pytest

from simindex import draw_frequency_distribution, show
from simindex import DyKMeans


def test_simawareindex_insert():
    dy_kmeans = DyKMeans("dataset1.csv", ["rec_id", "given_name", "suburb", "surname", "postcode"],
                         "dataset1_gold.csv", ["org_id", "dup_id"])

    # print(dy_kmeans.recall())
    # draw_frequency_distribution({"t": dy_kmeans.frequency_distribution()}, "", "")
    dy_kmeans.insert(["rec-499-org", "oliver", "yep", "club terrace", "7052"])
    # show()
    assert False
