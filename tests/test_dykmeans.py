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
from difflib import SequenceMatcher

def _compare(a, b):
    return SequenceMatcher(None, a, b).ratio()


def _encode(a):
    return a[:1]


def test_simawareindex_insert():
    dy_kmeans = DyKMeans(_encode, _compare, threshold=1., top_n=3,
                         gold_standard="dataset1_gold.csv",
                         gold_attributes=["org_id", "dup_id"])
    dy_kmeans.fit("dataset1.csv", ["rec_id", "given_name", "suburb", "surname", "postcode"])

    # draw_frequency_distribution({"t": dy_kmeans.frequency_distribution()}, "", "")
    dy_kmeans.insert(["rec-499-org", "oliver", "yep", "club terrace", "7052"])
    dy_kmeans.insert(["rec-499-dup", "olive", "yeb", "club terraco", "7056"])
    res = dy_kmeans.query(["rec-499-dup0", "olivi", "yep", "club terrace", "7056"])
    print(res)
    print("Recall:", dy_kmeans.recall())
    # show()
