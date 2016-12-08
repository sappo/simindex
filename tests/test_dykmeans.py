#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
test_simindex
----------------------------------

Tests for `simindex` module.
"""

import pytest

from simindex import draw_frequency_distribution, show
from simindex import DyKMeans, DyLSH
from difflib import SequenceMatcher

def _compare(a, b):
    return SequenceMatcher(None, a, b).ratio()


def _encode(a):
    return a[:1]


# def test_simawareindex_insert():
    # dy_kmeans = DyKMeans(_encode, _compare, threshold=1., top_n=3,
                         # gold_standard="dataset1_gold.csv",
                         # gold_attributes=["org_id", "dup_id"])
    # dy_kmeans.fit("dataset1.csv", ["rec_id", "given_name", "suburb", "surname", "postcode"])

    # # draw_frequency_distribution({"t": dy_kmeans.frequency_distribution()}, "", "")
    # dy_kmeans.insert(["rec-499-org", "oliver", "yep", "club terrace", "7056"])
    # dy_kmeans.insert(["rec-499-dup", "olive", "yeb", "club terrace", "7054"])
    # res = dy_kmeans.query(["rec-499-dup0", "olivi", "yep", "club terrace", "7056"])
    # print(res)
    # print("Recall:", dy_kmeans.recall())
    # # show()


def test_lsh():
    n_thresholds = [0.05, 0.08, 0.09, 0.1, 0.2, 0.3]
    n_perms = range(2, 240)
    fp = open("eval_lsh.txt", mode='w')
    for threshold in n_thresholds:
        for perm in n_perms:
            dy_lsh = DyLSH(_encode, _compare, threshold=1., top_n=3,
                        gold_standard="restaurant_gold.csv",
                        gold_attributes=["id_1", "id_2"],
                        lsh_threshold=threshold, lsh_num_perm=perm)
            dy_lsh.fit("restaurant.csv", ["id", "name", "addr", "city", "phone"])
            fp.write("%f, %d, %f\n" % (threshold, perm, dy_lsh.recall()))
            fp.flush()

    fp.close()

    # draw_frequency_distribution({"t": dy_kmeans.frequency_distribution()}, "", "")
    # dy_kmeans.insert(["rec-499-org", "oliver", "yep", "club terrace", "7056"])
    # dy_kmeans.insert(["rec-499-dup", "olive", "yeb", "club terrace", "7054"])
    # res = dy_kmeans.query(["rec-499-dup0", "olivi", "yep", "club terrace", "7056"])
    # print(res)
    # print("Recall:", dy_kmeans.recall())
    # show()
