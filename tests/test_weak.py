#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
test_weak
----------------------------------

Tests for `weak` module.
"""

import pytest
from .testprofiler import profile
from .testdata import restaurant_records as records
from .testhelper import has_common_token

from pprint import pprint

from simindex.weak_labels import WeakLabels, SimTupel, \
                                 Feature, DisjunctiveBlockingScheme, \
                                 BlockingKey, stoplist


def test_weak_labels():
    labels = WeakLabels(max_positive_pairs=4, max_negative_pairs=4)
    labels.fit(records)
    P_actual, N_actual = labels.predict()
    P_expected = [SimTupel(t1='0', t2='1', sim=0.),
                  SimTupel(t1='2', t2='3', sim=0.),
                  SimTupel(t1='4', t2='5', sim=0.),
                  SimTupel(t1='6', t2='7', sim=0.)]
    N_expected = [SimTupel(t1='8', t2='9', sim=0.),
                  SimTupel(t1='10', t2='11', sim=0.),
                  SimTupel(t1='12', t2='13', sim=0.),
                  SimTupel(t1='14', t2='15', sim=0.)]

    # Similarity robust tupel assertion
    for actual_pair in P_actual:
        hit = False
        for expected_pair in P_expected:
            if actual_pair.t1 == expected_pair.t1 and \
               actual_pair.t2 == expected_pair.t2:
                hit = True
        assert(hit)

    for actual_pair in N_actual:
        hit = False
        for expected_pair in N_expected:
            if actual_pair.t1 == expected_pair.t1 and \
               actual_pair.t2 == expected_pair.t2:
                hit = True
        assert(hit)

    blocking_keys=[]
    blocking_keys.append(BlockingKey(has_common_token, 0, str.split))
    blocking_keys.append(BlockingKey(has_common_token, 1, str.split))

    dnfblock = DisjunctiveBlockingScheme(blocking_keys, labels)
    dnf = dnfblock.transform()
    dnf_expected = [Feature([BlockingKey(has_common_token, 0, None),
                             BlockingKey(has_common_token, 1, None)], 0., 0.),
                    Feature([BlockingKey(has_common_token, 0, None)], 0., 0.),
                    Feature([BlockingKey(has_common_token, 1, None)], 0., 0.)]
    assert(dnf == dnf_expected)


def test_filter_labels():
    labels = WeakLabels(max_positive_pairs=4, max_negative_pairs=4)
    labels.fit(records)
    P, N = labels.predict()

    blocking_keys=[]
    blocking_keys.append(BlockingKey(has_common_token, 0, str.split))
    blocking_keys.append(BlockingKey(has_common_token, 1, str.split))

    dnfblock = DisjunctiveBlockingScheme(blocking_keys, labels)
    dnf = dnfblock.transform()

    # Only use first combined blocking key which filters half positives and all
    # negatives.
    fP, fN = dnfblock.filter_labels(dnf[:1])

    fP_expected = [SimTupel(t1='0', t2='1', sim=0.),
                   SimTupel(t1='4', t2='5', sim=0.)]
    # Similarity robust tupel assertion
    for actual_pair in fP:
        hit = False
        for expected_pair in fP_expected:
            if actual_pair.t1 == expected_pair.t1 and \
               actual_pair.t2 == expected_pair.t2:
                hit = True
        assert(hit)

    assert fN == []


def test_fisher_score():
    P = [
            [1, 0, 1, 1],
            [1, 1, 1, 0],
            [1, 0, 1, 0]
        ]
    N = [
            [1, 0, 1, 0],
            [0, 1, 0, 1],
            [0, 0, 0, 0]
        ]
    fc1 = DisjunctiveBlockingScheme.fisher_score(P, N, 0)
    fc2 = DisjunctiveBlockingScheme.fisher_score(P, N, 1)
    fc12 = DisjunctiveBlockingScheme.fisher_score(P, N, 2)
    assert (fc1 == fc2)
    assert (fc1 < fc12)


def test_my_score():
    P = [
            [1, 1, 0, 1, 1],
            [1, 1, 1, 1, 0],
            [1, 1, 0, 1, 0]
        ]
    N = [
            [1, 0, 1, 0, 1, 0],
            [0, 1, 0, 1, 0, 1],
            [0, 0, 0, 0, 0, 0]
        ]
    mc1 = DisjunctiveBlockingScheme.my_score(P, N, 0)
    mc2 = DisjunctiveBlockingScheme.my_score(P, N, 1)
    mc12 = DisjunctiveBlockingScheme.my_score(P, N, 2)
    assert (mc1 == mc2)
    assert (mc1 < mc12)


def test_tfidf_similarity():
    labels = WeakLabels(max_positive_pairs=4, max_negative_pairs=4)
    labels.fit(records)
    sim01 = labels.tfidf_similarity("0", "1")
    sim23 = labels.tfidf_similarity("2", "3")
    sim45 = labels.tfidf_similarity("4", "5")
    sim67 = labels.tfidf_similarity("6", "7")
    assert(round(sim01, 2) == 0.57)
    assert(round(sim23, 2) == 0.57)
    assert(round(sim45, 2) == 0.72)
    assert(round(sim67, 2) == 0.66)


@profile(follow=[WeakLabels.predict,
                 WeakLabels.tfidf_similarity,
                 DisjunctiveBlockingScheme.terms])
def test_profile_restaurant_dnf():
    labels = WeakLabels(max_positive_pairs=50, max_negative_pairs=200)
    labels.fit_csv("restaurant.csv")

    blocking_keys=[]
    blocking_keys.append(BlockingKey(has_common_token, 0, str.split))
    blocking_keys.append(BlockingKey(has_common_token, 1, str.split))
    blocking_keys.append(BlockingKey(has_common_token, 2, str.split))
    blocking_keys.append(BlockingKey(has_common_token, 3, str.split))
    blocking_keys.append(BlockingKey(has_common_token, 4, str.split))
    blocking_keys.append(BlockingKey(has_common_token, 5, str.split))

    dnfblock = DisjunctiveBlockingScheme(blocking_keys, labels)
    dnfblock.transform()
