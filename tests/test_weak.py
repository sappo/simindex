#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
test_weak
----------------------------------

Tests for `weak` module.
"""

import pytest
from .testprofiler import profile
from simindex.weak_labels import WeakLabels, SimTupel, \
                                 Feature, DisjunctiveBlockingScheme, \
                                 BlockingPredicate, stoplist

records = [
    # P (0, 1), (2, 3), (4, 5), (6, 7)
    ["0", "Mario's Pizza", "Italian"],
    ["1", "Marios Pizza", "Italian"],
    ["2", "Fringal", "French Bistro"],
    ["3", "Fringale", "French Bistro"],
    ["4", "Yujean Kang's Gourmet Cuisine", "Asian"],
    ["5", "Yujean Kang's Best Cuisine", "Asian"],
    ["6", "Big Belly Burger", "American"],
    ["7", "Big Belly Burger", "German"],
    # N (8, 9), (10, 11), (12, 13), (14, 15)
    ["8", "Sally's Cafè and Internet", "Tex-Mex Cafe"],
    ["9", "Cafè Sunflower and More", "Health Food"],
    ["10", "Roosevelt Tamale Parlor", "Mexican"],
    ["11", "Wa-Ha-Ka Oaxaca Moaxo", "Mexican"],
    ["12", "Thailand Restaurant", "Thai"],
    ["13", "Andre's Petit Restaurant", "Spanish"],
    ["14", "Zubu", "Japanese"],
    ["15", "Nobu", "Japanese"],
]

# -> hasCommonToken:f1
# P (0, 1)
# P (2, 3)
# P (4, 5)
# P (6, 7)
# -> hasCommonToken:f2
# P (0, 1)
# P (2, 3)
# P (4, 5)
# P (6, 7)


def has_common_token(t1, t2):
    t1_tokens = set(word for word in t1 if word not in stoplist)
    t2_tokens = set(word for word in t2 if word not in stoplist)

    if len(t1_tokens.intersection(t2_tokens)) > 0:
        return 1
    else:
        return 0


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

    dnfblock = DisjunctiveBlockingScheme([has_common_token], labels)
    dnf = dnfblock.transform()
    dnf_expected = [Feature([BlockingPredicate(has_common_token, 0),
                             BlockingPredicate(has_common_token, 1)], 0., 0.),
                    Feature([BlockingPredicate(has_common_token, 0)], 0., 0.),
                    Feature([BlockingPredicate(has_common_token, 1)], 0., 0.)]
    assert(dnf == dnf_expected)


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
def test_profile_dnf():
    labels = WeakLabels(max_positive_pairs=4, max_negative_pairs=4)
    labels.fit(records)
    dnfblock = DisjunctiveBlockingScheme([has_common_token], labels)
    dnfblock.transform()
