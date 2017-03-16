#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
test_weak
----------------------------------

Tests for `weak` module.
"""

import pytest
import math
from .testprofiler import profile
from .testdata import restaurant_dataset, restaurant_gold_pairs
from .testhelper import tokens

from pprint import pprint

from simindex.dysim import MDySimIII
from simindex.weak_labels import WeakLabels, SimTupel, \
                                 Feature, DisjunctiveBlockingScheme, \
                                 BlockingKey
import simindex.helper as hp


def test_weak_labels():
    labels = WeakLabels(2, max_positive_pairs=4, max_negative_pairs=4,
                        upper_threshold=0.5, lower_threshold=0.4999)
    labels.fit(restaurant_dataset)
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
    for expected_pair in P_expected:
        hit = False
        for actual_pair in P_actual:
            if expected_pair.t1 == actual_pair.t1 and \
               expected_pair.t2 == actual_pair.t2:
                hit = True
        # assert hit is True

    for expected_pair in N_expected:
        hit = False
        for actual_pair in N_actual:
            if expected_pair.t1 == actual_pair.t1 and \
               expected_pair.t2 == actual_pair.t2:
                hit = True
        # assert hit is True

    blocking_keys = []
    blocking_keys.append(BlockingKey(0, tokens))
    blocking_keys.append(BlockingKey(1, tokens))

    dnfblock = DisjunctiveBlockingScheme(blocking_keys, P_actual, N_actual, MDySimIII)
    dnf = dnfblock.transform(restaurant_dataset)
    dnf_expected = [Feature([BlockingKey(0, tokens),
                             BlockingKey(1, tokens)], 0.),
                    Feature([BlockingKey(0, tokens)], 0.),
                    Feature([BlockingKey(1, tokens)], 0.)]
    # assert(dnf == dnf_expected)


@profile
def test_filter_labels():
    labels = WeakLabels(2, max_positive_pairs=4, max_negative_pairs=4,
                        upper_threshold=0.5, lower_threshold=0.3)
    labels.fit(restaurant_dataset)
    P, N = labels.predict()

    blocking_keys = []
    blocking_keys.append(BlockingKey(0, tokens))
    blocking_keys.append(BlockingKey(1, tokens))

    dnfblock = DisjunctiveBlockingScheme(blocking_keys, P, N, MDySimIII)

    dnf = dnfblock.transform(restaurant_dataset)

    # Only use first combined blocking key which filters half positives and all
    # negatives.
    fP, fN = labels.filter(dnf[:1], restaurant_dataset, P, N)

    fP_expected = [SimTupel(t1='0', t2='1', sim=0.),
                   SimTupel(t1='4', t2='5', sim=0.)]
    # Similarity robust tupel assertion
    for actual_pair in fP:
        hit = False
        for expected_pair in fP_expected:
            if actual_pair.t1 == expected_pair.t1 and \
               actual_pair.t2 == expected_pair.t2:
                hit = True
        # assert(hit)

    # assert fN == []


def test_probability_distribution_choice_small():
    labels = WeakLabels(2, max_positive_pairs=4, max_negative_pairs=4,
                        gold_pairs=restaurant_gold_pairs,
                        upper_threshold=0.5, lower_threshold=0.3,
                        window_size=5)
    labels.fit(restaurant_dataset)
    P, N = labels.predict()

    for simtupel in P:
        assert (simtupel.t1, simtupel.t2) in restaurant_gold_pairs

    assert len(N) <= 4


def test_probability_distribution_choice_large():
    gold_csv = "../../master_thesis/datasets/restaurant/restaurant_train_gold.csv"
    gold_lines = hp.read_csv(gold_csv, ["id_1", "id_2"])
    gold_paris = []
    for line in gold_lines:
        gold_paris.append((line[0], line[1]))

    dataset = {}
    records = hp.read_csv("../../master_thesis/datasets/restaurant/restaurant_train.csv",
                          ["id","name","addr","city","phone","type"])
    for record in records:
        dataset[record[0]] = record[1:]

    labels = WeakLabels(5, gold_pairs=gold_paris, window_size=5)
    labels.fit(dataset)
    P, N = labels.predict()

    assert len(P) == 38

    bins = 20
    N_bins = [[] for x in range(bins)]
    for simtupel in N:
        N_bins[int(simtupel.sim * bins)].append(simtupel)

    actual_weights = [len(bin) for bin in N_bins]
    wsum = sum(actual_weights)
    actual_weights[:] = [float(weight)/wsum for weight in actual_weights]

    expected_weights = [0.19546968687541638, 0.33111259160559625, 0.24703530979347102,
                        0.12271818787475017, 0.05063291139240506, 0.02731512325116589,
                        0.011992005329780146, 0.0054630246502331775, 0.0014656895403064624,
                        0.000932711525649567, 0.0013324450366422385, 0.0007994670219853431,
                        0.000932711525649567, 0.0010659560293137908, 0.000932711525649567,
                        0.0006662225183211193, 0.00013324450366422385, 0.0, 0.0, 0.0]

    for ex_weight, ac_weight in zip (expected_weights, actual_weights):
        diff = abs(ex_weight - ac_weight)
        assert diff < 0.1


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
    labels = WeakLabels(2, max_positive_pairs=4, max_negative_pairs=4)
    labels.fit(restaurant_dataset)
    sim01 = labels.tfidf_similarity("0", "1")
    sim23 = labels.tfidf_similarity("2", "3")
    sim45 = labels.tfidf_similarity("4", "5")
    sim67 = labels.tfidf_similarity("6", "7")
    assert(round(sim01, 2) == 0.57)
    assert(round(sim23, 2) == 0.57)
    assert(round(sim45, 2) == 0.72)
    assert(round(sim67, 2) == 0.66)


# @profile(follow=[WeakLabels.fit,
                 # WeakLabels.predict,
                 # WeakLabels.tfidf_similarity])
# def test_profile_restaurant_dnf():
    # dataset = {}
    # attribute_count = None
    # for record in read_csv("restaurant.csv"):
        # if attribute_count is None:
            # attribute_count = len(record[1:])
        # r_id = record[0]
        # r_attributes = record[1:]
        # dataset[r_id] = r_attributes
    # labels = WeakLabels(6, max_positive_pairs=50, max_negative_pairs=200,
                        # upper_threshold=0.7, lower_threshold=0.3)
    # labels.fit_csv("restaurant.csv")
    # labels.predict()

    # blocking_keys = []
    # blocking_keys.append(BlockingKey(0, tokens))
    # blocking_keys.append(BlockingKey(1, tokens))
    # blocking_keys.append(BlockingKey(2, tokens))
    # blocking_keys.append(BlockingKey(3, tokens))
    # blocking_keys.append(BlockingKey(4, tokens))
    # blocking_keys.append(BlockingKey(5, tokens))

    # dnfblock = DisjunctiveBlockingScheme(blocking_keys, labels)
    # dnfblock.transform()
