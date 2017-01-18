#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
test_similarity
----------------------------------

Tests for `simindex` module.
"""

import pytest
from .testdata import restaurant_records, restaurant_dataset
from .testhelper import has_common_token

from simindex.weak_labels import WeakLabels, DisjunctiveBlockingScheme, \
                                 BlockingKey
from simindex.similarity import sim_damerau, sim_jaro, sim_ratio
from simindex import SimLearner


def test_similarity():
    sl = SimLearner(restaurant_dataset)

    labels = WeakLabels(max_positive_pairs=4, max_negative_pairs=4)
    labels.fit(restaurant_records)
    P, N = labels.predict()

    predicted_similarities = sl.predict(P, N)
    assert predicted_similarities[0] == sim_damerau
    assert predicted_similarities[1] == sim_ratio


def test_similarity_pipline():
    labels = WeakLabels(max_positive_pairs=4, max_negative_pairs=4)
    labels.fit(restaurant_records)
    P, N = labels.predict()

    blocking_keys = []
    blocking_keys.append(BlockingKey(has_common_token, 0, str.split))
    blocking_keys.append(BlockingKey(has_common_token, 1, str.split))

    dnfblock = DisjunctiveBlockingScheme(blocking_keys, labels)
    dnf = dnfblock.transform()

    # Only use first combined blocking key which filters half positives and all
    # negatives.
    fP, fN = dnfblock.filter_labels(dnf[:1])

    sl = SimLearner(restaurant_dataset)
    predicted_similarities = sl.predict(fP, fN)
    # assert predicted_similarities[0] == sim_jaro
    # assert predicted_similarities[1] == sim_damerau
