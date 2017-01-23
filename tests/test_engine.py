#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
test_simindex
----------------------------------

Tests for `simindex` module.
"""

import pytest
import os
from .testprofiler import profile
from .testdata import restaurant_records
from .testhelper import has_common_token

from simindex import SimEngine
from simindex import DisjunctiveBlockingScheme, WeakLabels, SimLearner


# @profile(follow=[SimEngine.fit,
                 # WeakLabels.predict,
                 # WeakLabels.tfidf_similarity,
                 # DisjunctiveBlockingScheme.feature_vector,
                 # DisjunctiveBlockingScheme.transform])
def test_engine():
    pass
    # engine = SimEngine("ferbl-90k")
    # engine.fit_csv("../../master_thesis/datasets/febrl/ferbl-90k-10k-1_train.csv",
                   # ["rec_id", "given_name", "surname", "state", "suburb"])
    # engine.build_csv("../../master_thesis/datasets/febrl/ferbl-90k-10k-1_index.csv",
                   # ["rec_id", "given_name", "surname", "state", "suburb"])
    # engine = SimEngine("ncvoter")
    # engine.fit_csv("../../master_thesis/datasets/ncvoter/ncvoter_train.csv",
                   # ["id", "first_name", "middle_name", "last_name", "city", "street_address"])


@profile(follow=[])
def test_engine_restaurant():
    # Expected results
    blocking_scheme_expected =[[0, 0, 'str', 'common_token'],
                               [0, 1, 'str', 'common_token'],
                               [1, 1, 'id', 'exact_match']]
    sim_strings_expected = ["damerau", "levenshtein", "jaro",
                            "damerau", "ratio"]
    # Test fresh engine
    engine = SimEngine("restaurant",
                       max_positive_labels=111, max_negative_labels=333)
    engine.fit_csv("../../master_thesis/datasets/restaurant/restaurant_train.csv",
                   ["id","name","addr","city","phone","type"])

    sim_strings_actual = \
        SimLearner.prediction_to_strings(engine.similarities)
    assert sim_strings_actual == sim_strings_expected
    assert engine.blocking_scheme_to_strings() == blocking_scheme_expected

    # Build the index
    engine.build_csv("../../master_thesis/datasets/restaurant/restaurant_index.csv",
                     ["id","name","addr","city","phone","type"])

    # Query the index
    engine.query_csv("../../master_thesis/datasets/restaurant/restaurant_train_query.csv",
                     ["id","name","addr","city","phone","type"])

    del engine

    # Test saved engine
    engine = SimEngine("restaurant",
                       max_positive_labels=111, max_negative_labels=333)
    engine.fit_csv("../../master_thesis/datasets/restaurant/restaurant_train.csv",
                   ["id","name","addr","city","phone","type"])

    sim_strings_acutal = \
        SimLearner.prediction_to_strings(engine.similarities)
    assert sim_strings_acutal == sim_strings_expected
    assert engine.blocking_scheme_to_strings() == blocking_scheme_expected

    # Build the index
    engine.build_csv("../../master_thesis/datasets/restaurant/restaurant_index.csv",
                     ["id","name","addr","city","phone","type"])

    # Query the index
    engine.query_csv("../../master_thesis/datasets/restaurant/restaurant_train_query.csv",
                     ["id","name","addr","city","phone","type"])

    # Cleanup
    if os.path.exists(engine.configstore_name):
        os.remove(engine.configstore_name)
    if os.path.exists(engine.traindatastore_name):
        os.remove(engine.traindatastore_name)
    if os.path.exists(engine.indexdatastore_name):
        os.remove(engine.indexdatastore_name)
    if os.path.exists(engine.querydatastore_name):
        os.remove(engine.querydatastore_name)
    if os.path.exists(".%s_RI.idx" % engine.name):
        os.remove(".%s_RI.idx" % engine.name)
    if os.path.exists(".%s_FBI.idx" % engine.name):
        os.remove(".%s_FBI.idx" % engine.name)
    if os.path.exists(".%s_SI.idx" % engine.name):
        os.remove(".%s_SI.idx" % engine.name)
