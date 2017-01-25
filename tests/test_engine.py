#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
test_simindex
----------------------------------

Tests for `simindex` module.
"""

import os
import pytest
from pprint import pprint

from .testprofiler import profile
from .testdata import restaurant_records

from simindex import SimEngine, MDySimII, MDySimIII
from simindex import DisjunctiveBlockingScheme, WeakLabels, SimLearner


# @profile(follow=[SimEngine.fit,
                 # WeakLabels.predict,
                 # WeakLabels.tfidf_similarity,
                 # DisjunctiveBlockingScheme.feature_vector,
                 # DisjunctiveBlockingScheme.transform])
# @profile(follow=[MDySimII.insert, MDySimII.query,
                 # MDySimIII.insert, MDySimIII.query])
def test_engine(verbose):
    engine = SimEngine("ferbl-90k", indexer=MDySimIII, verbose=verbose)
    print("Fit")
    engine.fit_csv("../../master_thesis/datasets/febrl/ferbl-90k-10k-1_train.csv",
                   ["rec_id", "given_name", "surname", "state", "suburb"])
    print("Build")
    engine.build_csv("../../master_thesis/datasets/febrl/ferbl-90k-10k-1_index.csv",
                     ["rec_id", "given_name", "surname", "state", "suburb"])
    print(engine.indexer.frequency_distribution())
    print("Query")
    engine.query_csv("../../master_thesis/datasets/febrl/ferbl-90k-10k-1_train_query.csv",
                     ["rec_id", "given_name", "surname", "state", "suburb"])
    print("Results")
    gold_csv = "../../master_thesis/datasets/febrl/ferbl-90k-10k-1_train_gold.csv"
    engine.read_ground_truth(gold_standard=gold_csv, gold_attributes=["id_1", "id_2"])
    print("Pair completeness:", engine.pair_completeness())
    print("Reduction ratio:", engine.reduction_ratio())


# @profile(follow=[MDySimII.insert, MDySimII.query,
#                  MDySimIII.insert, MDySimIII.query])
def test_engine_restaurant(verbose):
    # Expected results
    blocking_scheme_expected =[[0, 0, 'tokens', 'has_common_token'],
                               [0, 1, 'tokens', 'has_common_token'],
                               [1, 1, 'term_id', 'is_exact_match']]
    sim_strings_expected = ["SimDamerau", "SimLevenshtein", "SimDamerau",
                            "SimDamerau", "SimRatio"]
    for indexer in [MDySimII, MDySimIII]:
        print()
        print("--------------------------------------------------------------")
        print("Testing fresh engine (%s) on restaurant dataset:" % indexer.__name__)
        # Test with fresh engine
        engine = SimEngine("restaurant", indexer=indexer, verbose=verbose,
                           max_positive_labels=111, max_negative_labels=333)
        # Clean any leftovers from failed tests, verbose=verbose
        restaurant_cleanup(engine)

        engine.fit_csv("../../master_thesis/datasets/restaurant/restaurant_train.csv",
                       ["id","name","addr","city","phone","type"])

        sim_strings_actual = engine.similarities
        assert sim_strings_actual == sim_strings_expected
        assert engine.blocking_scheme_to_strings() == blocking_scheme_expected

        # Build the index
        engine.build_csv("../../master_thesis/datasets/restaurant/restaurant_index.csv",
                         ["id","name","addr","city","phone","type"])

        # Query the index
        engine.query_csv("../../master_thesis/datasets/restaurant/restaurant_train_query.csv",
                         ["id","name","addr","city","phone","type"])

        # Metrics
        gold_csv = "../../master_thesis/datasets/restaurant/restaurant_train_gold.csv"
        engine.read_ground_truth(gold_standard=gold_csv, gold_attributes=["id_1", "id_2"])
        fresh_pc = engine.pair_completeness()
        fresh_rr = engine.reduction_ratio()
        print("Pair completeness:", fresh_pc)
        print("Reduction ratio:", fresh_rr)
        del engine

        print()
        print("--------------------------------------------------------------")
        print("Testing saved engine (%s) on restaurant dataset:" % indexer.__name__)
        # Test with saved engine
        engine = SimEngine("restaurant", indexer=indexer, verbose=verbose,
                           max_positive_labels=111, max_negative_labels=333)
        engine.fit_csv("../../master_thesis/datasets/restaurant/restaurant_train.csv",
                       ["id","name","addr","city","phone","type"])

        sim_strings_actual = engine.similarities
        assert sim_strings_actual == sim_strings_expected
        assert engine.blocking_scheme_to_strings() == blocking_scheme_expected

        # Build the index
        engine.build_csv("../../master_thesis/datasets/restaurant/restaurant_index.csv",
                         ["id","name","addr","city","phone","type"])

        # Query the index
        engine.query_csv("../../master_thesis/datasets/restaurant/restaurant_train_query.csv",
                         ["id","name","addr","city","phone","type"])

        # Metrics
        gold_csv = "../../master_thesis/datasets/restaurant/restaurant_train_gold.csv"
        engine.read_ground_truth(gold_standard=gold_csv, gold_attributes=["id_1", "id_2"])
        saved_pc = engine.pair_completeness()
        saved_rr = engine.reduction_ratio()
        assert fresh_pc == saved_pc
        assert fresh_rr == saved_rr
        print("Pair completeness:", saved_pc)
        print("Reduction ratio:", saved_rr)

        # Cleanup
        restaurant_cleanup(engine)


def restaurant_cleanup(engine):
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
