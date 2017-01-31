#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
test_simindex
----------------------------------

Tests for `simindex` module.
"""

import pytest
from .testprofiler import profile
from .testdata import restaurant_records
from .testhelper import tokens

from pprint import pprint
from difflib import SequenceMatcher

from simindex import DyKMeans, DyLSH, MultiSimAwareAttributeIndex
from simindex import Feature, BlockingKey

def __compare(a, b):
    return SequenceMatcher(None, a, b).ratio()


def __encode(a):
    return a[:1]


def test_kmeans():
    dy_kmeans = DyKMeans(__encode, __compare, k=2)
    dy_kmeans.fit([["r1", "tony"],
                   ["r5", "tonya"],
                   ["r8", "tonia"],
                   ["r2", "cathrine"],
                   ["r4", "catrine"]])
    assert dy_kmeans.saai.BI == {'t': {'tony', 'tonya', 'tonia'},
                                 'c': {'cathrine', 'catrine'}}
    assert dy_kmeans.saai.SI == {'catrine': {'cathrine': 0.9},
                                 'cathrine': {'catrine': 0.9},
                                 'tony':  {'tonya': 0.9, 'tonia': 0.7},
                                 'tonya': {'tony': 0.9, 'tonia': 0.8},
                                 'tonia': {'tony': 0.7, 'tonya': 0.8}}


def test_lsh():
    dy_lsh = DyLSH(__encode, __compare, lsh_num_perm=2)
    dy_lsh.insert(["r1", "tony"])
    assert dy_lsh.lsh.hashtables == \
        [{b'\x00\x00\x00\x006\x83\xd1&': ['r1']},
         {b'\x00\x00\x00\x00\xdbQYu': ['r1']}]
    assert dy_lsh.saai.BI == {'t': {'tony'}}
    assert dy_lsh.saai.SI == {}
    dy_lsh.insert(["r3", "tony"])
    assert dy_lsh.lsh.hashtables == \
        [{b'\x00\x00\x00\x006\x83\xd1&': ['r1', 'r3']},
         {b'\x00\x00\x00\x00\xdbQYu': ['r1', 'r3']}]
    assert dy_lsh.saai.BI == {'t': {'tony'}}
    assert dy_lsh.saai.SI == {}
    dy_lsh.insert(["r5", "tonya"])
    assert dy_lsh.lsh.hashtables == \
        [{b'\x00\x00\x00\x006\x83\xd1&': ['r1', 'r3'],
          b'\x00\x00\x00\x00 \x94dG': ['r5']},
         {b'\x00\x00\x00\x00y(\xaej': ['r5'],
          b'\x00\x00\x00\x00\xdbQYu': ['r1', 'r3']}]
    assert dy_lsh.saai.BI == {'t': {'tony', 'tonya'}}
    assert dy_lsh.saai.SI == {'tony':  {'tonya': 0.9},
                              'tonya': {'tony': 0.9}}
    dy_lsh.insert(["r8", "tonia"])
    assert dy_lsh.lsh.hashtables == \
        [{b'\x00\x00\x00\x00\xeeo\x89\xfd': ['r8'],
          b'\x00\x00\x00\x006\x83\xd1&': ['r1', 'r3'],
          b'\x00\x00\x00\x00 \x94dG': ['r5']},
         {b'\x00\x00\x00\x00\xdbQYu': ['r1', 'r3'],
          b'\x00\x00\x00\x00\xd4\x93\xcfw': ['r8'],
          b'\x00\x00\x00\x00y(\xaej': ['r5']}]
    assert dy_lsh.saai.BI == {'t': {'tony', 'tonya', 'tonia'}}
    assert dy_lsh.saai.SI == {'tony':  {'tonya': 0.9, 'tonia': 0.7},
                              'tonya': {'tony': 0.9, 'tonia': 0.8},
                              'tonia': {'tony': 0.7, 'tonya': 0.8}}
    dy_lsh.insert(["r2", "cathrine"])
    assert dy_lsh.lsh.hashtables == \
        [{b'\x00\x00\x00\x00 \x94dG': ['r5'],
          b'\x00\x00\x00\x006\x83\xd1&': ['r1', 'r3'],
          b'\x00\x00\x00\x00\xeeo\x89\xfd': ['r8'],
          b'\x00\x00\x00\x00\xfaj\x02\x00': ['r2']},
         {b'\x00\x00\x00\x00\xd4\x93\xcfw': ['r8'],
          b'\x00\x00\x00\x00\xdbQYu': ['r1', 'r3'],
          b'\x00\x00\x00\x00\x03\x7f\xa9\xe9': ['r2'],
          b'\x00\x00\x00\x00y(\xaej': ['r5']}]
    assert dy_lsh.saai.BI == {'t': {'tony', 'tonya', 'tonia'},
                              'c': {'cathrine'}}
    assert dy_lsh.saai.SI == {'tony':  {'tonya': 0.9, 'tonia': 0.7},
                              'tonya': {'tony': 0.9, 'tonia': 0.8},
                              'tonia': {'tony': 0.7, 'tonya': 0.8}}
    dy_lsh.insert(["r4", "catrine"])
    assert dy_lsh.lsh.hashtables == \
        [{b'\x00\x00\x00\x00 \x94dG': ['r5'],
          b'\x00\x00\x00\x006\x83\xd1&': ['r1', 'r3'],
          b'\x00\x00\x00\x00\xeeo\x89\xfd': ['r8'],
          b'\x00\x00\x00\x00\xfaj\x02\x00': ['r2'],
          b'\x00\x00\x00\x00\xcf\x01\x9d\xc9': ['r4']},
         {b'\x00\x00\x00\x00\xd4\x93\xcfw': ['r8'],
          b'\x00\x00\x00\x00\xdbQYu': ['r1', 'r3'],
          b'\x00\x00\x00\x00\x03\x7f\xa9\xe9': ['r2'],
          b'\x00\x00\x00\x00y(\xaej': ['r5'],
          b'\x00\x00\x00\x00\xc7`\x0f\xa5': ['r4']}]
    assert dy_lsh.saai.BI == {'t': {'tony', 'tonya', 'tonia'},
                              'c': {'cathrine', 'catrine'}}
    assert dy_lsh.saai.SI == {'catrine': {'cathrine': 0.9},
                              'cathrine': {'catrine': 0.9},
                              'tony':  {'tonya': 0.9, 'tonia': 0.7},
                              'tonya': {'tony': 0.9, 'tonia': 0.8},
                              'tonia': {'tony': 0.7, 'tonya': 0.8}}


def test_MultiSimAwareAttributeIndex():
    dnf = [Feature([BlockingKey(0, tokens),
                    BlockingKey(1, tokens)], 0., 0.),
           Feature([BlockingKey(0, tokens)], 0., 0.),
           Feature([BlockingKey(1, tokens)], 0., 0.)]

    msaai = MultiSimAwareAttributeIndex(dnf, [__compare, __compare])

    msaai.insert(restaurant_records[0][0], restaurant_records[0][1:])
    result = msaai.query(restaurant_records[1], ["0"])
    assert result == {"0": 1.96}
    assert msaai.FBI[0] == {"mario'sitalian": {"mario's pizza"},
                           'mariositalian': {'marios pizza'},
                           'pizzaitalian': {"mario's pizza", 'marios pizza'},
                           "mario's": {"mario's pizza"},
                           'marios': {'marios pizza'},
                           'pizza': {"mario's pizza", 'marios pizza'}}
    assert msaai.FBI[1] == {"italian": {'italian'},
                           "mario'sitalian": {'italian'},
                           'mariositalian': {'italian'},
                           'pizzaitalian': {'italian'}}
    assert msaai.SI == {"mario's pizza": {'marios pizza': 0.96},
                        "marios pizza": {"mario's pizza": 0.96}}

    msaai.insert(restaurant_records[4][0], restaurant_records[4][1:])
    result = msaai.query(restaurant_records[1], ["0"])
    result = msaai.query(restaurant_records[5], ["4"])
    assert result == {"4": 1.8727272727272726}
    assert msaai.FBI[0] == {"mario'sitalian": {"mario's pizza"},
                           'mariositalian': {'marios pizza'},
                           'pizzaitalian': {"mario's pizza", 'marios pizza'},
                           "mario's": {"mario's pizza"},
                           'marios': {'marios pizza'},
                           'pizza': {"mario's pizza", 'marios pizza'},
                           "yujean": {"yujean kang's best cuisine",
                                      "yujean kang's gourmet cuisine"},
                           "kang's": {"yujean kang's best cuisine",
                                      "yujean kang's gourmet cuisine"},
                           "best": {"yujean kang's best cuisine"},
                           "gourmet": {"yujean kang's gourmet cuisine"},
                           "cuisine": {"yujean kang's best cuisine",
                                       "yujean kang's gourmet cuisine"},
                           "yujeanasian": {"yujean kang's best cuisine",
                                           "yujean kang's gourmet cuisine"},
                           "kang'sasian": {"yujean kang's best cuisine",
                                           "yujean kang's gourmet cuisine"},
                           "bestasian": {"yujean kang's best cuisine"},
                           "gourmetasian": {"yujean kang's gourmet cuisine"},
                           "cuisineasian": {"yujean kang's best cuisine",
                                            "yujean kang's gourmet cuisine"}}
    assert msaai.FBI[1] == {"italian": {'italian'},
                           "mario'sitalian": {'italian'},
                           'mariositalian': {'italian'},
                           'pizzaitalian': {'italian'},
                           "asian": {'asian'},
                           "yujeanasian": {"asian"},
                           "kang'sasian": {"asian"},
                           "bestasian": {"asian"},
                           "gourmetasian": {"asian"},
                           "cuisineasian": {"asian"}}
    assert msaai.SI == {"mario's pizza": {'marios pizza': 0.96},
                        "marios pizza": {"mario's pizza": 0.96},
                        "yujean kang's best cuisine": {"yujean kang's gourmet cuisine": 0.8727272727272727},
                        "yujean kang's gourmet cuisine": {"yujean kang's best cuisine": 0.8727272727272727}
                       }


# @profile(follow=[DyLSH.query_from_csv,
                 # DyLSH.query,
                 # SimAwareAttributeIndex.query])
# def test_profile_lash():
    # dy_lsh = DyLSH(__encode, __compare, lsh_num_perm=20)
    # dy_lsh.fit_csv("restaurant.csv", ["id", "name", "addr", "city"])
    # dy_lsh.query_from_csv("restaurant.csv", ["id", "name", "addr", "city"])
