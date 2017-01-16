#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
test_simindex
----------------------------------

Tests for `simindex` module.
"""

import pytest
from .testdata import restaurant_records
from .testhelper import has_common_token

from pprint import pprint
from difflib import SequenceMatcher

from simindex import draw_frequency_distribution, show
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
    dnf = [Feature([BlockingKey(has_common_token, 0, str.split),
                    BlockingKey(has_common_token, 1, str.split)], 0., 0.),
           Feature([BlockingKey(has_common_token, 0, str.split)], 0., 0.),
           Feature([BlockingKey(has_common_token, 1, str.split)], 0., 0.)]

    msaai = MultiSimAwareAttributeIndex(dnf, __compare)

    msaai.insert(restaurant_records[0])
    result = msaai.query(restaurant_records[1], ["0"])
    assert result == {"0": 2.0}
    assert msaai.FI[0] == {"Mario'sItalian": {"Mario's Pizza"},
                           'MariosItalian': {'Marios Pizza'},
                           'PizzaItalian': {"Mario's Pizza", 'Marios Pizza'},
                           "Mario's": {"Mario's Pizza"},
                           'Marios': {'Marios Pizza'},
                           'Pizza': {"Mario's Pizza", 'Marios Pizza'}}
    assert msaai.FI[1] == {"Italian": {'Italian'},
                           "Mario'sItalian": {'Italian'},
                           'MariosItalian': {'Italian'},
                           'PizzaItalian': {'Italian'}}
    assert msaai.SI == {"Mario's Pizza": {'Marios Pizza': 1.0},
                        "Marios Pizza": {"Mario's Pizza": 1.0}}

    msaai.insert(restaurant_records[4])
    result = msaai.query(restaurant_records[5], ["4"])
    assert result == {"4": 1.9}
    assert msaai.FI[0] == {"Mario'sItalian": {"Mario's Pizza"},
                           'MariosItalian': {'Marios Pizza'},
                           'PizzaItalian': {"Mario's Pizza", 'Marios Pizza'},
                           "Mario's": {"Mario's Pizza"},
                           'Marios': {'Marios Pizza'},
                           'Pizza': {"Mario's Pizza", 'Marios Pizza'},
                           "Yujean": {"Yujean Kang's Best Cuisine",
                                      "Yujean Kang's Gourmet Cuisine"},
                           "Kang's": {"Yujean Kang's Best Cuisine",
                                      "Yujean Kang's Gourmet Cuisine"},
                           "Best": {"Yujean Kang's Best Cuisine"},
                           "Gourmet": {"Yujean Kang's Gourmet Cuisine"},
                           "Cuisine": {"Yujean Kang's Best Cuisine",
                                       "Yujean Kang's Gourmet Cuisine"},
                           "YujeanAsian": {"Yujean Kang's Best Cuisine",
                                           "Yujean Kang's Gourmet Cuisine"},
                           "Kang'sAsian": {"Yujean Kang's Best Cuisine",
                                           "Yujean Kang's Gourmet Cuisine"},
                           "BestAsian": {"Yujean Kang's Best Cuisine"},
                           "GourmetAsian": {"Yujean Kang's Gourmet Cuisine"},
                           "CuisineAsian": {"Yujean Kang's Best Cuisine",
                                            "Yujean Kang's Gourmet Cuisine"}}
    assert msaai.FI[1] == {"Italian": {'Italian'},
                           "Mario'sItalian": {'Italian'},
                           'MariosItalian': {'Italian'},
                           'PizzaItalian': {'Italian'},
                           "Asian": {'Asian'},
                           "YujeanAsian": {"Asian"},
                           "Kang'sAsian": {"Asian"},
                           "BestAsian": {"Asian"},
                           "GourmetAsian": {"Asian"},
                           "CuisineAsian": {"Asian"}}
    assert msaai.SI == {"Mario's Pizza": {'Marios Pizza': 1.0},
                        "Marios Pizza": {"Mario's Pizza": 1.0},
                        "Yujean Kang's Best Cuisine": {"Yujean Kang's Gourmet Cuisine": 0.9},
                        "Yujean Kang's Gourmet Cuisine": {"Yujean Kang's Best Cuisine": 0.9}
                       }
