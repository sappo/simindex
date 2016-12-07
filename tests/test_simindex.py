#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
test_simindex
----------------------------------

Tests for `simindex` module.
"""

import pytest

# from contextlib import contextmanager
from click.testing import CliRunner
from difflib import SequenceMatcher

from simindex.dysim import SimAwareIndex
from simindex import DySimII
from simindex import cli


def _compare(a, b):
    return SequenceMatcher(None, a, b).ratio()


def _encode(a):
    return a[:1]


@pytest.fixture
def response():
    """Sample pytest fixture.
    See more at: http://doc.pytest.org/en/latest/fixture.html
    """
    # import requests
    # return requests.get('https://github.com/audreyr/cookiecutter-pypackage')


def test_content(response):
    """Sample pytest test function with the pytest fixture as an argument.
    """
    # from bs4 import BeautifulSoup
    # assert 'GitHub' in BeautifulSoup(response.content).title.string


def test_command_line_interface():
    runner = CliRunner()
    result = runner.invoke(cli.main)
    assert result.exit_code == 0
    assert 'simindex.cli.main' in result.output
    help_result = runner.invoke(cli.main, ['--help'])
    assert help_result.exit_code == 0
    assert '--help  Show this message and exit.' in help_result.output


def test_simawareindex_insert():
    s = SimAwareIndex(_compare, _encode)
    s.insert("tony", "r1")
    assert s.RI == {'tony': {'r1'}}
    assert s.BI == {'t': {'tony'}}
    assert s.SI == {}
    s.insert("tony", "r3")
    assert s.RI == {'tony': {'r1', 'r3'}}
    assert s.BI == {'t': {'tony'}}
    assert s.SI == {}
    s.insert("tonya", "r5")
    assert s.RI == {'tony': {'r1', 'r3'}, 'tonya': {'r5'}}
    assert s.BI == {'t': {'tony', 'tonya'}}
    assert s.SI == {'tony':  {'tonya': 0.9},
                    'tonya': {'tony': 0.9}}
    s.insert("tonia", "r8")
    assert s.RI == {'tony': {'r1', 'r3'}, 'tonya': {'r5'}, 'tonia': {'r8'}}
    assert s.BI == {'t': {'tony', 'tonya', 'tonia'}}
    assert s.SI == {'tony':  {'tonya': 0.9, 'tonia': 0.7},
                    'tonya': {'tony': 0.9, 'tonia': 0.8},
                    'tonia': {'tony': 0.7, 'tonya': 0.8}}
    s.insert("cathrine", "r2")
    assert s.RI == {'cathrine': {'r2'},
                    'tony': {'r1', 'r3'}, 'tonya': {'r5'}, 'tonia': {'r8'}}
    assert s.BI == {'t': {'tony', 'tonya', 'tonia'},
                    'c': {'cathrine'}}
    assert s.SI == {'tony':  {'tonya': 0.9, 'tonia': 0.7},
                    'tonya': {'tony': 0.9, 'tonia': 0.8},
                    'tonia': {'tony': 0.7, 'tonya': 0.8}}
    s.insert("catrine", "r4")
    assert s.RI == {'cathrine': {'r2'}, 'catrine': {'r4'},
                    'tony': {'r1', 'r3'}, 'tonya': {'r5'}, 'tonia': {'r8'}}
    assert s.BI == {'t': {'tony', 'tonya', 'tonia'},
                    'c': {'cathrine', 'catrine'}}
    assert s.SI == {'catrine': {'cathrine': 0.9},
                    'cathrine': {'catrine': 0.9},
                    'tony':  {'tonya': 0.9, 'tonia': 0.7},
                    'tonya': {'tony': 0.9, 'tonia': 0.8},
                    'tonia': {'tony': 0.7, 'tonya': 0.8}}


def test_simawareindex_query():
    s = SimAwareIndex(_compare, _encode)
    s.insert("tony", "r1")
    s.insert("cathrine", "r2")
    s.insert("catrine", "r4")
    s.insert("tonya", "r5")
    s.insert("tonia", "r8")

    res = s.query("tony", "r10")
    assert res == {'r1': 1.0, 'r5': 0.9, 'r8': 0.7}
    res = s.query("catrine", "r11")
    assert res == {'r2': 0.9, 'r4': 1.0}
    res = s.query("catrina", "r12")
    assert res == {'r2': 0.8, 'r4': 0.9, 'r11': 0.9}
    res = s.query("catrine", "r13")
    assert res == {'r2': 0.9, 'r4': 1.0, 'r11': 1.0, 'r12': 0.9}


def test_dysimII_insert():
    s1 = DySimII(1, [_encode], [_compare])
    s1.insert(["r1", "tony"])
    assert s1.indicies[0].RI == {'tony': {'r1'}}
    assert s1.indicies[0].BI == {'t': {'tony'}}
    assert s1.indicies[0].SI == {}
    s1.insert(["r3", "tony"])
    assert s1.indicies[0].RI == {'tony': {'r1', 'r3'}}
    assert s1.indicies[0].BI == {'t': {'tony'}}
    assert s1.indicies[0].SI == {}

    s2 = DySimII(2, [_encode, _encode], [_compare, _compare])
    s2.insert(["r1", "tony", "23465"])
    assert s2.indicies[0].RI == {'tony': {'r1'}}
    assert s2.indicies[0].BI == {'t': {'tony'}}
    assert s2.indicies[0].SI == {}
    assert s2.indicies[1].RI == {'23465': {'r1'}}
    assert s2.indicies[1].BI == {'2': {'23465'}}
    assert s2.indicies[1].SI == {}
    s2.insert(["r2", "tonya", "23465"])
    s2.insert(["r3", "tonia", "23578"])
    assert s2.indicies[0].RI == {'tony': {'r1'}, 'tonya': {'r2'},
                                 'tonia': {'r3'}}
    assert s2.indicies[0].BI == {'t': {'tony', 'tonya', 'tonia'}}
    assert s2.indicies[0].SI == {'tony':  {'tonya': 0.9, 'tonia': 0.7},
                                 'tonya': {'tony': 0.9, 'tonia': 0.8},
                                 'tonia': {'tony': 0.7, 'tonya': 0.8}}
    assert s2.indicies[1].RI == {'23465': {'r1', 'r2'}, '23578': {'r3'}}
    assert s2.indicies[1].BI == {'2': {'23465', '23578'}}
    assert s2.indicies[1].SI == {'23465': {'23578': 0.6},
                                 '23578': {'23465': 0.6}}


def test_dysimII_query():
    s1 = DySimII(2, [_encode, _encode], [_compare, _compare])
    s1.insert(["r1", "tony", "23465"])
    s1.insert(["r2", "tonya", "23465"])
    s1.insert(["r3", "tonia", "23578"])
    res = s1.query(["r4", "tony", "23578"])
    assert res == {'r1': 1.6, 'r2': 1.5, 'r3': 1.7}

    s1t = DySimII(2, [_encode, _encode], [_compare, _compare], threshold=1.6)
    s1t.insert(["r1", "tony", "23465"])
    s1t.insert(["r2", "tonya", "23465"])
    s1t.insert(["r3", "tonia", "23578"])
    res = s1t.query(["r4", "tony", "23578"])
    assert res == {'r1': 1.6, 'r3': 1.7}

    s2 = DySimII(2, [_encode, _encode], [_compare, _compare], normalize=True)
    s2.insert(["r1", "tony", "23465"])
    s2.insert(["r2", "tonya", "23465"])
    s2.insert(["r3", "tonia", "23578"])
    res = s2.query(["r4", "tony", "23578"])
    assert res == {'r1': 0.8, 'r2': 0.75, 'r3': 0.85}

    s2t = DySimII(2, [_encode, _encode], [_compare, _compare],
                  threshold=0.79, normalize=True)
    s2t.insert(["r1", "tony", "23465"])
    s2t.insert(["r2", "tonya", "23465"])
    s2t.insert(["r3", "tonia", "23578"])
    res = s2t.query(["r4", "tony", "23578"])
    assert res == {'r1': 0.8, 'r3': 0.85}

    s3 = DySimII(2, [_encode, _encode], [_compare, _compare], top_n=2)
    s3.insert(["r1", "tony", "23465"])
    s3.insert(["r2", "tonya", "23465"])
    s3.insert(["r3", "tonia", "23578"])
    res = s3.query(["r4", "tony", "23578"])
    assert res == {'r1': 1.6, 'r3': 1.7}
