# -*- coding: utf-8 -*-
from collections import defaultdict
import numpy as np
from harry import Measures, Hstring

from .helper import read_csv

import jellyfish
from difflib import SequenceMatcher

bag = Measures("dist_bag")
bag.config_set_string("measures.dist_bag.norm", "max")
damerau = Measures("dist_damerau")
damerau.config_set_string("measures.dist_damerau.norm", "max")
levenshtein = Measures("dist_levenshtein")
levenshtein.config_set_string("measures.dist_levenshtein.norm", "max")


def sim_bag(a, b):
    ha = Hstring(a)
    hb = Hstring(b)
    return bag.compare(ha, hb)


def sim_damerau(a, b):
    ha = Hstring(a)
    hb = Hstring(b)
    return damerau.compare(ha, hb)


def sim_levenshtein(a, b):
    ha = Hstring(a)
    hb = Hstring(b)
    return levenshtein.compare(ha, hb)


def sim_ratio(a, b):
    return SequenceMatcher(None, a, b).ratio()


def sim_jaro(a, b):
    return jellyfish.jaro_distance(a, b)


class SimLearner():

    def __init__(self, dataset):
        self.dataset = dataset
        self.attribute_count = len(dataset[next(iter(dataset))])
        self.similarities = [sim_bag,
                             sim_damerau,
                             sim_levenshtein,
                             sim_jaro,
                             sim_ratio]

    def parse_gold(self, gold_standard, gold_attributes):
        # Gold Standard/Ground Truth attributes
        self.gold_pairs = None
        self.gold_records = None
        if gold_standard and gold_attributes:
            self.gold_pairs = read_csv(gold_standard, gold_attributes)
            self.gold_records = {}
            for a, b in self.gold_pairs:
                if a not in self.gold_records.keys():
                    self.gold_records[a] = set()
                if b not in self.gold_records.keys():
                    self.gold_records[b] = set()
                self.gold_records[a].add(b)
                self.gold_records[b].add(a)

    def predict(self, P, N):
        pred = defaultdict(dict)
        for attribute in range(self.attribute_count):
            for similarity in self.similarities:
                mean = self.mean_similarity(P, attribute, similarity)
                pred[attribute][similarity] = 1 - mean

        for attribute in range(self.attribute_count):
            for similarity in self.similarities:
                mean = self.mean_similarity(N, attribute, similarity)
                pred[attribute][similarity] += mean

        print(pred)
        result = {}
        for attribute in pred.keys():
            min = 2.
            for similarity in pred[attribute]:
                if pred[attribute][similarity] < min:
                    min = pred[attribute][similarity]
                    result[attribute] = similarity

        return result

    def mean_similarity(self, pairs, attribute, similarity):
        result = []

        for a_id, b_id, sim in pairs:
            a_attribute = self.dataset[a_id][attribute]
            b_attribute = self.dataset[b_id][attribute]
            result.append(similarity(a_attribute, b_attribute))

        if len(result) == 0:
            return 0.

        return np.mean(result)
