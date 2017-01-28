# -*- coding: utf-8 -*-
from abc import ABCMeta, abstractmethod
from collections import defaultdict, OrderedDict

import numpy as np
import jellyfish
from difflib import SequenceMatcher
from harry import Measures, Hstring
from pprint import pprint

import simindex.helper as hp


class SimLearner():

    def __init__(self, count, dataset):
        self.dataset = dataset
        self.attribute_count = count
        self.measures = [SimBag,
                         SimDamerau,
                         SimLevenshtein,
                         SimJaro,
                         SimRatio]
        self.similarity_objs = []
        for measure in self.measures:
            self.similarity_objs.append(measure())

    def parse_gold(self, gold_standard, gold_attributes):
        # Gold Standard/Ground Truth attributes
        self.gold_pairs = None
        self.gold_records = None
        if gold_standard and gold_attributes:
            self.gold_pairs = hp.read_csv(gold_standard, gold_attributes)
            self.gold_records = {}
            for a, b in self.gold_pairs:
                if a not in self.gold_records.keys():
                    self.gold_records[a] = set()
                if b not in self.gold_records.keys():
                    self.gold_records[b] = set()
                self.gold_records[a].add(b)
                self.gold_records[b].add(a)

    def predict(self, P, N):
        prediction = defaultdict(OrderedDict)
        for field in range(self.attribute_count):
            for sim_obj in self.similarity_objs:
                pmean = self.mean_similarity(P, field, sim_obj.compare)
                nmean = self.mean_similarity(N, field, sim_obj.compare)
                prediction[field][type(sim_obj).__name__] = 1 - pmean + nmean

        result = OrderedDict()
        for field in prediction.keys():
            min = 2.
            for sim_name in prediction[field]:
                sim_value = np.round(prediction[field][sim_name], 5)
                if sim_value < min:
                    min = sim_value
                    result[field] = sim_name

        return list(result.values())

    def mean_similarity(self, pairs, field, similarity):
        result = []

        count = 0
        for a_id, b_id, sim in pairs:
            a_attribute = self.dataset[a_id][field]
            b_attribute = self.dataset[b_id][field]
            result.append(similarity(a_attribute, b_attribute))
            count += 1

        if len(result) == 0:
            return 0.

        return np.mean(result)

    @staticmethod
    def strings_to_prediction(strings):
        possibles = globals().copy()
        possibles.update(locals())
        result_list = []
        for name in strings:
            result_list.append(possibles.get(name))

        return result_list

    @staticmethod
    def prediction_to_strings(prediction):
        result_list = []
        for measure in prediction:
            result_list.append(type(measure).__name__)

        return result_list


class Measure(metaclass=ABCMeta):

    @abstractmethod
    def compare(a, b):
        pass

    @abstractmethod
    def __name__(self):
        pass


class SimBag(Measure):

    def __init__(self):
        self.bag = Measures("dist_bag")
        self.bag.config_set_string("measures.dist_bag.norm", "max")
        self.bag.config_set_bool("measures.global_cache", True)

    def __name__(self):
        return "SimBag"

    def compare(self, a, b):
        ha = Hstring(a)
        hb = Hstring(b)
        return self.bag.compare(ha, hb)


class SimDamerau(Measure):

    def __init__(self):
        self.damerau = Measures("dist_damerau")
        self.damerau.config_set_string("measures.dist_damerau.norm", "max")
        self.damerau.config_set_bool("measures.global_cache", True)

    def __name__(self):
        return "SimDamerau"

    def compare(self, a, b):
        ha = Hstring(a)
        hb = Hstring(b)
        return self.damerau.compare(ha, hb)


class SimLevenshtein(Measure):

    def __init__(self):
        self.levenshtein = Measures("dist_levenshtein")
        self.levenshtein.config_set_string("measures.dist_levenshtein.norm", "max")
        self.levenshtein.config_set_bool("measures.global_cache", True)

    def __name__(self):
        return "SimLevenshtein"

    def compare(self, a, b):
        ha = Hstring(a)
        hb = Hstring(b)
        return self.levenshtein.compare(ha, hb)


class SimJaro(Measure):

    def __init__(self):
        pass

    def __name__(self):
        return "SimJaro"

    def compare(self, a, b):
        return jellyfish.jaro_distance(a, b)


class SimRatio(Measure):

    def __init__(self):
        pass

    def __name__(self):
        return "SimRatio"

    def compare(self, a, b):
        return SequenceMatcher(None, a, b).ratio()