# -*- coding: utf-8 -*-
from collections import Counter, defaultdict
import numpy as np
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted
from harry import Measures, Hstring

from .helper import read_csv

import jellyfish
from difflib import SequenceMatcher

damerau = Measures("dist_damerau")
damerau.config_set_string("measures.dist_damerau.norm", "max")


def sim_damerau(a, b, ins=1., dele=1., sub=1., tra=1.):
    ha = Hstring(a)
    hb = Hstring(b)
    damerau.config_set_float("measures.dist_damerau.cost_ins", ins)
    damerau.config_set_float("measures.dist_damerau.cost_del", dele)
    damerau.config_set_float("measures.dist_damerau.cost_sub", sub)
    damerau.config_set_float("measures.dist_damerau.cost_tra", tra)
    return damerau.compare(ha, hb)


def sim_ratio(a, b):
    return SequenceMatcher(None, a, b).ratio()


def sim_jaro(a, b):
    return jellyfish.jaro_distance(a, b)


class SimLearner(BaseEstimator):

    def __init__(self, similarity='sim_ratio', parameter_a=None,
                 parameter_b=None, parameter_c=None,
                 parameter_d=None, dataset=None, attribute=None):
        self.similarity = similarity
        self.dataset = dataset
        self.parameter_a = parameter_a
        self.parameter_b = parameter_b
        self.parameter_c = parameter_c
        self.parameter_d = parameter_d
        self.attribute = attribute
        self.result = 0.

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

    def compare(self, a, b):
        if self.similarity == 'sim_damerau':
            if self.parameter_a:
                return sim_damerau(a, b,
                                   self.parameter_a,
                                   self.parameter_b,
                                   self.parameter_c,
                                   self.parameter_d)
            else:
                return sim_damerau(a, b)
        elif self.similarity == 'sim_ratio':
            return sim_ratio(a, b)
        elif self.similarity == 'sim_jaro':
            return sim_jaro(a, b)
        else:
            return sim_jaro(a, b)

    def fit(self, X, y):
        # Check that X and y have correct shape
        check_X_y(X, y)
        self.X_ = X
        self.y_ = y
        self.result = 0.

        if not self.dataset:
            # Return the classifier
            return self

        for a_id, b_id in X:
            a_attribute = self.dataset[a_id][self.attribute]
            b_attribute = self.dataset[b_id][self.attribute]
            self.result = np.mean([self.result,
                                   self.compare(a_attribute, b_attribute)])

        # Return the classifier
        return self

    def predict(self, X):
        # Check is fit had been called
        check_is_fitted(self, ['X_', 'y_'])

        # Input validation
        check_array(X)

        toplist = []
        # for field in result.keys():
            # toplist.append(Counter(result[field]).most_common(1)[0][0])

        return toplist

    def score(self, X, y):
        self.fit(X, y)
        print(X, self.similarity, self.result, self.attribute, self.parameter_a, self.parameter_b,
              self.parameter_c, self.parameter_d)

        return self.result

    def get_params(self, deep=True):
        return {"similarity": self.similarity,
                "dataset": self.dataset,
                "attribute": self.attribute,
                "parameter_a": self.parameter_a,
                "parameter_b": self.parameter_b,
                "parameter_c": self.parameter_c,
                "parameter_d": self.parameter_d}

    def set_params(self, **parameters):
        for parameter, value in parameters.items():
            setattr(self, parameter, value)

        return self
