# -*- coding: utf-8 -*-
from .dysim import DySimII
from .weak_labels import WeakLabels, \
                         DisjunctiveBlockingScheme, \
                         BlockingKey, \
                         has_common_token
from .similarity import SimLearner
from .helper import read_csv


class SimEngine(object):

    def __init__(self, indexer=DySimII):
        self.indexer = indexer

        # # Gold Standard/Ground Truth attributes
        # self.gold_pairs = None
        # self.gold_records = None
        # if gold_standard and gold_attributes:
            # self.gold_pairs = []
            # pairs = read_csv(gold_standard, gold_attributes)
            # for gold_pair in pairs:
                # self.gold_pairs.append(gold_pair)

            # self.gold_records = {}
            # for a, b in self.gold_pairs:
                # if a not in self.gold_records.keys():
                    # self.gold_records[a] = set()
                # if b not in self.gold_records.keys():
                    # self.gold_records[b] = set()
                # self.gold_records[a].add(b)
                # self.gold_records[b].add(a)

    def fit_csv(self, index_file):
        self.fit(read_csv(index_file))

    def fit(self, records):
        labels = WeakLabels(max_positive_pairs=400, max_negative_pairs=4000)
        labels.fit(records)
        labels.predict()

        # blocking_keys=[]
        # blocking_keys.append(BlockingKey(has_common_token, 0, str.split))
        # blocking_keys.append(BlockingKey(has_common_token, 1, str.split))

        # dbs = DisjunctiveBlockingScheme(blocking_keys, labels)
        # blocking_key = dbs.transform()
        # P, N = dbs.filter_labels()

        # sl = SimLearner(labels.dataset)
        # best_similarities = sl.predict(P, N)

        # TODO: DySIMII msaai
        # self.indexer.fit_csv(self.index)
