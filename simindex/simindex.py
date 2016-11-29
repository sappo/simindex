# -*- coding: utf-8 -*-
import csv
from sklearn.metrics import recall_score, precision_score
from difflib import SequenceMatcher


def __compare__(a, b):
    return SequenceMatcher(None, a, b).ratio()


def __encode__(a):
    return a[0:1]


class DySimII(object):
    def __init__(self, count, threshold=-1.0, normalize=False,
                 encode_fn=__encode__, simmetric_fn=__compare__,
                 gold_standard=None, gold_attributes=None):
        self.indicies = []  # Similarity-Aware Inverted Indexes
        self.count = count
        for index in range(self.count):
            self.indicies.append(SimAwareIndex(simmetric_fn=simmetric_fn,
                                               encode_fn=encode_fn))
        self.threshold = threshold
        self.normalize = normalize
        self.gold_pairs = self._read_csv(gold_standard, gold_attributes)
        self.gold_records = {}
        for a, b in self.gold_pairs:
            if a not in self.gold_records.keys():
                self.gold_records[a] = set()
            if b not in self.gold_records.keys():
                self.gold_records[b] = set()
            self.gold_records[a].add(b)
            self.gold_records[b].add(a)

    def __insert_to_accumulator__(self, accumulator, key, value):
        if key in accumulator:
            accumulator[key] += value
            if self.normalize:
                accumulator[key] /= self.count
        else:
            accumulator[key] = value

    def __max_similarity_value__(self, value, current, max):
        if self.normalize:
            return (value + max - current - 1) / max
        else:
            return value + max - current - 1

    def _read_csv(self, filename, attributes=[],
                  percentage=1.0, delimiter=','):
        lines = []
        columns = []
        with open(filename, newline='') as csvfile:
            reader = csv.reader(csvfile, delimiter=delimiter, quotechar='"')
            data = list(reader)
            row_count = len(data)
            threshold = int(row_count * percentage)
            for index, row in enumerate(data[:threshold]):
                if index == 0:
                    for attribute in attributes:
                        columns.append(row.index(attribute))
                else:
                    if len(columns) > 0:
                        line = []
                        for col in columns:
                            line.append(str(row[col]).strip())
                        lines.append(line)
                    else:
                        lines.append(row)

        return lines

    def recall(self):
        """
        Returns the recall from the passed gold standard and the index data.

        Warning: The recall calculation does not take the threshold into
                 account!
        """
        total_p = len(self.gold_pairs)  # True positives + False negatives
        true_p = 0
        for id1, id2 in self.gold_pairs:
            # 1. For each duplicate pair
            hit = False
            for index in range(self.count):
                # 2. Look at each indexed attribute
                for key in self.indicies[index].BI:
                    # 3. Check every block to find out, if both records have at
                    # at least one block in common
                    hit_1, hit_2 = False, False
                    values = self.indicies[index].BI[key]
                    for value in values:
                        if not hit_1:
                            hit_1 = id1 in self.indicies[index].RI[value]
                        if not hit_2:
                            hit_2 = id2 in self.indicies[index].RI[value]

                    if hit_1 and hit_2:
                        # 4. If both records are in same block abort search
                        hit = True
                        break

                if hit:
                    # 5. If duplicate pair has been found in at least on block
                    # count it as true positive
                    true_p += 1
                    break

        return true_p / total_p

    def insert(self, record):
        count = len(record)
        for index, attribute in enumerate(record[1:count]):
            self.indicies[index].insert(attribute, record[0])

    def insert_from_csv(self, filename, attributes=[]):
        records = self._read_csv(filename, attributes)
        for record in records:
            self.insert(record)

    def query(self, record):
        count = len(self.indicies) + 1
        accumulator = {}
        for index, attribute in enumerate(record[1:count]):
            m = self.indicies[index].query(attribute, record[0])
            for key, value in m.items():
                if self.threshold > 0:
                    max_sim = self.__max_similarity_value__(
                        accumulator.get(key, 0.0) + value, index, self.count)
                    if max_sim >= self.threshold:
                        self.__insert_to_accumulator__(accumulator, key, value)
                else:
                    self.__insert_to_accumulator__(accumulator, key, value)

        if self.threshold > 0:
            remove = [key for key in accumulator.keys() if accumulator[key] < self.threshold]
            for key in remove:
                del accumulator[key]

        return accumulator

    def _calc_micro_scores(self, record, result, y_true, y_score):
        for a in result.keys():
            y_score.append(result[a])
            if record[0] in self.gold_records.keys() and a in self.gold_records[record[0]]:
                y_true.append(1)
            else:
                y_true.append(0)

        if record[0] in self.gold_records:
            for fn in self.gold_records[record[0]].difference(result.keys()):
                y_true.append(1)
                y_score.append(0.)

    def _calc_micro_metrics(self, record, result, y_true, y_pred):
        for a in result.keys():
            y_pred.append(1)
            if record[0] in self.gold_records.keys() and a in self.gold_records[record[0]]:
                y_true.append(1)
            else:
                y_true.append(0)

        if record[0] in self.gold_records:
            for fn in self.gold_records[record[0]].difference(result.keys()):
                y_true.append(1)
                y_pred.append(0)

    def query_from_csv(self, filename, attributes=[]):
        records = self._read_csv(filename, attributes)
        accumulator = {}
        y_true1 = []
        y_true2 = []
        y_pred = []
        y_scores = []
        for record in records:
            if self.gold_records:
                accumulator[record[0]] = self.query(record)
                self._calc_micro_scores(record, accumulator[record[0]],
                                        y_true1, y_scores)
                self._calc_micro_metrics(record, accumulator[record[0]],
                                         y_true2, y_pred)
            else:
                accumulator[record[0]] = self.query(record)

        if self.gold_records:
            return accumulator, y_true1, y_scores, y_true2, y_pred
        else:
            return accumulator


class SimAwareIndex(object):
    def __init__(self, simmetric_fn=__compare__, encode_fn=__encode__):
        self.RI = {}    # Record Index (RI)
        self.BI = {}    # Block Index (BI)
        self.SI = {}    # Similarity Index (SI)
        self.simmetric = simmetric_fn
        self.encode = encode_fn

    def insert(self, value, identifier):
        value_known = value in self.RI.keys()
        if value_known:
            self.RI[value].append(identifier)
        else:
            self.RI[value] = [identifier]
            #  Insert value into Block Index
            encoding = self.encode(value)
            if encoding not in self.BI.keys():
                self.BI[encoding] = [value]
            else:
                self.BI[encoding].append(value)

            #  Calculate similarities and update SI
            block = list(filter(lambda x: x != value, self.BI[encoding]))
            for block_value in block:
                similarity = self.simmetric(value, block_value)
                similarity = round(similarity, 1)
                #  Append similarity to block_value
                if block_value not in self.SI.keys():
                    self.SI[block_value] = [(value, similarity)]
                else:
                    self.SI[block_value].append((value, similarity))
                #  Append similarity to value
                if value not in self.SI.keys():
                    self.SI[value] = [(block_value, similarity)]
                else:
                    self.SI[value].append((block_value, similarity))

    def query(self, value, identifier):
        accumulator = {}
        #  Insert new record into index
        self.insert(value, identifier)

        ids = list(filter(lambda x: x != identifier, self.RI[value]))
        for id in ids:
            accumulator[id] = 1.0

        if value in self.SI:
            for value, sim in self.SI[value]:
                for id in self.RI[value]:
                    accumulator[id] = sim

        return accumulator
