from collections import Counter
from .helper import read_csv, calc_micro_scores, calc_micro_metrics


class DySimII(object):

    def __init__(self, count, encode_fn, simmetric_fn,
                 threshold=-1.0, top_n=-1.0, normalize=False,
                 gold_standard=None, gold_attributes=None,
                 insert_timer=None, query_timer=None):
        self.indicies = []  # Similarity-Aware Inverted Indexes
        self.count = count
        for index in range(self.count):
            self.indicies.append(SimAwareIndex(simmetric_fn=simmetric_fn[index],
                                               encode_fn=encode_fn[index]))
        self.threshold = threshold
        self.top_n = top_n
        self.normalize = normalize
        self.insert_timer = insert_timer
        self.query_timer = query_timer
        # Read gold standard from csv file
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
        self.candidate_count = 0

    def _insert_to_accumulator(self, accumulator, key, value):
        nom_value = value
        if self.normalize:
            nom_value /= self.count

        if key in accumulator:
            accumulator[key] += nom_value
        else:
            accumulator[key] = nom_value

    def _max_similarity(self, cur_val, new_val, current, max):
        if self.normalize:
            return cur_val + (new_val / max) + max - current - 1
        else:
            return cur_val + new_val + max - current - 1

    def fit_csv(self, filename, attributes=[]):
        records = read_csv(filename, attributes)
        for record in records:
            if self.insert_timer:
                with self.insert_timer:
                    self.insert(record)
            else:
                self.insert(record)

    def insert(self, record):
        count = len(record)
        for index, attribute in enumerate(record[1:count]):
            self.indicies[index].insert(attribute, record[0])

    def query(self, record):
        count = len(self.indicies) + 1
        accumulator = {}
        q_id = record[0]
        q_attributes = record[1:count]
        # For each attribute run a query in its index
        for index, attribute in enumerate(q_attributes):
            m = self.indicies[index].query(attribute, q_id)
            self.candidate_count += len(m.items())
            for key, value in m.items():
                if self.threshold > 0:
                    max_sim = self._max_similarity(accumulator.get(key, 0.0),
                                                   value, index, self.count)
                    if max_sim >= self.threshold:
                        self._insert_to_accumulator(accumulator, key, value)
                    elif key in accumulator:
                        del accumulator[key]
                else:
                    self._insert_to_accumulator(accumulator, key, value)

        if self.threshold > 0:
            remove = [key for key in accumulator.keys()
                      if accumulator[key] < self.threshold]
            for key in remove:
                del accumulator[key]

        if self.top_n > 0:
            return dict(Counter(accumulator).most_common(self.top_n))

        return accumulator

    def query_from_csv(self, filename, attributes=[]):
        records = read_csv(filename, attributes)
        accumulator = {}
        y_true1 = []
        y_true2 = []
        y_pred = []
        y_scores = []
        for record in records:
            if self.query_timer:
                with self.query_timer:
                    accumulator[record[0]] = self.query(record)
            else:
                accumulator[record[0]] = self.query(record)

            if self.gold_records:
                calc_micro_scores(record[0], accumulator[record[0]],
                                  y_true1, y_scores, self.gold_records)
                calc_micro_metrics(record[0], accumulator[record[0]],
                                   y_true2, y_pred, self.gold_records)

        if self.gold_records:
            return accumulator, y_true1, y_scores, y_true2, y_pred
        else:
            return accumulator

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

    def frequency_distribution(self):
        """
        Returns the frequency distribution for each attribute
        """
        freq_dis = []
        for index in self.indicies:
            freq_dis.append(index.frequency_distribution())
        return freq_dis


class SimAwareIndex(object):
    def __init__(self, simmetric_fn, encode_fn):
        self.RI = {}    # Record Index (RI)
        self.BI = {}    # Block Index (BI)
        self.SI = {}    # Similarity Index (SI)
        self.simmetric = simmetric_fn
        self.encode = encode_fn

    def insert(self, attribute, r_id):
        if attribute in self.RI.keys():
            self.RI[attribute].add(r_id)
        else:
            self.RI[attribute] = set()
            self.RI[attribute].add(r_id)
            #  Insert value into Block Index
            encoding = self.encode(attribute)
            if encoding not in self.BI.keys():
                self.BI[encoding] = set()

            self.BI[encoding].add(attribute)

            #  Calculate similarities and update SI
            block = list(filter(lambda x: x != attribute, self.BI[encoding]))
            for block_value in block:
                similarity = self.simmetric(attribute, block_value)
                similarity = round(similarity, 1)
                #  Append similarity to block_value
                if block_value not in self.SI.keys():
                    self.SI[block_value] = {}

                self.SI[block_value][attribute] = similarity
                #  Append similarity to value
                if attribute not in self.SI.keys():
                    self.SI[attribute] = {}

                self.SI[attribute][block_value] = similarity

    def query(self, attribute, q_id):
        accumulator = {}
        #  Insert new record into index
        self.insert(attribute, q_id)

        ids = list(filter(lambda x: x != q_id, self.RI[attribute]))
        for id in ids:
            accumulator[id] = 1.0

        if attribute in self.SI:
            for attribute, sim in self.SI[attribute].items():
                for id in self.RI[attribute]:
                    accumulator[id] = sim

        return accumulator

    def frequency_distribution(self):
        """
        Returns the frequency distribution for the block index as dict. Where
        key is size a block and value number of blocks with this size.
        """
        block_sizes = []
        for block_key in self.BI.keys():
            block_sizes.append(len(self.BI[block_key]))

        return Counter(block_sizes)
