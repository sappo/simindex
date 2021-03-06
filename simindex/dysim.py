import os
import pickle
from collections import Counter, defaultdict
from functools import partial
import simindex.helper as hp
import numpy as np


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
            self.gold_pairs = []
            pairs = hp.read_csv(gold_standard, gold_attributes)
            for gold_pair in pairs:
                self.gold_pairs.append(gold_pair)

            self.gold_records = defaultdict(set)
            for a, b in self.gold_pairs:
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
        records = hp.read_csv(filename, attributes)
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
        accumulator = {}
        y_true1 = []
        y_true2 = []
        y_pred = []
        y_scores = []
        records = read_csv(filename, attributes)
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
        self.RI = defaultdict(set)    # Record Index (RI)
        self.BI = defaultdict(set)    # Block Index (BI)
        self.SI = defaultdict(dict)   # Similarity Index (SI)
        self.simmetric = simmetric_fn
        self.encode = encode_fn

    def insert(self, r_attribute, r_id):
        self.RI[r_attribute].add(r_id)
        if r_attribute not in self.SI.keys():
            #  Insert value into Block Index
            encoding = self.encode(r_attribute)
            self.BI[encoding].add(r_attribute)

            #  Calculate similarities and update SI
            block = list(filter(lambda x: x != r_attribute, self.BI[encoding]))
            for block_value in block:
                similarity = self.simmetric(r_attribute, block_value)
                similarity = round(similarity, 1)
                #  Append similarity to block_value
                self.SI[block_value][r_attribute] = similarity
                #  Append similarity to value
                self.SI[r_attribute][block_value] = similarity

    def query(self, q_attribute, q_id):
        accumulator = {}
        #  Insert new record into index
        self.insert(q_attribute, q_id)

        for id in self.RI[q_attribute]:
            if q_id != id:
                accumulator[id] = 1.0

        if q_attribute in self.SI:
            for attribute, sim in self.SI[q_attribute].items():
                for id in self.RI[attribute]:
                    accumulator[id] = sim

        return accumulator

    def frequency_distribution(self):
        """
        Returns the frequency distribution for the block index as dict. Where key is size a block and value number of blocks with this size.
        """
        block_sizes = []
        for block_key in self.BI.keys():
            block_sizes.append(len(self.BI[block_key]))

        return Counter(block_sizes)


class MDySimII(object):

    def __init__(self, count, dns_blocking_scheme, similarity_fns,
                 use_full_simvector=False, use_parfull_simvector=False):
        self.attribute_count = count
        self.dns_blocking_scheme = dns_blocking_scheme
        self.similarity_fns = similarity_fns
        self.dataset = {}

        # Class structure
        self.nrecords = 0              # Number of records indexed
        self.RI = defaultdict(set)     # Record Index (RI)
        self.FBI = defaultdict(dict)   # Field Block Indicies (FBI)
        self.SI = defaultdict(dict)    # Similarity Index (SI)

        self.use_full_simvector = use_full_simvector
        self.use_parfull_simvector = use_parfull_simvector

        if self.use_parfull_simvector:
            # Calculate similarites between all attributes covered by
            # the blocking schema
            self.fields = set()
            for blocking_key in self.dns_blocking_scheme:
                self.fields.update(blocking_key.covered_fields())

    def insert(self, r_id, r_attributes):
        for feature in self.dns_blocking_scheme:
            # Add id to RI index
            for field in feature.covered_fields():
                self.RI[r_attributes[field]].add(r_id)

            # Generate blocking keys for feature
            blocking_key_values = feature.blocking_key_values(r_attributes)
            # Insert record into block and calculate similarities
            for encoding in blocking_key_values:
                for blocking_key in feature.blocking_keys:
                    field = blocking_key.field
                    attribute = r_attributes[field]

                    BI = self.FBI[field]
                    if encoding not in BI.keys():
                        BI[encoding] = set()

                    BI[encoding].add(attribute)

                    #  Calculate similarities and update SI
                    block = list(filter(lambda x: x != attribute, BI[encoding]))
                    for block_value in block:
                        if block_value not in self.SI[attribute]:
                            similarity = self.similarity_fns[field](attribute, block_value)
                            #  Append similarity to block_value
                            self.SI[block_value][attribute] = similarity

                            #  Append similarity to attribute
                            self.SI[attribute][block_value] = similarity

        if self.use_full_simvector or self.use_parfull_simvector:
            self.dataset[r_id] = r_attributes

        self.nrecords += 1

    @staticmethod
    def blocks(feature, dataset):
        """
            Builds all blocks for a specific feature. Returns the blocks build
            by this indexer. Each block only contains the record ids instead of
            the attributes.
        """
        RI = defaultdict(set)
        FBI = defaultdict(dict)
        for r_id, r_attributes in dataset.items():
            # Add id to RI index
            for field in feature.covered_fields():
                RI[r_attributes[field]].add(r_id)

            blocking_key_values = feature.blocking_key_values(r_attributes)
            for encoding in blocking_key_values:
                for blocking_key in feature.blocking_keys:
                    field = blocking_key.field
                    attribute = r_attributes[field]
                    if not attribute:
                        continue    # Do not block on empty attributes

                    BI = FBI[field]
                    if encoding not in BI.keys():
                        BI[encoding] = set()

                    BI[encoding].add(r_id)

        blocks = []
        for block_key in RI.keys():
            blocks.append((block_key, RI[block_key]))

        for BI in FBI.values():
            for block_key in BI.keys():
                blocks.append((block_key, BI[block_key]))

        return blocks

    def query(self, q_record):
        accumulator = defaultdict(partial(np.zeros, self.attribute_count, np.float))
        q_id = q_record[0]
        q_attributes = q_record[1:]

        #  Insert new record into index
        self.insert(q_id, q_attributes)

        for field, q_attribute in enumerate(q_attributes):
            if q_attribute in self.RI:
                for id in self.RI[q_attribute]:
                    accumulator[id][field] = 1.0

            if q_attribute in self.SI:
                for attribute, sim in self.SI[q_attribute].items():
                    for id in self.RI[attribute]:
                        accumulator[id][field] = sim

        if q_id in accumulator:
            del accumulator[q_id]

        if self.use_full_simvector:
            for id in accumulator.keys():
                for field, sim in enumerate(accumulator[id]):
                    if sim == 0 and q_attributes[field] and self.dataset[id][field]:
                        sim = self.similarity_fns[field](q_attributes[field], self.dataset[id][field])
                        accumulator[id][field] = sim

        elif self.use_parfull_simvector:
            for id in accumulator.keys():
                for field, sim in enumerate(accumulator[id]):
                    if field in self.fields and sim == 0 and q_attributes[field] and self.dataset[id][field]:
                        sim = self.similarity_fns[field](q_attributes[field], self.dataset[id][field])
                        accumulator[id][field] = sim

        return accumulator

    def save(self, name, datadir):
        # Dump number of records
        with open("%s/.%s_nrecords.idx" % (datadir, name), "wb") as handle:
            pickle.dump(self.nrecords, handle, protocol=pickle.HIGHEST_PROTOCOL)
        # Dump RI
        with open("%s/.%s_RI.idx" % (datadir, name), "wb") as handle:
            pickle.dump(self.RI, handle, protocol=pickle.HIGHEST_PROTOCOL)
        # Dump FBI
        with open("%s/.%s_FBI.idx" % (datadir, name), "wb") as handle:
            pickle.dump(self.FBI, handle, protocol=pickle.HIGHEST_PROTOCOL)
        # Dump SI
        with open("%s/.%s_SI.idx" % (datadir, name), "wb") as handle:
            pickle.dump(self.SI, handle, protocol=pickle.HIGHEST_PROTOCOL)

    def load(self, name, datadir):
        nrecords_filename = "%s/.%s_nrecords.idx" % (datadir, name)
        ri_filename = "%s/.%s_RI.idx" % (datadir, name)
        fbi_filename = "%s/.%s_FBI.idx" % (datadir, name)
        si_filename = "%s/.%s_SI.idx" % (datadir, name)
        if os.path.exists(nrecords_filename) and \
           os.path.exists(ri_filename) and \
           os.path.exists(fbi_filename) and \
           os.path.exists(si_filename):
            with open(nrecords_filename, "rb") as handle:
                self.nrecords = pickle.load(handle)
            with open(ri_filename, "rb") as handle:
                self.RI = pickle.load(handle)
            with open(fbi_filename, "rb") as handle:
                self.FBI = pickle.load(handle)
            with open(si_filename, "rb") as handle:
                self.SI = pickle.load(handle)

            return True
        else:
            return False

    def pair_completeness(self, gold_pairs, dataset):
        """
        Returns the pair completeness from the passed gold standard and the
        index data.
        """
        total_p = len(gold_pairs)  # True positives + False negatives
        true_p = 0
        for id1, id2 in gold_pairs:
            hit = False
            id1_attributes = dataset[id1]
            id2_attributes = dataset[id2]
            # Check RI
            for attributes in id1_attributes:
                if id2 in self.RI[attributes]:
                    hit = True
                    break

            # Check FBI
            if not hit:
                for feature in self.dns_blocking_scheme:
                    id1_bkvs = feature.blocking_key_values(id1_attributes)
                    id2_bkvs = feature.blocking_key_values(id2_attributes)
                    if len(id1_bkvs.intersection(id2_bkvs)) > 0:
                        hit = True
                        break

            if hit:
                true_p += 1

        return true_p / total_p

    def ri_distribution(self):
        block_sizes = []
        for block_key in self.RI.keys():
            block_sizes.append(len(self.RI[block_key]))

        return Counter(block_sizes)

    def frequency_distribution(self):
        """
        Returns the frequency distribution for the block index as dict. Where
        key is size a block and value number of blocks with this size.
        """
        block_sizes = []
        for BI in self.FBI.values():
            for block_key in BI.keys():
                block_sizes.append(len(BI[block_key]))

        return Counter(block_sizes)


class MDySimIII(object):

    def __init__(self, count, dns_blocking_scheme, similarity_fns,
                 use_full_simvector=False, use_parfull_simvector=False):
        self.dns_blocking_scheme = dns_blocking_scheme
        self.similarity_fns = similarity_fns

        # Class structure
        self.nrecords = 0              # Number of records indexed
        self.FBI = defaultdict(dict)   # Field Block Indicies (FBI)
        self.SI = defaultdict(dict)    # Similarity Index (SI)
        self.dataset = {}

        # Format output
        self.attribute_count = count

        self.use_full_simvector = use_full_simvector
        self.use_parfull_simvector = use_parfull_simvector

        if self.use_parfull_simvector:
            # Calculate similarites between all attributes covered by
            # the blocking schema
            self.fields = set()
            for blocking_key in self.dns_blocking_scheme:
                self.fields.update(blocking_key.covered_fields())

    def insert(self, r_id, r_attributes):
        for feature in self.dns_blocking_scheme:
            # Generate blocking keys for feature
            blocking_key_values = feature.blocking_key_values(r_attributes)
            # Insert record into block and calculate similarities
            for encoding in blocking_key_values:
                for blocking_key in feature.blocking_keys:
                    field = blocking_key.field
                    attribute = r_attributes[field]
                    if not attribute:
                        continue    # Do not block on empty attributes

                    BI = self.FBI[field]
                    if encoding not in BI.keys():
                        BI[encoding] = defaultdict(set)

                    BI[encoding][attribute].add(r_id)

                    #  Calculate similarities and update SI
                    block = filter(lambda x: x != attribute, BI[encoding].keys())
                    for block_value in block:
                        if block_value not in self.SI[attribute]:
                            similarity = self.similarity_fns[field](attribute, block_value)
                            #  Append similarity to block_value
                            self.SI[block_value][attribute] = similarity

                            #  Append similarity to attribute
                            self.SI[attribute][block_value] = similarity

        if self.use_full_simvector or self.use_parfull_simvector:
            self.dataset[r_id] = r_attributes

        self.nrecords += 1

    @staticmethod
    def blocks(feature, dataset):
        """
            Builds all blocks for a specific feature. Returns the blocks build
            by this indexer. Each block only contains the record ids instead of
            the attributes.
        """
        FBI = defaultdict(dict)
        for r_id, r_attributes in dataset.items():
            blocking_key_values = feature.blocking_key_values(r_attributes)
            for encoding in blocking_key_values:
                for blocking_key in feature.blocking_keys:
                    field = blocking_key.field
                    attribute = r_attributes[field]
                    if not attribute:
                        continue    # Do not block on empty attributes

                    BI = FBI[field]
                    if encoding not in BI.keys():
                        BI[encoding] = set()

                    BI[encoding].add(r_id)

        blocks = []
        for BI in FBI.values():
            for block_key in BI.keys():
                blocks.append((block_key, BI[block_key]))

        return blocks

    def query(self, q_record):
        q_id = q_record[0]
        q_attributes = q_record[1:]
        accumulator = defaultdict(partial(np.zeros, self.attribute_count, np.float))

        # Insert new record into index
        self.insert(q_id, q_attributes)

        # for q_attribute in q_attributes:
        for feature in self.dns_blocking_scheme:
            # Generate blocking keys for feature
            blocking_key_values = feature.blocking_key_values(q_attributes)
            for encoding in blocking_key_values:
                for field in feature.covered_fields():
                    BI = self.FBI[field]
                    q_attribute = q_attributes[field]
                    if not q_attribute:
                        continue    # Empty attributes are not blocked

                    for attribute in BI[encoding].keys():
                        # print("Added %d for %r in %r" %
                                # (len(BI[encoding][attribute]), attribute, encoding))
                        if attribute == q_attribute:
                            for id in BI[encoding][q_attribute]:
                                accumulator[id][field] = 1.0
                        else:
                            sim = self.SI[q_attribute][attribute]
                            for id in BI[encoding][attribute]:
                                accumulator[id][field] = sim

        if q_id in accumulator:
            del accumulator[q_id]

        if self.use_full_simvector:
            for id in accumulator.keys():
                for field, sim in enumerate(accumulator[id]):
                    if sim == 0 and q_attributes[field] and self.dataset[id][field]:
                        sim = self.similarity_fns[field](q_attributes[field], self.dataset[id][field])
                        accumulator[id][field] = sim

        elif self.use_parfull_simvector:
            for id in accumulator.keys():
                for field, sim in enumerate(accumulator[id]):
                    if field in self.fields and sim == 0 and q_attributes[field] and self.dataset[id][field]:
                        sim = self.similarity_fns[field](q_attributes[field], self.dataset[id][field])
                        accumulator[id][field] = sim

        return accumulator

    def save(self, name, datadir):
        # Dump number of records
        with open("%s/.%s_nrecords.idx" % (datadir, name), "wb") as handle:
            pickle.dump(self.nrecords, handle, protocol=pickle.HIGHEST_PROTOCOL)
        # Dump FBI
        with open("%s/.%s_FBI.idx" % (datadir, name), "wb") as handle:
            pickle.dump(self.FBI, handle, protocol=pickle.HIGHEST_PROTOCOL)
        # Dump SI
        with open("%s/.%s_SI.idx" % (datadir, name), "wb") as handle:
            pickle.dump(self.SI, handle, protocol=pickle.HIGHEST_PROTOCOL)

    def load(self, name, datadir):
        nrecords_filename = "%s/.%s_nrecords.idx" % (datadir, name)
        fbi_filename = "%s/.%s_FBI.idx" % (datadir, name)
        si_filename = "%s/.%s_SI.idx" % (datadir, name)
        if os.path.exists(nrecords_filename) and \
           os.path.exists(fbi_filename) and \
           os.path.exists(si_filename):
            with open(nrecords_filename, "rb") as handle:
                self.nrecords = pickle.load(handle)
            with open(fbi_filename, "rb") as handle:
                self.FBI = pickle.load(handle)
            with open(si_filename, "rb") as handle:
                self.SI = pickle.load(handle)

            return True
        else:
            return False

    def pair_completeness(self, gold_pairs, dataset):
        """
        Returns the pair completeness from the passed gold standard and the
        index data.
        """
        total_p = len(gold_pairs)  # True positives + False negatives
        true_p = 0
        for id1, id2 in gold_pairs:
            id1_attributes = dataset[id1]
            id2_attributes = dataset[id2]
            for feature in self.dns_blocking_scheme:
                id1_bkvs = feature.blocking_key_values(id1_attributes)
                id2_bkvs = feature.blocking_key_values(id2_attributes)
                if len(id1_bkvs.intersection(id2_bkvs)) > 0:
                    true_p += 1
                    break

        return true_p / total_p

    def ri_distribution(self):
        block_sizes = []
        for block_key in self.RI.keys():
            block_sizes.append(len(self.RI[block_key]))

        return Counter(block_sizes)

    def frequency_distribution(self):
        """
        Returns the frequency distribution for the block index as dict. Where
        key is size a block and value number of blocks with this size.
        """
        block_sizes = []
        for BI in self.FBI.values():
            for block_key in BI.keys():
                block_sizes.append(len(BI[block_key]))

        return Counter(block_sizes)
