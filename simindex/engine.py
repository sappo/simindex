# -*- coding: utf-8 -*-
import pandas as pd
from pprint import pprint

from .dysim import MDySimII
from .weak_labels import WeakLabels, \
                         DisjunctiveBlockingScheme, \
                         BlockingKey, \
                         Feature, \
                         SimTupel, \
                         has_common_token, \
                         is_exact_match
from simindex.similarity import SimLearner
import simindex.helper as hp


class SimEngine(object):

    def __init__(self, name, indexer=MDySimII,
                 max_positive_labels=None, max_negative_labels=None):
        self.name = name
        self.configstore_name = ".%s_config.h5" % self.name
        self.traindatastore_name = ".%s_traindata.h5" % self.name
        self.indexdatastore_name = ".%s_indexdata.h5" % self.name
        self.querydatastore_name = ".%s_querydata.h5" % self.name
        self.indexer = indexer
        self.attribute_count = None
        self.stoplist = set('for a of the and to in'.split())

        self.max_p = max_positive_labels
        self.max_n = max_negative_labels

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

    def pre_process_data(self, store, csvfile, attributes):
        group_name = "/%s" % self.name
        if group_name not in store.keys():
            # Read CSV in chunks
            csv_frames = pd.read_csv(csvfile,
                                     usecols=attributes,
                                     skipinitialspace=True,
                                     iterator=True,
                                     chunksize=50000,
                                     error_bad_lines=False,
                                     index_col=0,
                                     dtype='unicode',
                                     encoding='utf-8')

            expectedrows = hp.file_len(csvfile)
            # Store each chunk in hdf5 store - without indexing
            for dataframe in csv_frames:
                dataframe.fillna('', inplace=True)
                # Pre-process data
                dataframe.applymap(lambda x: x if type(x) != str else x.lower())
                dataframe.applymap(lambda x: x if type(x) != str else \
                        ' '.join(filter(lambda s: s not in self.stoplist, x.split())))

                # Append chunk to store
                store.append(self.name, dataframe, format='table', index=False,
                             data_columns=True, min_itemsize=255,
                             expectedrows=expectedrows,
                             complib='blosc', complevel=9)
                self.attribute_count = len(dataframe.columns)
                del dataframe

        # Create index on index column for the whole dataset
        store.create_table_index(self.name, columns=['index'], optlevel=9, kind='full')


    def fit_csv(self, train_file, attributes=None):
        store = pd.HDFStore(self.traindatastore_name, mode="w")
        self.pre_process_data(store, train_file, attributes)
        store.close()
        self.fit(hp.hdf_records(store, self.name))

    def fit(self, records):
        dataset = {}
        for record in records:
            if self.attribute_count == None:
                self.attribute_count = len(record[1:])
            r_id = record[0]
            r_attributes = record[1:]
            dataset[r_id] = r_attributes

        P, N = self.load_labels()
        if P is None and N is None:
            if self.max_p is None:
                self.max_p = int(len(dataset) * 0.1)

            if self.max_n is None:
                self.max_n = int(len(dataset) * 0.25)

            labels = WeakLabels(self.attribute_count,
                                max_positive_pairs=self.max_p,
                                max_negative_pairs=self.max_n)
            labels.fit(dataset)
            P, N = labels.predict()
            self.save_labels(P, N)

        self.blocking_key = self.load_blocking_scheme()
        if self.blocking_key is None:
            blocking_keys = []
            for field in range(self.attribute_count):
                blocking_keys.append(BlockingKey(has_common_token, field, str.split))
                blocking_keys.append(BlockingKey(is_exact_match, field, lambda x: [x]))

            dbs = DisjunctiveBlockingScheme(blocking_keys, P, N)
            self.blocking_key = dbs.transform(dataset)
            self.save_blocking_scheme()

        pprint(self.blocking_key)

        self.similarities = self.load_similarities()
        if self.similarities is None:
            P, N = WeakLabels.filter(self.blocking_key, P, N)
            sl = SimLearner(dataset)
            self.similarities = sl.predict(P, N)
            self.save_similarities()

        pprint(self.similarities)

        # Cleanup
        del dataset

    def build_csv(self, index_file, attributes=None):
        store = pd.HDFStore(self.indexdatastore_name, mode="w")
        self.pre_process_data(store, index_file, attributes)
        store.close()
        self.build(hp.hdf_records(store, self.name))

    def build(self, records):
        self.indexer = MDySimII(self.attribute_count,
                                self.blocking_key,
                                self.similarities)
        if not self.indexer.msai.load(self.name):
            for record in records:
                r_id = record[0]
                r_attributes = record[1:]
                self.indexer.insert(r_id, r_attributes)

            self.indexer.msai.save(self.name)

    def query_csv(self, query_file, attributes=None):
        store = pd.HDFStore(self.querydatastore_name, mode="w")
        self.pre_process_data(store, query_file, attributes)
        store.close()
        self.query(hp.hdf_records(store, self.name))

    def query(self, records):
        for record in records:
            self.indexer.query(record)

    def save_labels(self, P, N):
        with pd.HDFStore(self.configstore_name,
                         complevel=9, complib='blosc') as store:
            df_p = pd.DataFrame(P, columns=["t1", "t2", "sim"])
            df_n = pd.DataFrame(N, columns=["t1", "t2", "sim"])
            store.put("labels_p", df_p, format="t")
            store.put("labels_n", df_n, format="t")

    def load_labels(self):
        with pd.HDFStore(self.configstore_name) as store:
            if "/labels_p" in store.keys() and "/labels_n" in store.keys():
                P = []
                N = []
                for label in hp.hdf_record_attributes(store, "labels_p"):
                    P.append(SimTupel(label[0], label[1], label[2]))
                for label in hp.hdf_record_attributes(store, "labels_n"):
                    N.append(SimTupel(label[0], label[1], label[2]))

                return P, N
            else:
                return None, None

    def blocking_scheme_to_strings(self):
        data = []
        for index, feature in enumerate(self.blocking_key):
            for predicate in feature.predicates:
                line = [index, predicate.field]
                if predicate.encoder == str.split:
                    line.append("str")
                else:
                    line.append("id")

                if predicate.predicate == has_common_token:
                    line.append("common_token")
                elif predicate.predicate == is_exact_match:
                    line.append("exact_match")

                data.append(line)

        return data


    def save_blocking_scheme(self):
        with pd.HDFStore(self.configstore_name,
                         complevel=9, complib='blosc') as store:
            data = self.blocking_scheme_to_strings()
            df = pd.DataFrame(data, columns=["feature", "field",
                                             "encoder", "predicate"])
            store.put("blocking_scheme", df, format="t")

    def load_blocking_scheme(self):
        with pd.HDFStore(self.configstore_name) as store:
            if "/blocking_scheme" in store.keys():
                id = None
                blocking_key = []
                for predicate in hp.hdf_record_attributes(store,
                                                          "blocking_scheme"):
                    if id != predicate[0]:
                        blocking_key.append(Feature([], 0, 0))

                    feature = blocking_key[predicate[0]]
                    id = predicate[0]

                    encoder = None
                    if predicate[2] == "str":
                        encoder = str.split
                    elif predicate[2] == "id":
                        encoder = (lambda x: [0])

                    pred = None
                    if predicate[3] == "common_token":
                        pred = has_common_token
                    elif predicate[3] == "exact_match":
                        pred = is_exact_match

                    feature.predicates.append(BlockingKey(pred,
                                                          predicate[1],
                                                          encoder))

                return blocking_key
            else:
                return None

    def save_similarities(self):
        with pd.HDFStore(self.configstore_name,
                         complevel=9, complib='blosc') as store:
            df = pd.DataFrame(SimLearner.prediction_to_strings(self.similarities),
                              columns=["function"])
            store.put("similarities", df, format="t")

    def load_similarities(self):
        with pd.HDFStore(self.configstore_name) as store:
            if "/similarities" in store.keys():
                prediction_strings = \
                    [s[0] for s in hp.hdf_record_attributes(store,
                                                            "similarities")]
                return SimLearner.strings_to_prediction(prediction_strings)
            else:
                return None
