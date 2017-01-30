# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
from pprint import pprint

from collections import defaultdict, Counter
import sklearn.metrics as skm
import redislite
import redis_collections

from .dysim import MDySimIII
from .weak_labels import WeakLabels, \
                         DisjunctiveBlockingScheme, \
                         BlockingKey, \
                         Feature, \
                         SimTupel, \
                         tokens, \
                         has_common_token, \
                         term_id, \
                         is_exact_match
from simindex.similarity import SimLearner
import simindex.helper as hp

try:
    profile
except NameError as e:
    def profile(func):
        def inner(*args, **kwargs):
            return func(*args, **kwargs)
        return inner


class SimEngine(object):

    def __init__(self, name, indexer=MDySimIII, max_bk_conjunction=1,
                 max_positive_labels=None, max_negative_labels=None,
                 threshold=0.0, top_n=0,
                 insert_timer=None, query_timer=None, verbose=False):
        self.name = name
        self.configstore_name = ".%s_config.h5" % self.name
        self.traindatastore_name = ".%s_traindata.h5" % self.name
        self.indexdatastore_name = ".%s_indexdata.h5" % self.name
        self.querydatastore_name = ".%s_querydata.h5" % self.name
        self.indexer_class = indexer
        self.attribute_count = None
        self.stoplist = set('for a of the and to in'.split())
        self.max_bk_conjunction = max_bk_conjunction

        self.max_p = max_positive_labels
        self.max_n = max_negative_labels
        self.threshold = threshold
        self.top_n = top_n

        self.reductionratio = None
        self.insert_timer = insert_timer
        self.query_timer = query_timer
        self.verbose = verbose

        # Gold Standard/Ground Truth attributes
        self.gold_pairs = None
        self.gold_records = None

    @profile
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
                dataframe = dataframe.applymap(lambda x: x if type(x) != str else x.lower())
                # Pre-process data
                dataframe = dataframe.applymap(lambda x: x if type(x) != str else \
                        ' '.join(filter(lambda s: s not in self.stoplist, x.split())))

                # Append chunk to store
                store.append(self.name, dataframe, format='table', index=False,
                             data_columns=True, min_itemsize=255,
                             expectedrows=expectedrows,
                             complib='blosc', complevel=9)
                self.attribute_count = len(dataframe.columns)
                self.index_dtype = dataframe.index.dtype
                del dataframe

        # Create index on index column for the whole dataset
        store.create_table_index(self.name, columns=['index'], optlevel=9, kind='full')

    def fit_csv(self, train_file, attributes=None):
        store = pd.HDFStore(self.traindatastore_name, mode="w")
        self.pre_process_data(store, train_file, attributes)
        store.close()
        self.fit(hp.hdf_records(store, self.name))

    @profile
    def fit(self, records):
        dataset = {}
        for record in records:
            if self.attribute_count is None:
                self.attribute_count = len(record[1:])
            r_id = record[0]
            r_attributes = record[1:]
            dataset[r_id] = r_attributes

        if self.verbose:
            print("Dataset has %d records" % len(dataset))

        #  Predict labels
        P, N = self.load_labels()
        if P is None and N is None:
            if self.max_p is None:
                self.max_p = int(len(dataset) * 0.1)

            if self.max_n is None:
                self.max_n = int(len(dataset) * 0.25)

            labels = WeakLabels(self.attribute_count,
                                gold_pairs=self.gold_pairs,
                                max_positive_pairs=self.max_p,
                                max_negative_pairs=self.max_n)
            labels.fit(dataset)
            P, N = labels.predict()
            self.save_labels(P, N)
            del labels

        if self.verbose:
            print("Generated %d P and %d N labels" % (len(P), len(N)))

        # Learn blocking scheme
        self.blocking_scheme = self.load_blocking_scheme()
        if self.blocking_scheme is None:
            blocking_keys = []
            for field in range(self.attribute_count):
                blocking_keys.append(BlockingKey(has_common_token, field, tokens))
                blocking_keys.append(BlockingKey(is_exact_match, field, term_id))

            dbs = DisjunctiveBlockingScheme(blocking_keys, P, N,
                                            self.max_bk_conjunction)
            self.blocking_scheme = dbs.transform(dataset)
            self.save_blocking_scheme()
            del dbs

        if self.verbose:
            print("Learned the following blocking scheme:")
            pprint(self.blocking_scheme)

        # Learn similarity functions per attribute
        self.similarities = self.load_similarities()
        if self.similarities is None:
            P, N = WeakLabels.filter(self.blocking_scheme, P, N)
            if self.verbose:
                print("Have %d P and %d N filtered labels" % (len(P), len(N)))

            sl = SimLearner(self.attribute_count, dataset)
            self.similarities = sl.predict(P, N)
            self.save_similarities()
            del sl

        if self.verbose:
            print("Predicted the following similarities:")
            pprint(self.similarities)

        # Cleanup
        del dataset
        del P, N

    def build_csv(self, index_file, attributes=None):
        store = pd.HDFStore(self.indexdatastore_name, mode="w")
        self.pre_process_data(store, index_file, attributes)
        store.close()
        self.build(hp.hdf_records(store, self.name))

    @profile
    def build(self, records):
        similarity_fns = []
        for measure in SimLearner.strings_to_prediction(self.similarities):
            similarity_fns.append(measure().compare)

        self.indexer = self.indexer_class(self.attribute_count,
                                          self.blocking_scheme,
                                          similarity_fns)
        if not self.indexer.load(self.name):
            for record in records:
                r_id = record[0]
                r_attributes = record[1:]
                if self.insert_timer:
                    with self.insert_timer:
                        self.indexer.insert(r_id, r_attributes)
                else:
                    self.indexer.insert(r_id, r_attributes)

            self.indexer.save(self.name)

        # pprint(self.indexer.FBI)

    @profile
    def query_csv(self, query_file, attributes=None):
        store = pd.HDFStore(self.querydatastore_name, mode="w")
        self.pre_process_data(store, query_file, attributes)
        store.close()

        for q_record in hp.hdf_records(store, self.name):
            if self.query_timer:
                with self.query_timer:
                    result = self.query(q_record)
            else:
                result = self.query(q_record)

    def query(self, q_record):
            # Run Query
            result = self.indexer.query(q_record)

            # Calculate reduction ratio
            query_reduction_ratio = 1 - len(result)/self.indexer.nrecords
            if self.reductionratio:
                self.reductionratio = np.mean([self.reductionratio,
                                                query_reduction_ratio])
            else:
                self.reductionratio = query_reduction_ratio

            # Apply filters
            if self.threshold > 0:
                result = {k: v for k, v in result.items() if v > self.threshold}

            if self.top_n > 0:
                result = dict(Counter(result).most_common(self.top_n))

            # Calculate ground truth metrics
            if self.gold_records:
                q_id = q_record[0]
                hp.calc_micro_scores(q_id, result, self.y_true_score,
                                     self.y_scores, self.gold_records)
                hp.calc_micro_metrics(q_id, result, self.y_true,
                                      self.y_pred, self.gold_records)

            return result


    def read_ground_truth(self, gold_standard, gold_attributes):
        self.gold_pairs = set()
        self.gold_records = defaultdict(set)

        pairs = hp.read_csv(gold_standard, gold_attributes)#, dtype=self.index_dtype)
        for gold_pair in pairs:
            self.gold_pairs.add((gold_pair[0], gold_pair[1]))

            for a, b in self.gold_pairs:
                self.gold_records[a].add(b)
                self.gold_records[b].add(a)

        # Initialize evaluation
        self.y_true_score = []
        self.y_scores = []
        self.y_true = []
        self.y_pred = []

    def pair_completeness(self):
        with pd.HDFStore(self.traindatastore_name) as store:
            gold_ids = list(hp.flatten(self.gold_pairs))
            query = "index == %r" % gold_ids
            dataset = {}
            for record in hp.hdf_records(store, self.name, query):
                dataset[record[0]] = record[1:]

        return self.indexer.pair_completeness(self.gold_pairs, dataset)

    def reduction_ratio(self):
        return self.reductionratio

    def recall(self):
        return skm.recall_score(self.y_true, self.y_pred)

    def precision(self):
        return skm.precision_score(self.y_true, self.y_pred)

    def f1_score(self):
        return skm.f1_score(self.y_true, self.y_pred)

    def precision_recall_curve(self):
        return skm.precision_recall_curve(self.y_true_score, self.y_scores)

    def roc_curve(self):
        return skm.roc_curve(self.y_true_score, self.y_scores)

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
        for index, feature in enumerate(self.blocking_scheme):
            for predicate in feature.predicates:
                data.append([index, predicate.field,
                             predicate.encoder.__name__,
                             predicate.predicate.__name__])

        return data

    def save_blocking_scheme(self):
        with pd.HDFStore(self.configstore_name,
                         complevel=9, complib='blosc') as store:
            data = self.blocking_scheme_to_strings()
            df = pd.DataFrame(data, columns=["feature", "field",
                                             "encoder", "predicate"])
            store.put("blocking_scheme", df, format="t")

    def load_blocking_scheme(self):
        possibles = globals().copy()
        possibles.update(locals())
        with pd.HDFStore(self.configstore_name) as store:
            if "/blocking_scheme" in store.keys():
                id = None
                blocking_scheme = []
                for predicate in hp.hdf_record_attributes(store, "blocking_scheme"):
                    if id != predicate[0]:
                        blocking_scheme.append(Feature([], 0, 0))

                    feature = blocking_scheme[predicate[0]]
                    id = predicate[0]

                    encoder = possibles.get(predicate[2])
                    pred = possibles.get(predicate[3])
                    feature.predicates.append(BlockingKey(pred,
                                                          predicate[1],
                                                          encoder))
                return blocking_scheme
            else:
                return None

    def save_similarities(self):
        with pd.HDFStore(self.configstore_name,
                         complevel=9, complib='blosc') as store:
            df = pd.DataFrame(self.similarities, columns=["function"])
            store.put("similarities", df, format="t")

    def load_similarities(self):
        with pd.HDFStore(self.configstore_name) as store:
            if "/similarities" in store.keys():
                return [s[0] for s in hp.hdf_record_attributes(store,
                                                               "similarities")]
            else:
                return None
