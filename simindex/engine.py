# -*- coding: utf-8 -*-
import os
import pickle
import subprocess
import pandas as pd
import numpy as np
import json
from pprint import pprint

from collections import defaultdict, Counter
import sklearn as skm

from .dysim import MDySimIII
from .weak_labels import WeakLabels, \
                         DisjunctiveBlockingScheme, \
                         BlockingKey, \
                         Feature, \
                         SimTupel, \
                         tokens, \
                         term_id
from .similarity import SimLearner
from .fusionlearner import FusionLearner
import simindex.helper as hp
import logging

logger = logging.getLogger(__name__)

try:
    profile
except NameError as e:
    def profile(func):
        def inner(*args, **kwargs):
            return func(*args, **kwargs)
        return inner


def file_len(fname):
    p = subprocess.Popen(['wc', '-l', fname], stdout=subprocess.PIPE,
                                              stderr=subprocess.PIPE)
    result, err = p.communicate()
    if p.returncode != 0:
        raise IOError(err)
    return int(result.strip().split()[0])


class SimEngine(object):

    def __init__(self, name, datadir='.',
                 indexer=MDySimIII, classifier_candidates=None,
                 max_positive_labels=None, max_negative_labels=None,
                 max_bk_conjunction=2,
                 insert_timer=None, query_timer=None,
                 use_classifier=True, use_full_simvector=False,
                 verbose=False):
        self.name = name
        self.datadir = datadir.strip('/')
        os.makedirs(self.datadir, exist_ok=True)
        self.configstore_name = "%s/.%s_config.h5" % (self.datadir, self.name)
        self.traindatastore_name = "%s/.%s_traindata.h5" % (self.datadir, self.name)
        self.indexdatastore_name = "%s/.%s_indexdata.h5" % (self.datadir, self.name)
        self.querydatastore_name = "%s/.%s_querydata.h5" % (self.datadir, self.name)
        self.indexer_class = indexer
        self.classifier_candidates = classifier_candidates
        self.attribute_count = None
        self.stoplist = set('for a of the and to in'.split())

        # WeakLabels parameters
        self.max_p = max_positive_labels
        self.max_n = max_negative_labels
        self.nfP = 0
        self.nfN = 0

        # DNF Blocking Scheme Learner parameters
        self.max_bk_conjunction = max_bk_conjunction

        # FusionLearner parameters
        self.clf = None
        self.clf_best_params = None
        self.clf_best_score = None

        # Evaluation attributes
        self.true_matches = 0
        self.true_nonmatches = 0
        self.total_matches = 0
        self.total_nonmatches = 0
        self.insert_timer = insert_timer
        self.query_timer = query_timer
        self.use_classifier = use_classifier
        self.use_full_simvector = use_full_simvector
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
                             complib='blosc', complevel=1)
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
        # Load Data
        dataset = {}
        for record in records:
            if self.attribute_count is None:
                self.attribute_count = len(record[1:])
            r_id = record[0]
            r_attributes = record[1:]
            dataset[r_id] = r_attributes

        if self.verbose:
            logger.info("Dataset has %d records" % len(dataset))

        # Predict labels
        P, N = self.load_labels()
        if P is None and N is None:
            if not self.gold_pairs:
                if self.max_p is None:
                    self.max_p = int(len(dataset) * 0.1)

                if self.max_n is None:
                    self.max_n = int(len(dataset) * 0.25)

            labels = WeakLabels(self.attribute_count,
                                gold_pairs=self.gold_pairs,
                                max_positive_pairs=self.max_p,
                                max_negative_pairs=self.max_n,
                                verbose=self.verbose)
            labels.fit(dataset)
            P, N = labels.predict()
            if self.verbose:
                logger.info("Saving labels now")

            self.save_labels(P, N)
            del labels

        if self.verbose:
            logger.info("Generated %d P and %d N labels" % (len(P), len(N)))
            self.nP = len(P)
            self.nN = len(N)

        # Learn blocking scheme
        self.blocking_scheme = self.load_blocking_scheme()
        if self.blocking_scheme is None:
            blocking_keys = []
            for field in range(self.attribute_count):
                blocking_keys.append(BlockingKey(field, tokens))
                blocking_keys.append(BlockingKey(field, term_id))

            dbs = DisjunctiveBlockingScheme(blocking_keys, P, N,
                                            self.indexer_class,
                                            self.max_bk_conjunction,
                                            verbose=self.verbose)
            self.blocking_scheme = dbs.transform(dataset)
            if self.verbose:
                logger.info("Saving blocking scheme now")

            self.save_blocking_scheme()
            del dbs

        if self.verbose:
            logger.info("Learned the following blocking scheme:")
            pprint(self.blocking_scheme)

        # Learn similarity functions per attribute
        self.similarities = self.load_similarities()
        if self.similarities is None:
            P, N = WeakLabels.filter(self.blocking_scheme, P, N)
            if self.verbose:
                logger.info("Have %d P and %d N filtered labels" % (len(P), len(N)))
                self.nfP = len(P)
                self.nfN = len(N)

            sl = SimLearner(self.attribute_count, dataset)
            self.similarities = sl.predict(P, N)
            self.save_similarities()
            del sl

        if self.verbose:
            logger.info("Predicted the following similarities:")
            pprint(self.similarities)

        # Train classifier
        if self.use_classifier:
            self.clf = self.load_model()

        if self.clf is None and self.use_classifier:
            similarity_fns = []
            for measure in SimLearner.strings_to_prediction(self.similarities):
                similarity_fns.append(measure().compare)

            X_P = []
            y_P = []
            for pair in P:
                x = np.zeros(self.attribute_count, np.float)
                for blocking_key in self.blocking_scheme:
                    p1_attributes = dataset[pair.t1]
                    p2_attributes = dataset[pair.t2]
                    if self.use_full_simvector:
                        # Calculate similarities between all attributes
                        for field, (p1_attribute, p2_attribute) in enumerate(zip(p1_attributes, p2_attributes)):
                            if p1_attribute and p2_attribute:
                                x[field] = similarity_fns[field](p1_attribute, p2_attribute)
                    else:
                        # Calculate similarites between pairs whose attributes
                        # have a common block.
                        p1_bkvs = blocking_key.blocking_key_values(p1_attributes)
                        p2_bkvs = blocking_key.blocking_key_values(p2_attributes)
                        if not p1_bkvs.isdisjoint(p2_bkvs):
                            for field in blocking_key.covered_fields():
                                p1_attribute = p1_attributes[field]
                                p2_attribute = p2_attributes[field]
                                if p1_attribute and p2_attribute:
                                    x[field] = similarity_fns[field](p1_attribute, p2_attribute)

                X_P.append(x)
                y_P.append(1)

            X_N = []
            y_N = []
            for pair in N:
                x = np.zeros(self.attribute_count, np.float)
                for blocking_key in self.blocking_scheme:
                    p1_attributes = dataset[pair.t1]
                    p2_attributes = dataset[pair.t2]
                    if self.use_full_simvector:
                        # Calculate similarities between all attributes
                        for field, (p1_attribute, p2_attribute) in enumerate(zip(p1_attributes, p2_attributes)):
                            if p1_attribute and p2_attribute:
                                x[field] = similarity_fns[field](p1_attribute, p2_attribute)
                    else:
                        # Calculate similarites between pairs whose attributes
                        # have a common block.
                        p1_bkvs = blocking_key.blocking_key_values(p1_attributes)
                        p2_bkvs = blocking_key.blocking_key_values(p2_attributes)
                        if not p1_bkvs.isdisjoint(p2_bkvs):
                            for field in blocking_key.covered_fields():
                                p1_attribute = p1_attributes[field]
                                p2_attribute = p2_attributes[field]
                                if p1_attribute and p2_attribute:
                                    x[field] = similarity_fns[field](p1_attribute, p2_attribute)

                X_N.append(x)
                y_N.append(0)

            # Shrink training set to max 5000 (max 1000 P, max 4000 N) samples
            X_train = []
            y_train = []
            P_samples = min(1000, len(X_P))
            choices = np.random.choice(np.arange(len(X_P)), size=P_samples, replace=False)
            for choice in choices:
                X_train.append(X_P[choice])
                y_train.append(y_P[choice])

            N_samples = min(4000, len(X_N))
            choices = np.random.choice(np.arange(len(X_N)), size=N_samples, replace=False)
            for choice in choices:
                X_train.append(X_N[choice])
                y_train.append(y_N[choice])

            X_P.extend(X_N)
            y_P.extend(y_N)
            X = X_P
            y = y_P
            assert len(X_train) < 5001
            assert len(y_train) < 5001

            # Train the best model
            ful = FusionLearner(FusionLearner.candidate_families())
            self.clf = ful.best_model(X_train, y_train)
            self.clf_best_params = ful.best_params
            self.clf_best_score = ful.best_quality
            self.clf.fit(X, y)  # refit classifier with whole ground truth
            self.save_model()

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

        if self.use_full_simvector:
            dataset = {}
            for record in hp.hdf_records(pd.HDFStore(self.traindatastore_name, mode="r"), self.name):
                if self.attribute_count is None:
                    self.attribute_count = len(record[1:])
                r_id = record[0]
                r_attributes = record[1:]
                dataset[r_id] = r_attributes

            self.indexer = self.indexer_class(self.attribute_count,
                                              self.blocking_scheme,
                                              similarity_fns,
                                              dataset=dataset)
        else:
            self.indexer = self.indexer_class(self.attribute_count,
                                              self.blocking_scheme,
                                              similarity_fns)

        if not self.indexer.load(self.name, self.datadir):
            for record in records:
                r_id = record[0]
                r_attributes = record[1:]
                if self.insert_timer:
                    with self.insert_timer:
                        self.indexer.insert(r_id, r_attributes)
                else:
                    self.indexer.insert(r_id, r_attributes)

            self.indexer.save(self.name, self.datadir)

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

            # Calculate complexity metrics
            if self.gold_records:
                q_id = q_record[0]
                matches_count = hp.matches_count(q_id, result, self.gold_records)
                all_matches_count = hp.all_matches_count(q_id, self.gold_records)
                self.true_matches += matches_count
                self.true_nonmatches += len(result) - matches_count
                self.total_matches += all_matches_count
                self.total_nonmatches += self.indexer.nrecords - 1 - all_matches_count

            # Apply classifier
            if self.use_classifier:
                for candidate in list(result.keys()):
                    prediction = self.clf.predict([result[candidate]])[0]
                    if prediction == 0:
                        del result[candidate]

            # Calculate quality metrics
            if self.gold_records:
                q_id = q_record[0]
                hp.calc_micro_scores(q_id, result, self.y_true_score,
                                     self.y_scores, self.gold_records)
                hp.calc_micro_metrics(q_id, result, self.y_true,
                                      self.y_pred, self.gold_records)

            return result


    def read_ground_truth(self, gold_standard, gold_attributes):
        if self.verbose:
            logger.info("Reading ground truth")

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

    def pairs_completeness(self):
        if self.gold_pairs:
            return self.true_matches / self.total_matches

        with pd.HDFStore(self.traindatastore_name) as store:
            gold_ids = list(hp.flatten(self.gold_pairs))
            query = "index == %r" % gold_ids
            dataset = {}
            for record in hp.hdf_records(store, self.name, query):
                dataset[record[0]] = record[1:]

        return self.indexer.pair_completeness(self.gold_pairs, dataset)

    def pairs_quality(self):
        return self.true_matches / (self.true_matches + self.true_nonmatches)

    def reduction_ratio(self):
        return 1 - (
                    (self.true_matches + self.true_nonmatches) /
                    (self.total_matches + self.total_nonmatches)
                   )

    def recall(self):
        return skm.metrics.recall_score(self.y_true, self.y_pred)

    def precision(self):
        return skm.metrics.precision_score(self.y_true, self.y_pred)

    def f1_score(self):
        return skm.metrics.f1_score(self.y_true, self.y_pred)

    def precision_recall_curve(self):
        return skm.metrics.precision_recall_curve(self.y_true_score, self.y_scores)

    def roc_curve(self):
        return skm.metrics.roc_curve(self.y_true_score, self.y_scores)

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

    def set_baseline(self, baseline):
        self.blocking_scheme = self.read_blocking_scheme(baseline['scheme'])
        self.similarities = baseline['similarities']
        self.save_blocking_scheme()
        self.save_similarities()

    def blocking_scheme_to_strings(self):
        data = []
        for index, feature in enumerate(self.blocking_scheme):
            for predicate in feature.blocking_keys:
                data.append([index, predicate.field,
                             predicate.encoder.__name__,
                             feature.fsc])

        return data

    def save_blocking_scheme(self):
        with open("%s/.%s_dnfbs.inc" % (self.datadir, self.name), "w") as handle:
            data = {}
            for index, feature in enumerate(self.blocking_scheme):
                data[index] = feature.to_json()

            json.dump(data, handle, sort_keys=True, indent=4)

    @staticmethod
    def read_blocking_scheme(blocking_keys):
        id = None
        blocking_scheme = []
        possibles = globals().copy()
        possibles.update(locals())
        for predicate in blocking_keys:
            if id != predicate[0]:
                blocking_scheme.append(Feature([], 0))

            feature = blocking_scheme[predicate[0]]
            id = predicate[0]

            encoder = possibles.get(predicate[2])
            feature.blocking_keys.append(BlockingKey(predicate[1], encoder))

        return blocking_scheme

    def load_blocking_scheme(self):
        if os.path.exists("%s/.%s_dnfbs.inc" % (self.datadir, self.name)):
            with open("%s/.%s_dnfbs.inc" % (self.datadir, self.name), "r") as handle:
                data = json.load(handle)
                return [Feature.from_json(b) for b in data.values()]
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

    def save_model(self):
        with open("%s/.%s_model.cls" % (self.datadir, self.name), "wb") as handle:
            pickle.dump(self.clf, handle, protocol=pickle.HIGHEST_PROTOCOL)

    def load_model(self):
        model_filename = "%s/.%s_model.cls" % (self.datadir, self.name)
        if os.path.exists(model_filename):
            with open(model_filename, "rb") as handle:
                return pickle.load(handle)
        else:
            return None
