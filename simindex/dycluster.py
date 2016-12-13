# -*- coding: utf-8 -*-
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import Normalizer
from sklearn import metrics
from sklearn.metrics import silhouette_samples, silhouette_score
from sklearn.cluster import KMeans
from datasketch import MinHash, MinHashLSH
from time import time
import numpy as np
from collections import Counter
from .helper import read_csv, calc_micro_scores, calc_micro_metrics
from .plot import *


class DyKMeans(object):

    def __init__(self, encode_fn, similarity_fn, threshold=-1.0, top_n=-1.0,
                 gold_standard=None, gold_attributes=None, n_features=10000,
                 use_hashing=False, use_idf=True, k=-1, verbose=False,
                 insert_timer=None, query_timer=None):
        self.verbose = verbose
        self.insert_timer = insert_timer
        self.query_timer = query_timer
        self.saai = SimAwareAttributeIndex(encode_fn, similarity_fn, threshold,
                                           top_n, verbose=verbose)
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
        # K-Means Attributes
        self.n_features = n_features
        self.use_hashing = use_hashing
        self.use_idf = use_idf
        self.k = k

    def fit_csv(self, recordset, attributes):
        self.fit(read_csv(recordset, attributes))

    def fit(self, records):
        if self.insert_timer:
            self.insert_timer.start_common()

        if self.use_hashing:
            if self.use_idf:
                # Perform an IDF normalization on the output of
                # HashingVectorizer
                hasher = HashingVectorizer(n_features=self.n_features,
                                           stop_words='english',
                                           non_negative=True,
                                           norm=None, binary=False)
                self.vectorizer = make_pipeline(hasher, TfidfTransformer())
            else:
                self.vectorizer = HashingVectorizer(n_features=self.n_features,
                                                    stop_words='english',
                                                    non_negative=False,
                                                    norm='l2',
                                                    binary=False)
        else:
            self.vectorizer = TfidfVectorizer(max_features=self.n_features,
                                              stop_words='english',
                                              use_idf=self.use_idf)

        data = []
        for record in records:
            data.append(' '.join(record[1:]))

        X = self.vectorizer.fit_transform(data)

        if self.k == -1:
            self.k = int(len(data) * 0.35)

        self.km = KMeans(n_clusters=self.k, init='k-means++', max_iter=100,
                         n_init=1, verbose=self.verbose)
        self.km.fit(X)

        # Build an inverted index and assign records to clusters
        self.k_cluster = {}
        predictions = self.km.predict(X)
        center_distances = self.km.transform(X)
        for index in range(len(predictions)):
            r_id = records[index][0]
            cluster_no = predictions[index]
            center_distance = center_distances[index][cluster_no]
            if cluster_no not in self.k_cluster:
                self.k_cluster[cluster_no] = {}

            self.k_cluster[cluster_no][r_id] = center_distance

        if self.insert_timer:
            self.insert_timer.stop_common()

        for record in records:
            if self.insert_timer:
                with self.insert_timer:
                    self.saai.insert(record)
            else:
                self.saai.insert(record)

    def fit_eval(self, recordset, attributes):
        print("Extracting features from the training dataset using a sparse\
              vectorizer")
        t0 = time()
        if self.use_hashing:
            if self.use_idf:
                # Perform an IDF normalization on the output of
                # HashingVectorizer
                hasher = HashingVectorizer(n_features=self.n_features,
                                           stop_words='english',
                                           non_negative=True,
                                           norm=None, binary=False)
                self.vectorizer = make_pipeline(hasher, TfidfTransformer())
            else:
                self.vectorizer = HashingVectorizer(n_features=self.n_features,
                                                    stop_words='english',
                                                    non_negative=False,
                                                    norm='l2',
                                                    binary=False)
        else:
            self.vectorizer = TfidfVectorizer(max_features=self.n_features,
                                              stop_words='english',
                                              use_idf=self.use_idf)

        records = read_csv(recordset, attributes)
        data = []
        for record in records:
            data.append(' '.join(record[1:]))

        X = self.vectorizer.fit_transform(data)

        print("done in %fs" % (time() - t0))
        print("n_samples: %d, n_features: %d" % X.shape)
        print()

        max = int(len(data) * 0.6)
        min = 2
        step = 1
        fp = open("eval_kmeans-%s.txt" % recordset, mode='w')
        for n_clusters in range(min, max, step):
            t0 = time()
            km = KMeans(n_clusters=n_clusters, init='k-means++', max_iter=100,
                        n_init=1, verbose=self.verbose)
            km.fit(X)
            k_time = time() - t0
            print("Fitting KMeans cluster with k=%d done in %fs" % (n_clusters, k_time))

            # The silhouette_score gives the average value for all the samples.
            # This gives a perspective into the density and separation of the
            # formed clusters
            t0 = time()
            silhouette_sc = silhouette_score(X, km.labels_)
            silhouette_time = time() - t0
            print("Calculating silhouette score done in %fs" % silhouette_time)

            t0 = time()
            # Build an inverted index and assign records to clusters
            self.k_cluster = {}
            predictions = km.predict(X)
            center_distances = km.transform(X)
            for index in range(len(predictions)):
                r_id = records[index][0]
                cluster_no = predictions[index]
                center_distance = center_distances[index][cluster_no]
                if cluster_no not in self.k_cluster:
                    self.k_cluster[cluster_no] = {}

                self.k_cluster[cluster_no][r_id] = center_distance
            print("Build Inverted Index done in %fs" % (time() - t0))

            t0 = time()
            recall_score = self.recall()
            recall_time = time() - t0
            print("Calculate recall as %f done in %fs" % (recall_score, time() - t0))
            print()

            self.km = km
            fp.write("%d, %f, %f, %f, %f, %f\n" % (n_clusters, k_time,
                                                   silhouette_sc, silhouette_time,
                                                   recall_score, recall_time))
            fp.flush()

        print("Choose k as", self.km.n_clusters)
        print()

        t0 = time()
        for record in records:
            self.insert(record)
        print("Pre-calculate similarities done in %fs" % (time() - t0))
        print()

    def insert(self, record):
        r_id = record[0]
        data = [' '.join(record[1:])]
        X = self.vectorizer.transform(data)

        cluster_no = self.km.predict(X)[0]
        center_distance = self.km.transform(X)[0][cluster_no]
        self.k_cluster[cluster_no][r_id] = center_distance

        self.saai.insert(record)

    def query(self, q_record):
        q_id = q_record[0]
        data = [' '.join(q_record[1:])]
        X = self.vectorizer.transform(data)

        cluster_no = self.km.predict(X)[0]
        cluster = self.k_cluster[cluster_no]
        cluster_record_ids = list(filter(lambda r_id: r_id != q_id, cluster.keys()))

        return self.saai.query(q_record, cluster_record_ids)

    def query_from_csv(self, filename, attributes=[]):
        records = read_csv(filename, attributes)
        accumulator = {}
        y_true1 = []
        y_true2 = []
        y_pred = []
        y_scores = []
        for record in records:
            r_id = record[0]
            if self.query_timer:
                with self.query_timer:
                    accumulator[r_id] = self.query(record)
            else:
                accumulator[r_id] = self.query(record)

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
            for cluster in self.k_cluster.values():
                # 2. Check every cluster to find out, if both records have at
                # at least one cluster in common
                if id1 in cluster.keys() and id2 in cluster.keys():
                    # 3. If both records are in same block abort search
                    hit = True

                if hit:
                    # 4. If duplicate pair has been found in at least on block
                    # count it as true positive
                    true_p += 1
                    break

        return true_p / total_p

    def frequency_distribution(self):
        """
        Returns the frequency distribution for the k-Means cluster as dict.
        Where key is size of a cluster and value is the number of clusters with
        this size.
        """
        cluster_sizes = []
        for cluster in self.k_cluster.values():
            cluster_sizes.append(len(cluster.keys()))

        return {'Records': Counter(cluster_sizes)}


class DyLSH(object):

    def __init__(self, encode_fn, similarity_fn, threshold=-1.0, top_n=-1.0,
                 gold_standard=None, gold_attributes=None,
                 lsh_threshold=0.3, lsh_num_perm=60, verbose=False,
                 insert_timer=None, query_timer=None):
        self.verbose = verbose
        self.insert_timer = insert_timer
        self.query_timer = query_timer
        self.saai = SimAwareAttributeIndex(encode_fn, similarity_fn, threshold,
                                           top_n, verbose=verbose)
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
        # LSH Attributes
        self.lsh_threshold = lsh_threshold
        self.lsh_num_perm = lsh_num_perm
        self.lsh = MinHashLSH(threshold=self.lsh_threshold,
                              num_perm=self.lsh_num_perm)

    def fit_csv(self, recordset, attributes):
        self.fit(read_csv(recordset, attributes))

    def fit(self, records):
        for record in records:
            if self.insert_timer:
                with self.insert_timer:
                    self.insert(record)
            else:
                self.insert(record)

    def insert(self, record):
        r_id = record[0]
        r_attributes = record[1:]
        min_hash = MinHash(num_perm=self.lsh_num_perm)
        for attribute in r_attributes:
            min_hash.update(attribute.encode('utf-8'))

        self.lsh.insert(r_id, min_hash)
        self.saai.insert(record)

    def query(self, q_record):
        q_attributes = q_record[1:]
        min_hash = MinHash(num_perm=self.lsh_num_perm)
        for attribute in q_attributes:
            min_hash.update(attribute.encode('utf-8'))

        candidate_ids = self.lsh.query(min_hash)
        return self.saai.query(q_record, candidate_ids)

    def query_from_csv(self, filename, attributes=[]):
        records = read_csv(filename, attributes)
        accumulator = {}
        y_true1 = []
        y_true2 = []
        y_pred = []
        y_scores = []
        for record in records:
            r_id = record[0]
            if self.query_timer:
                with self.query_timer:
                    accumulator[r_id] = self.query(record)
            else:
                accumulator[r_id] = self.query(record)

            if self.gold_records:
                calc_micro_scores(r_id, accumulator[r_id],
                                  y_true1, y_scores, self.gold_records)
                calc_micro_metrics(r_id, accumulator[r_id],
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
            for table in self.lsh.hashtables:
                for block in table.values():
                    # 2. Check every block to find out, if both records have at
                    # at least one block in common
                    if id1 in block and id2 in block:
                        # 3. If both records are in same block abort search
                        hit = True
                        break

                if hit:
                    # 4. If duplicate pair has been found in at least on block
                    # count it as true positive and proceed with next
                    true_p += 1
                    break

        return true_p / total_p

    def frequency_distribution(self):
        """
        Returns the frequency distribution of the minhash table as dict. Where
        key is size of a cluster and value is the number of clusters with this
        size.
        """
        self.lsh.hashtables
        pass
        cluster_sizes = []
        for table in self.lsh.hashtables:
            for values in table.values():
                cluster_sizes.append(len(values))

        return {'Records': Counter(cluster_sizes)}


class SimAwareAttributeIndex(object):

    def __init__(self, encode_fn, similarity_fn, threshold=-1.0, top_n=-1.0,
                 gold_standard=None, gold_attributes=None, verbose=False):
        self.verbose = verbose

        self.BI = {}    # Block Index (BI)
        self.SI = {}    # Similarity Index (SI)
        self.dataset = {}

        self.similarity_fn = similarity_fn
        self.encode_fn = encode_fn
        self.threshold = threshold
        self.top_n = top_n

    def insert(self, record):
        r_id = record[0]
        r_attributes = record[1:]

        for index, attribute in enumerate(r_attributes):
            if attribute not in self.SI:
                # Insert value into Block Index
                if isinstance(self.encode_fn, list):
                    encoding = self.encode_fn[index](attribute)
                else:
                    encoding = self.encode_fn(attribute)

                if encoding not in self.BI.keys():
                    self.BI[encoding] = set()

                self.BI[encoding].add(attribute)

                #  Calculate similarities and update SI
                block = list(filter(lambda x: x != attribute, self.BI[encoding]))
                for block_value in block:
                    if isinstance(self.similarity_fn, list):
                        similarity = self.similarity_fn[index](attribute, block_value)
                    else:
                        similarity = self.similarity_fn(attribute, block_value)
                    similarity = round(similarity, 1)
                    #  Append similarity to block_value
                    if block_value not in self.SI.keys():
                        self.SI[block_value] = {}

                    self.SI[block_value][attribute] = similarity
                    #  Append similarity to value
                    if attribute not in self.SI.keys():
                        self.SI[attribute] = {}

                    self.SI[attribute][block_value] = similarity

        self.dataset[r_id] = r_attributes

    def query(self, q_record, candidate_ids):
        """
            Compares a query record against a set of candidate records. Returns
            the accumulated attribute score for each candidate.
        """
        q_id = q_record[0]
        q_attributes = q_record[1:]

        if q_id not in self.dataset:
            print("INSERT")
            self.insert(q_record)

        accumulator = {}
        for c_id in candidate_ids:
            if q_id == c_id:
                continue

            c_attributes = self.dataset[c_id]
            s = 0.
            for index, (q_attribute, c_attribute) in enumerate(zip(q_attributes, c_attributes)):
                if q_attribute == c_attribute:
                    s += 1.
                elif q_attribute in self.SI and c_attribute in self.SI[q_attribute]:
                    s += self.SI[q_attribute][c_attribute]
                else:
                    if isinstance(self.similarity_fn, list):
                        similarity = self.similarity_fn[index](q_attribute, c_attribute)
                    else:
                        similarity = self.similarity_fn(q_attribute, c_attribute)

                    s += round(similarity, 1)

            if s > self.threshold:
                accumulator[c_id] = s

        if self.top_n > 0:
            return dict(Counter(accumulator).most_common(self.top_n))

        return accumulator
