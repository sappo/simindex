# -*- coding: utf-8 -*-
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import Normalizer
from sklearn import metrics
from sklearn.metrics import silhouette_samples, silhouette_score
from sklearn.cluster import KMeans, MiniBatchKMeans
from time import time
import numpy as np
from collections import Counter
from .helper import read_csv
from .plot import *

class DyKMeans(object):

    def __init__(self, encode_fn, simmetric_fn, threshold=-1.0, top_n=-1.0,
                 gold_standard=None, gold_attributes=None,
                 n_features=10000, use_hashing=False, use_idf=True, verbose=False):
        self.verbose = verbose
        # ER Index attributes
        self.BI = {}    # Block Index (BI)
        self.SI = {}    # Similarity Index (SI)
        self.simmetric = simmetric_fn
        self.encode = encode_fn
        self.dataset = {}
        self.threshold = threshold
        self.top_n = top_n
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

    def fit(self, recordset, attributes):
        print("Extracting features from the training dataset using a sparse\
              vectorizer")
        t0 = time()
        if self.use_hashing:
            if use_idf:
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

        max = int(len(data) * 0.5)
        min = int(len(data) * 0.3)
        step = int((max - min) / 5)
        if step == 0: step = 1
        fp = open("plot-len%d-min%d-max%d-step%d.txt" % (len(data), min, max, step), mode='w')
        cluster_range = []
        cluster_silhouette_score = []
        cluster_silhouette_time = []
        cluster_recall_score = []
        cluster_recall_time = []
        for n_clusters in range(min, max, step):
            cluster_range.append(n_clusters)
            t0 = time()
            km = KMeans(n_clusters=n_clusters, init='k-means++', max_iter=100,
                        n_init=1, verbose=self.verbose)
            km.fit(X)
            print("Fitting KMeans cluster with k=%d done in %fs" % (n_clusters, time() - t0))

            # The silhouette_score gives the average value for all the samples.
            # This gives a perspective into the density and separation of the
            # formed clusters
            t0 = time()
            silhouette_sc = silhouette_score(X, km.labels_)
            silhouette_time = time() - t0
            cluster_silhouette_score.append(silhouette_sc)
            cluster_silhouette_time.append(silhouette_time)
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
            cluster_recall_score.append(recall_score)
            cluster_recall_time.append(recall_time)
            print("Calculate recall as %f done in %fs" % (recall_score, time() - t0))
            print()

            self.km = km
            fp.write("%d, %f, %f, %f, %f\n" % (n_clusters, silhouette_sc, silhouette_time, recall_score, recall_time))

        print("Choose k as", self.km.n_clusters)
        print()

        t0 = time()
        for record in records:
            self.insert(record)
        print("Pre-calculate similarities done in %fs" % (time() - t0))
        print()

        # draw_plots(cluster_range, [cluster_avg_sils, cluster_recalls])

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

    def insert(self, record):
        r_id = record[0]
        data = [' '.join(record[1:])]
        X = self.vectorizer.transform(data)

        cluster_no = self.km.predict(X)[0]
        center_distance = self.km.transform(X)[0][cluster_no]
        self.k_cluster[cluster_no][r_id] = center_distance

        for attribute in record[1:]:
            if attribute not in self.SI:
                # Insert value into Block Index
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

        self.dataset[r_id] = record[1:]

    def query(self, q_record):
        q_id = q_record[0]
        data = [' '.join(q_record[1:])]
        X = self.vectorizer.transform(data)

        if q_id not in self.dataset:
            self.insert(q_record)

        cluster_no = self.km.predict(X)[0]
        cluster = self.k_cluster[cluster_no]
        cluster_records = list(filter(lambda r_id: r_id != q_id, cluster.keys()))
        accumulator = {}
        for r_id in cluster_records:
            if r_id not in accumulator.keys():
                record = self.dataset[r_id]
                s = 0.
                for q_attribute, attribute in zip(q_record[1:], record):
                    if q_attribute == attribute:
                        s += 1.
                    elif attribute in self.SI[q_attribute]:
                        s += self.SI[q_attribute][attribute]
                    else:
                        similarity = self.simmetric(q_attribute, attribute)
                        s += round(similarity, 1)

                if s > self.threshold:
                    accumulator[r_id] = s

        if self.top_n > 0:
            return dict(Counter(accumulator).most_common(self.top_n))

        return accumulator

    def frequency_distribution(self):
        """
        Returns the frequency distribution for the k-Means cluster as dict.
        Where key is size of a cluster and value is the number of clusters with
        this size.
        """
        cluster_sizes = []
        for cluster in self.k_cluster.values():
            cluster_sizes.append(len(cluster.keys()))

        return Counter(cluster_sizes)
