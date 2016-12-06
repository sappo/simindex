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

    def __init__(self, recordset, attributes = [],
                 gold_standard=None, gold_attributes=None,
                 n_features=10000, use_hashing=False, use_idf=True,
                 verbose=False):
        self.n_features = n_features
        self.use_idf = use_idf
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

        print("Extracting features from the training dataset using a sparse\
              vectorizer")
        t0 = time()
        if use_hashing:
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
                                              use_idf=use_idf)

        records = read_csv(recordset, attributes)
        data = []
        for record in records:
            data.append(' '.join(record[1:]))

        X = self.vectorizer.fit_transform(data)

        print("done in %fs" % (time() - t0))
        print("n_samples: %d, n_features: %d" % X.shape)
        print()

        # self.km = KMeans(n_clusters=6, init='k-means++', max_iter=100,
                         # n_init=1, verbose=verbose)

        # print("Clustering sparse data with %s" % self.km)
        # t0 = time()
        # self.km.fit(X)
        # print("done in %0.3fs" % (time() - t0))
        # print()

        max = int(len(data) * 0.6)
        min = int(len(data) * 0.005)
        step = int((max - min) / 10)
        if step == 0: step = 1
        max_silhouette = -1.0
        cluster_range = []
        cluster_avg_sils = []
        cluster_recalls = []
        for n_clusters in range(min, max, step):
            km = KMeans(n_clusters=n_clusters, init='k-means++', max_iter=100,
                        n_init=1, verbose=verbose)
            km.fit(X)

            # The silhouette_score gives the average value for all the samples.
            # This gives a perspective into the density and separation of the
            # formed clusters
            silhouette_avg = silhouette_score(X, km.labels_)
            print(n_clusters, "Avg", silhouette_avg)
            self.km = km
            cluster_range.append(n_clusters)
            cluster_avg_sils.append(silhouette_avg)
            # Compute the silhouette scores for each sample
            # sample_silhouette_values = silhouette_samples(X, km.labels_)
            # avg_distance = 0
            # for i in np.arange(n_clusters):
                # print(sample_silhouette_values[i == km.labels_])
                # for sample in sample_silhouette_values[i == km.labels_]:
                    # avg_distance += sample - silhouette_avg

            # if (max_silhouette < silhouette_avg):
                # max_silhouette = silhouette_avg
                # self.km = km

            # print("Choose k as", self.km.n_clusters)
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

            cluster_recalls.append(self.recall())

        draw_plots(cluster_range, [cluster_avg_sils, cluster_recalls])

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
        q_id = record[0]
        data = [' '.join(record[1:])]
        X = self.vectorizer.transform(data)

        cluster_no = self.km.predict(X)[0]
        center_distance = self.km.transform(X)[0][cluster_no]
        self.k_cluster[cluster_no][q_id] = center_distance

    def query(self, record):
        pass

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
