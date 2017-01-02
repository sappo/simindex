# -*- coding: utf-8 -*-
import math
import heapq
import numpy as np
from collections import defaultdict
from collections import namedtuple
from simindex.helper import read_csv
from gensim import corpora, models

# documents_ii = {
    # "1":["arnie morton's of chicago","435 s. la cienega blv.","los angeles","310/246-1501","american", '0'],
    # "2":["arnie morton's of chicago","435 s. la cienega blvd.","los angeles","310-246-1501","steakhouses", '0'],
    # "3":["art's delicatessen","12224 ventura blvd.","studio city","818/762-1221","american", '1'],
    # "4":["art's deli","12224 ventura blvd.","studio city","818-762-1221","delis", '1'],
    # "5":["hotel bel-air","701 stone canyon rd.","bel air","310/472-1211","american", '2'],
    # "6":["motel interstate","Route 66","Nowhere","555-666-777","american", '5'],
    # "7":["motel interstate","Route 66","Nowhere","555-666-777","american", '5'],
# }
# documents = [
    # ["1","arnie morton's of chicago","435 s. la cienega blv.","los angeles","310/246-1501","american", '0'],
    # ["2","arnie morton's of chicago","435 s. la cienega blvd.","los angeles","310-246-1501","steakhouses", '0'],
    # ["3","art's delicatessen","12224 ventura blvd.","studio city","818/762-1221","american", '1'],
    # ["4","art's deli","12224 ventura blvd.","studio city","818-762-1221","delis", '1'],
    # ["5","hotel bel-air","701 stone canyon rd.","bel air","310/472-1211","american", '2'],
    # ["6","motel interstate","Route 66","Nowhere","555-666-777","american", '5'],
    # ["7","motel interstate","Route 66","Nowhere","555-666-777","american", '5'],
# ]


def wglobal(doc_freq, total_docs):
    return math.log1p(total_docs / doc_freq)



class WeakLabels(object):

    def __init__(self, stoplist=None):
        self.dataset = {}
        self.attribute_count = 999
        if stoplist:
            self.stoplist = stoplist
        else:
            self.stoplist = set('for a of the and to in'.split())

    def string_to_bow(self, record):
        # Remove stop words from record
        words = record.lower().split()
        words = [word for word in words if word not in self.stoplist]
        return self.dictionary.doc2bow(words)

    def fit_csv(self, filename, attributes=[]):
        records = read_csv(filename, attributes)
        self.fit(records)

    def fit(self, records):
        texts = []
        self.attribute_count = min([self.attribute_count, len(records[0]) - 1])
        for record in records:
            r_id = record[0]
            r_attributes = record[1:]
            # Store dataset
            self.dataset[r_id] = r_attributes

            # Remove stop words
            for attribute in r_attributes:
                words = attribute.lower().split()
                texts.append([word for word in words if word not in self.stoplist])

        self.dictionary = corpora.Dictionary(texts)
        corpus = [self.dictionary.doc2bow(text) for text in texts]
        self.tfidf_model = models.TfidfModel(corpus, normalize=True,
                                             wlocal=math.log1p,
                                             wglobal=wglobal)

    def tfidf_similarity(self, t1, t2):
        """
        Calculates the TF/IDF similarity between two terms `t1` and `t2` taking
        a dictionary and TfidfModel into account.

        Returns similarity value between 0 and 1.
        """
        similarity = 0.
        t1_attribute_str = ' '.join(self.dataset[t1])
        t2_attribute_str = ' '.join(self.dataset[t2])
        bow_t1 = self.string_to_bow(t1_attribute_str)
        bow_t2 = self.string_to_bow(t2_attribute_str)

        tfidf_1 = self.tfidf_model[bow_t1]
        tfidf_2 = self.tfidf_model[bow_t2]
        for weight1 in tfidf_1:
            for weight2 in tfidf_2:
                if (weight1[0] == weight2[0]):
                    similarity += weight1[1] * weight1[1]

        return similarity

    def predict(self):
        blocker = {}
        candidates = set()
        window_size = 5
        upper_threshold = 0.7
        lower_threshold = 0.3
        max_positive_pairs = 5
        max_negative_pairs = 10

        P = set()
        N = set()

        for field in range(0, self.attribute_count):
            blocker[field] = defaultdict(list)
            for r_id, r_attributes in self.dataset.items():
                attribute = r_attributes[field]
                words = attribute.lower().split()
                tokens = [word for word in words if word not in self.stoplist]
                for token in tokens:
                    if r_id not in blocker[field][token]:
                        blocker[field][token].append(r_id)

        for field in range(0, self.attribute_count):
            field_blocks = blocker[field]
            for token in field_blocks.keys():
                token_block = sorted(field_blocks[token])
                sorted
                index = 0
                # Move window over token block for field
                while len(token_block[index + 1:index + window_size]) > 0:
                    for candidate in token_block[index + 1:index + window_size]:
                        candidates.add((token_block[index], candidate))
                    index += 1

        SimTupel = namedtuple('SimTupel', ['t1', 't2', 'sim'])
        for t1, t2 in candidates:
            sim = self.tfidf_similarity(t1, t2)
            if sim >= upper_threshold:
                P.add(SimTupel(t1, t2, sim))
            if sim < lower_threshold:
                N.add(SimTupel(t1, t2, sim))

        P = heapq.nlargest(max_positive_pairs, P, key=lambda pair: pair.sim)
        N = heapq.nsmallest(max_negative_pairs, N, key=lambda pair: pair.sim)

        return P, N


# documents_ii = {}
# documents = read_csv("restaurant.csv")
# for record in documents:
    # documents_ii[record[0]] = record[1:]

# documents = ["Human machine", "interface for", "lab abc computer", "applications"]
             # "A survey of user opinion of computer system response time",
             # "The EPS user interface management system",
             # "System and human system engineering testing of EPS",
             # "Relation of user perceived response time to error measurement",
             # "The generation of random binary unordered trees",
             # "Thee intersection graph of paths in trees",
             # "Graph minors IV Widths of trees and well quasi ordering",
             # "Graph minors A survey"]

stoplist = set('for a of the and to in'.split())


# def build_corpus(records):
    # pass


# def string_to_bow(slf, record):
    # pass

# def simtfidf(t1, t2, dictionary, tfidf):
    # pass

# texts, dictionary, corpus = build_corpus(documents)
# tfidf = models.TfidfModel(corpus, normalize=True, wlocal=math.log1p,
                          # wglobal=wglobal)

# print(simtfidf("Thee intersection graph", "Thee intersection graph well",
      # dictionary, tfidf))

class DisjunctiveBlockingScheme(object):

    def __init__(self, features, labels):
        self.features = features
        self.labels = labels

    @staticmethod
    def fisher_score(Pf, Nf, i):
        Pfi = Pf[i]
        Nfi = Nf[i]

        mpi = np.mean(Pfi)
        vpi = np.var(Pfi)
        mni = np.mean(Nfi)
        vni = np.var(Nfi)
        m = np.mean(Pfi + Nfi)
        d = len(Pfi)
        nd = len(Nfi)

        if mpi == 0:
            return 0.
        else:
            if vpi == 0 and vni == 0:
                vpi = 0.01
                vni = 0.01

            pi = ((d * math.pow(mpi - m, 2)) + (nd * math.pow(mni - m, 2))) / \
                 ((d * math.pow(vpi, 2)) + (nd * math.pow(vni, 2)))
            return pi


    @staticmethod
    def my_score(Pf, Nf, i):
        Pfi = Pf[i]
        Nfi = Nf[i]

        Pp = Pfi.count(1) / len(Pfi)
        Nn = Nfi.count(1) / len(Nfi)
        return Pp - Nn

    def transform(self):
        Pf = []
        Nf = []
        # Create boolean feature vectors
        for index, feature in enumerate(self.features):
            for field in range(0, self.labels.attribute_count):
                Pf.append([])
                Nf.append([])
                for pair in P:
                    t1_field = self.labels.dataset[pair.t1][field]
                    t2_field = self.labels.dataset[pair.t2][field]
                    t1_token = t1_field.lower().split()
                    t2_token = t2_field.lower().split()
                    Pf[index * self.labels.attribute_count + field].append(feature(t1_token, t2_token))
                for pair in N:
                    t1_field = self.labels.dataset[pair.t1][field]
                    t2_field = self.labels.dataset[pair.t2][field]
                    t1_token = t1_field.lower().split()
                    t2_token = t2_field.lower().split()
                    Nf[index * self.labels.attribute_count + field].append(feature(t1_token, t2_token))

        Kf = []
        Km = []
        # Score feature DNFs
        for index, feature in enumerate(self.features):
            for field in range(0, self.labels.attribute_count):
                Kf.append((feature, field, self.fisher_score(Pf, Nf, index * self.labels.attribute_count + field)))
                Km.append((feature, field, self.my_score(Pf, Nf, index * self.labels.attribute_count + field)))

        Kf = sorted(Kf, key=lambda score: score[2], reverse=True)
        Km = sorted(Km, key=lambda score: score[2], reverse=True)
        print(Kf)
        # print(Km)

        fPDisj = []
        for feature, field, score in Kf:
            if field not in [field for feature, field in fPDisj]:
                fPDisj.append((feature, field))

        return fPDisj



def has_common_token(t1, t2):
    t1_tokens = set(word for word in t1 if word not in stoplist)
    t2_tokens = set(word for word in t2 if word not in stoplist)

    if len(t1_tokens.intersection(t2_tokens)) > 0:
        return 1
    else:
        return 0


def is_exact_match(t1, t2):
    t1_tokens = set(word for word in t1 if word not in stoplist)
    t2_tokens = set(word for word in t2 if word not in stoplist)

    if len(t1_tokens.symmetric_difference(t2_tokens)) == 0:
        return 1
    else:
        return 0


labels = WeakLabels()
labels.fit_csv("restaurant.csv")
P, N = labels.predict()

print(P)
print(N)

dnfblock = DisjunctiveBlockingScheme([has_common_token, is_exact_match], labels)
fPDisj = dnfblock.transform()
print(fPDisj)
