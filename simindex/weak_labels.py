# -*- coding: utf-8 -*-
import math
import heapq
import numpy as np
import itertools as it
import sklearn
from collections import defaultdict, namedtuple
from gensim import corpora, models
from pprint import pprint
import simindex.helper as hp

try:
    profile
except NameError as e:
    def profile(func):
        def inner(*args, **kwargs):
            return func(*args, **kwargs)
        return inner


def wglobal(doc_freq, total_docs):
    return math.log1p(total_docs / doc_freq)

SimTupel = namedtuple('SimTupel', ['t1', 't2', 'sim'])


class WeakLabels(object):

    def __init__(self, attribute_count,
                 gold_pairs=None,
                 max_positive_pairs=100,
                 max_negative_pairs=200,
                 upper_threshold=0.6,
                 lower_threshold=0.1):
        self.attribute_count = attribute_count

        self.P = None
        self.N = None
        self.gold_pairs = gold_pairs
        self.max_positive_pairs = max_positive_pairs
        self.max_negative_pairs = max_negative_pairs
        self.upper_threshold = upper_threshold
        self.lower_threshold = lower_threshold

    def string_to_bow(self, record):
        # Remove stop words from record
        words = record.split()
        return self.dictionary.doc2bow(words)

    def fit(self, dataset):
        texts = []
        self.dataset = dataset
        for r_attributes in dataset.values():
            # Remove stop words
            for attribute in r_attributes:
                texts.append(attribute.split())

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

        tfidf_1 = dict(self.tfidf_model[bow_t1])
        tfidf_2 = dict(self.tfidf_model[bow_t2])
        for w1_id in tfidf_1.keys():
            if w1_id in tfidf_2:
                similarity += tfidf_1[w1_id] * tfidf_2[w1_id]

        return similarity

    @profile
    def predict(self):
        blocker = {}
        candidates = set()
        window_size = 5
        bins = 20
        # Set max_candidates depending on available gold pairs and user definded
        # values
        if self.gold_pairs:
            max_positive_pairs = len(self.gold_pairs)
        else:
            if self.max_positive_pairs:
                max_positive_pairs = self.max_positive_pairs
            else:
                max_positive_pairs = len(self.dataset) * 0.05

        if self.max_negative_pairs:
            max_negative_pairs = self.max_negative_pairs
        else:
            max_negative_pairs = max_positive_pairs * 3

        P = []
        N = []
        for field in range(0, self.attribute_count):
            blocker[field] = defaultdict(list)
            for r_id, r_attributes in self.dataset.items():
                attribute = r_attributes[field]
                tokens = attribute.split()
                for tokens in tokens:
                    if r_id not in blocker[field][tokens]:
                        blocker[field][tokens].append(r_id)

        for field in range(0, self.attribute_count):
            field_blocks = blocker[field]
            for tokens in field_blocks.keys():
                token_block = sorted(field_blocks[tokens])
                sorted
                index = 0
                # Move window over token block for field
                while len(token_block[index + 1:index + window_size]) > 0:
                    for candidate in token_block[index + 1:index + window_size]:
                        candidates.add((token_block[index], candidate))
                    index += 1

        if self.gold_pairs:
            for t1, t2 in self.gold_pairs:
                P.append(SimTupel(t1, t2, 1))

            trim_threshold = max_negative_pairs * 3
            candidate = candidates.difference(self.gold_pairs)
            N_bins = [[] for x in range(bins)]
            for t1, t2 in candidates:
                sim = self.tfidf_similarity(t1, t2)
                bin = int(sim * bins)
                if bin >= bins:
                    bin = bins - 1

                N_bins[bin].append(SimTupel(t1, t2, sim))

            # Calculate probability distribution
            weights = [len(bin) for bin in N_bins]
            wsum = sum(weights)
            weights[:] = [float(weight)/wsum for weight in weights]

            # Select pairs by probability distribution
            for dummy in range(max_negative_pairs):
                bin = np.random.choice(range(bins), p=weights)
                N_choice = N_bins[bin]
                if len(N_choice) > 0:
                    r_choice = np.random.choice(range(len(N_choice)))
                    N.append(N_choice[r_choice])
                    del N_choice[r_choice]

        else:
            trim_threshold = max_negative_pairs * 3
            for index, (t1, t2) in enumerate(candidates):
                sim = self.tfidf_similarity(t1, t2)
                if sim >= self.upper_threshold:
                    P.append(SimTupel(t1, t2, sim))
                if sim < self.lower_threshold:
                    N.append(SimTupel(t1, t2, sim))

                if index % trim_threshold == 0:
                    P = heapq.nlargest(max_positive_pairs, P, key=lambda pair: pair.sim)
                    N = heapq.nlargest(max_negative_pairs, N, key=lambda pair: pair.sim)

            P = heapq.nlargest(max_positive_pairs, P, key=lambda pair: pair.sim)
            N = heapq.nlargest(max_negative_pairs, N, key=lambda pair: pair.sim)

        return P, N

    @staticmethod
    def filter(fPDisj, P, N):
        # filtered_P = []
        # for index, pair in enumerate(P):
            # for feature in fPDisj:
                # if feature.pv[index] == 1:
                    # filtered_P.append(pair)
                    # break

        # filtered_N = []
        # for index, pair in enumerate(N):
            # for feature in fPDisj:
                # if feature.nv[index] == 1:
                    # filtered_N.append(pair)
                    # break

        return P, N


BlockingKey = namedtuple('BlockingKey', ['field', 'encoder'])


class Feature:
    def __init__(self, blocking_keys, fsc, msc):
        self.blocking_keys = blocking_keys
        self.fsc = fsc
        self.y_true = None
        self.y_pred = None

    def union(self, other):
        bk_conjunction = self.blocking_keys + other.blocking_keys
        feature = Feature(bk_conjunction, None, None)
        return feature

    def signature(self):
        return set((bk.encoder, bk.field) for bk in self.blocking_keys)

    def blocking_key_values(self, r_attributes):
        BKVs = set()
        for blocking_key in self.blocking_keys:
            attribute = r_attributes[blocking_key.field]
            if not attribute:
                # No need to encode empty attributes
                continue

            if len(BKVs) == 0:
                BKVs = blocking_key.encoder(attribute)
            else:
                concat_BKVs = set()
                for dummy in range(0, len(BKVs)):
                    f_encoding = BKVs.pop()
                    for bk_encoding in blocking_key.encoder(attribute):
                        concat_BKVs.add(f_encoding + bk_encoding)

                BKVs = concat_BKVs

        return BKVs

    def covered_fields(self):
        fields = set()
        for blocking_key in self.blocking_keys:
            fields.add(blocking_key.field)

        return fields

    def __repr__(self):
        return "Feature(%s, fsc=%s)" % (self.blocking_keys, self.fsc)

    def __eq__(self, other):
        if isinstance(other, self.__class__):
            return self.signature() == other.signature()
        return False

    def __ne__(self, other):
        return not self.__eq__(other)

    def __hash__(self):
        return hash(str(self.signature()))


class DisjunctiveBlockingScheme(object):

    def __init__(self, blocking_keys, P, N, k=1):
        self.P = [frozenset((p.t1, p.t2)) for p in P]
        self.N = [frozenset((p.t1, p.t2)) for p in N]
        self.frozen_P = set(self.P)
        self.flat_P = set(hp.flatten(self.frozen_P))
        self.features = []
        self.k = k

        for blocking_key in blocking_keys:
            feature = Feature([blocking_key], None, None)
            self.features.append(feature)

    @staticmethod
    def fisher_score(Pf, Nf, i):
        Pfi = Pf[i]
        Nfi = Nf[i]

        mpi = np.mean(Pfi)
        vpi = np.var(Pfi)
        mni = np.mean(Nfi)
        vni = np.var(Nfi)
        m = np.mean(np.concatenate((Pfi, Nfi)))
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

        Pp = np.mean(Pfi)
        Nn = np.mean(Nfi)
        return Pp - Nn

    def terms(self):
        count = 0
        forbidden = []
        original_features = self.features.copy()

        while count < self.k:
            Kf = []
            for index, feature in enumerate(self.features):
                if feature not in forbidden:
                    feature.fsc = self.block_metrics(feature)
                    Kf.append(feature)

            # fsc_avg = np.mean([feature.fsc for feature in self.features])
            Kf = sorted(Kf, key=lambda feature: feature.fsc, reverse=True)
            for feature in Kf:
                for original_feature in original_features:
                    original_signature = original_feature.signature().pop()
                    if original_signature not in feature.signature():
                        new_feature = feature.union(original_feature)
                        if new_feature not in self.features:
                            # Calculate new scores
                            new_feature.fsc = self.block_metrics(new_feature)
                            self.features.append(new_feature)

                forbidden.append(feature)

            count += 1

    def block_metrics(self, feature):
        FBI = defaultdict(dict)
        for r_id, r_attributes in self.dataset.items():
            blocking_key_values = feature.blocking_key_values(r_attributes)
            for encoding in blocking_key_values:
                for blocking_key in feature.blocking_keys:
                    field = blocking_key.field

                    BI = FBI[field]
                    if encoding not in BI.keys():
                        BI[encoding] = set()

                    BI[encoding].add(r_id)

        FBI_candidate_pairs = set()
        TP, FP, FN = 0, 0, 0
        for BI in FBI.values():
            for block_key in BI.keys():
                candidates = BI[block_key]
                # Generate candidate pairs
                candidate_pairs = it.combinations(candidates, 2)
                candidate_pairs = set([frozenset(p) for p in candidate_pairs])

                # Calculate TP, FP, FN
                npairs = len(candidate_pairs)
                block_TP_pairs = candidate_pairs.intersection(self.P)
                block_FP_pairs = candidate_pairs.difference(block_TP_pairs)
                FP_canidates = set(hp.flatten(block_FP_pairs))
                assert npairs == len(block_TP_pairs) + len(block_FP_pairs)
                TP += len(block_TP_pairs)
                FP += len(block_FP_pairs)
                FN += len(FP_canidates.intersection(self.flat_P))

                FBI_candidate_pairs.update(candidate_pairs)

        # Create feature vectors based on positive and negative labels
        y_true = np.zeros(len(self.P) + len(self.N), np.bool)
        y_pred = np.zeros(len(self.P) + len(self.N), np.bool)
        for index, pair in enumerate(self.P):
            if pair in FBI_candidate_pairs:
                y_true[index] = True
                y_pred[index] = True
            else:
                y_true[index] = True
                y_pred[index] = False

        for index, pair in enumerate(self.N):
            if pair in FBI_candidate_pairs:
                y_true[index + len(self.P)] = False
                y_pred[index + len(self.P)] = True
            else:
                y_true[index + len(self.P)] = False
                y_pred[index + len(self.P)] = False

        feature.y_true = y_true
        feature.y_pred = y_pred

        # No need to calculate score if there are no true positives
        if TP == 0:
            return 0

        recall = TP / (FN + TP)
        precision = TP / (TP + FP)
        f1_score = 2 * ((precision * recall) / (precision + recall))
        return f1_score

    @profile
    def transform(self, dataset):
        self.dataset = dataset
        # Build and score features
        self.terms()

        # Sort features
        Kf = []
        for feature in self.features:
            Kf.append(feature)

        Kf = sorted(Kf, key=lambda feature: feature.fsc, reverse=True)
        top_10 = heapq.nlargest(50, Kf, key=lambda feature: feature.fsc)

        blocking_scheme_candidates = []
        for index in range(4):
            blocking_scheme_candidates.extend(it.combinations(top_10, index))

        best_blocking_scheme = None
        best_blocking_score = 0
        for blocking_scheme in blocking_scheme_candidates:
            y_true = np.zeros(len(self.P) + len(self.N), np.bool)
            y_pred = np.zeros(len(self.P) + len(self.N), np.bool)
            for blocking_key in blocking_scheme:
                y_true |= blocking_key.y_true
                y_pred |= blocking_key.y_pred

            score = sklearn.metrics.f1_score(y_true, y_pred)
            if score > best_blocking_score:
                best_blocking_scheme = blocking_scheme
                best_blocking_score = score

        return best_blocking_scheme


def tokens(term):
    return set(term.split())


def has_common_token(t1, t2):
    if len(tokens(t1).intersection(tokens(t2))) > 0:
        return 1
    else:
        return 0


def term_id(term):
    r = set()
    r.add(term)
    return r


def is_exact_match(t1, t2):
    if len(term_id(t1).intersection(term_id(t2))) > 0:
        return 1
    else:
        return 0
