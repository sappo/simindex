# -*- coding: utf-8 -*-
import math
import heapq
import numpy as np
from pprint import pprint
from collections import defaultdict, namedtuple, Counter
from gensim import corpora, models


def wglobal(doc_freq, total_docs):
    return math.log1p(total_docs / doc_freq)

SimTupel = namedtuple('SimTupel', ['t1', 't2', 'sim'])


class WeakLabels(object):

    def __init__(self, attribute_count, stoplist=None,
                 max_positive_pairs=100,
                 max_negative_pairs=200,
                 upper_threshold=0.6,
                 lower_threshold=0.1):
        self.attribute_count = attribute_count
        if stoplist:
            self.stoplist = stoplist
        else:
            self.stoplist = set('for a of the and to in'.split())

        self.P = None
        self.N = None
        self.max_positive_pairs = max_positive_pairs
        self.max_negative_pairs = max_negative_pairs
        self.upper_threshold = upper_threshold
        self.lower_threshold = lower_threshold

    def string_to_bow(self, record):
        # Remove stop words from record
        words = record.split()
        words = [word for word in words if word not in self.stoplist]
        return self.dictionary.doc2bow(words)

    def fit(self, dataset):
        texts = []
        self.dataset = dataset
        for r_attributes in dataset.values():
            # Remove stop words
            for attribute in r_attributes:
                words = attribute.split()
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

        tfidf_1 = dict(self.tfidf_model[bow_t1])
        tfidf_2 = dict(self.tfidf_model[bow_t2])
        for w1_id in tfidf_1.keys():
            if w1_id in tfidf_2:
                similarity += tfidf_1[w1_id] * tfidf_2[w1_id]

        return similarity

    def predict(self):
        blocker = {}
        candidates = set()
        window_size = 5
        max_positive_pairs = self.max_positive_pairs
        max_negative_pairs = self.max_negative_pairs

        P = []
        N = []

        for field in range(0, self.attribute_count):
            blocker[field] = defaultdict(list)
            for r_id, r_attributes in self.dataset.items():
                attribute = r_attributes[field]
                words = attribute.split()
                tokens = [word for word in words if word not in self.stoplist]
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
        filtered_P = []
        for index, pair in enumerate(P):
            for feature in fPDisj:
                if feature.pv[index] == 1:
                    filtered_P.append(pair)
                    break

        filtered_N = []
        for index, pair in enumerate(N):
            for feature in fPDisj:
                if feature.nv[index] == 1:
                    filtered_N.append(pair)
                    break

        return filtered_P, filtered_N


BlockingKey = namedtuple('BlockingKey', ['predicate', 'field', 'encoder'])


class Feature:
    def __init__(self, predicates, fsc, msc):
        self.predicates = predicates
        self.fsc = fsc
        self.msc = msc
        self.pv = None
        self.nv = None

    def union(self, other):
        pred_conjunction = self.predicates + other.predicates
        feature = Feature(pred_conjunction, None, None)
        feature.pv = self.pv & other.pv
        feature.nv = self.nv & other.nv
        return feature

    def signature(self):
        return set((p.predicate, p.field) for p in self.predicates)

    def blocking_key_values(self, r_attributes):
        BKVs = set()
        for blocking_key in self.predicates:
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
        for blocking_key in self.predicates:
            fields.add(blocking_key.field)

        return fields

    def __repr__(self):
        return "Feature(%s, fsc=%s, msc=%s)" % (self.predicates,
                                                self.fsc,
                                                self.msc)

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
        self.P = P
        self.N = N
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

    @staticmethod
    def f1_score(Pf, Nf, i):
        Pfi = Pf[i]
        Nfi = Nf[i]

        TP = np.sum(Pfi)
        FP = len(Pfi) - TP
        FN = np.sum(Nfi)
        TN = len(Nfi) - FN

        recall = TP / (FN + TP)
        precision = TP / (TP + FP)
        harmonic_mean = 2 * ((precision * recall) / (precision + recall))
        return harmonic_mean

    @staticmethod
    def overall_score(key_coverage, bavg, bvar, cavg, cvar):
        return 1 - key_coverage + bavg + bvar + cavg + cvar

    def terms(self, Pf, Nf):
        count = 0
        forbidden = []
        PfT = Pf.copy()
        NfT = Nf.copy()
        original_features = self.features.copy()

        while count < self.k:
            Kf = []
            for index, feature in enumerate(self.features):
                if feature not in forbidden:
                    bmean, bvar, cmean, cvar = self.block_metrics(feature)
                    feature.fsc = self.overall_score(
                            self.f1_score(PfT, NfT, index),
                            bmean, bvar, cmean, cvar)
                    feature.msc = self.my_score(PfT, NfT, index)
                    Kf.append(feature)

            fsc_avg = np.mean([feature.fsc for feature in self.features])
            Kf = sorted(Kf, key=lambda feature: feature.msc, reverse=True)
            for feature in Kf:
                for original_feature in original_features:
                    original_signature = original_feature.signature().pop()
                    if original_signature not in feature.signature():
                        new_feature = feature.union(original_feature)
                        if new_feature not in self.features:
                            # Calculate new scores
                            bmean, bvar, cmean, cvar = self.block_metrics(new_feature)
                            new_feature.fsc = self.overall_score(
                                    self.f1_score([new_feature.pv], [new_feature.nv], 0),
                                    bmean, bvar, cmean, cvar)
                            new_feature.msc = self.my_score([new_feature.pv], [new_feature.nv], 0)
                            # Add new feature if above average score
                            if new_feature.fsc < fsc_avg:
                                self.features.append(new_feature)
                                PfT.append(new_feature.pv)
                                NfT.append(new_feature.nv)

                forbidden.append(feature)

            count += 1

        return PfT, NfT

    def block_metrics(self, feature):
        FBI = defaultdict(dict)
        for r_id, r_attributes in self.dataset.items():
            blocking_key_values = feature.blocking_key_values(r_attributes)
            for encoding in blocking_key_values:
                for blocking_key in feature.predicates:
                    field = blocking_key.field
                    attribute = r_attributes[field]

                    BI = FBI[field]
                    if encoding not in BI.keys():
                        BI[encoding] = defaultdict(set)

                    BI[encoding][attribute].add(r_id)

        block_sizes = []
        candidate_sizes = []
        for BI in FBI.values():
            for block_key in BI.keys():
                block_sizes.append(len(BI[block_key]))
                for attribute in BI[block_key]:
                    candidate_sizes.append(len(BI[block_key][attribute]))

        # Calculate Block metrics
        bmax = np.max(block_sizes)
        bavg = np.mean(block_sizes)
        bvar = np.var(block_sizes)
        # Calculate Candidate metrics
        cmax = np.max(candidate_sizes)
        cavg = np.mean(candidate_sizes)
        cvar = np.var(candidate_sizes)
        return bavg, bvar, cavg, cvar

    def feature_vector(self, feature):
            pv = np.empty(len(self.P), np.bool)
            nv = np.empty(len(self.N), np.bool)
            for index, pair in enumerate(self.P):
                result = True
                for predicate in feature.predicates:
                    field = predicate.field
                    pred = predicate.predicate
                    t1_field = self.dataset[pair.t1][field]
                    t2_field = self.dataset[pair.t2][field]
                    result &= pred(t1_field, t2_field)
                    if not result:
                        break

                pv[index] = result

            for index, pair in enumerate(self.N):
                result = True
                for predicate in feature.predicates:
                    field = predicate.field
                    pred = predicate.predicate
                    t1_field = self.dataset[pair.t1][field]
                    t2_field = self.dataset[pair.t2][field]
                    result &= pred(t1_field, t2_field)
                    if not result:
                        break

                nv[index] = result

            return pv, nv

    def transform(self, dataset):
        Pf = []
        Nf = []
        self.dataset = dataset
        # Create boolean feature vectors
        for index, feature in enumerate(self.features):
            pv, nv = self.feature_vector(feature)
            feature.pv = pv
            feature.nv = nv
            Pf.append(pv)
            Nf.append(nv)

        # Build and score features
        Pf, Nf = self.terms(Pf, Nf)

        # Sort features
        Kf = []
        Km = []
        for feature in self.features:
            Kf.append(feature)
            Km.append(feature)

        Kf = sorted(Kf, key=lambda feature: feature.fsc)
        Km = sorted(Km, key=lambda feature: feature.msc, reverse=True)

        fPDisj = []
        fPDisj_p = None
        # Example coverage variant
        for feature in Kf:
            if fPDisj_p is None:
                # Init NULL-vector
                fPDisj_p = np.array([0 for index in range(0, len(feature.pv))])

            new_sig = fPDisj_p | np.array(feature.pv)
            if not np.array_equal(new_sig, fPDisj_p):
                fPDisj_p = new_sig
                fPDisj.append(feature)

        # Post filter bad scores: Those bad scores might be considerable smaller
        # than those of other but may still be considerable larger then the top
        # ranked.
        score_mean = np.mean([feature.fsc for feature in fPDisj])
        fPDisj[:] = filter(lambda f: f.fsc < 100, fPDisj)

        return fPDisj


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
