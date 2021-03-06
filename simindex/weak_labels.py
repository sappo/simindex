# -*- coding: utf-8 -*-
import math
import heapq
import numpy as np
import itertools as it
import sklearn
import json
from nltk import ngrams
from collections import defaultdict, namedtuple, Counter
from gensim import corpora, models
from pprint import pprint
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


def wglobal(doc_freq, total_docs):
    return math.log1p(total_docs / doc_freq)

SimTupel = namedtuple('SimTupel', ['t1', 't2', 'sim'])

class WeakLabels(object):
    def __init__(self, attribute_count, dataset,
                 gold_pairs=None,
                 max_positive_pairs=None,
                 max_negative_pairs=None,
                 upper_threshold=0.6,
                 lower_threshold=0.1,
                 window_size=2,
                 verbose=False):
        self.attribute_count = attribute_count

        self.P = None
        self.N = None
        self.gold_pairs = gold_pairs
        self.dataset = dataset
        self.upper_threshold = upper_threshold
        self.lower_threshold = lower_threshold
        self.window_size = window_size
        self.verbose = verbose

        # Set max_candidates depending on available gold pairs and user definded
        # values
        if self.gold_pairs:
            self.max_positive_pairs = len(self.gold_pairs)
        else:
            if max_positive_pairs:
                self.max_positive_pairs = max_positive_pairs
            else:
                self.max_positive_pairs = len(self.dataset) * 0.05

        if max_negative_pairs:
            self.max_negative_pairs = max_negative_pairs
        else:
            self.max_negative_pairs = self.max_positive_pairs * 3

        # Selecting negative pairs
        self.npairs = 0
        self.bins = 20
        self.choices = np.arange(self.bins)
        self.N_complete = None
        self.weights = None

    def string_to_bow(self, record):
        # Remove stop words from record
        words = record.split()
        return self.dictionary.doc2bow(words)

    def fit(self):
        texts = []
        for r_attributes in self.dataset.values():
            # Remove stop words
            for attribute in r_attributes:
                texts.append(attribute.split())

        logger.info("Create dictionary")
        self.dictionary = corpora.Dictionary(texts)
        logger.info("Create corpus")
        corpus = [self.dictionary.doc2bow(text) for text in texts]
        logger.info("Create tfidf_model")
        self.tfidf_model = models.TfidfModel(corpus, normalize=True,
                                             wlocal=math.log1p,
                                             wglobal=wglobal)

    def tfidf_similarity(self, t1, t2):
        """
        Calculates the TF/IDF similarity between two terms `t1` and `t2` taking
        a dictionary and TfidfModel into account. (WHIRL)

        Returns similarity value between 0 and 1.
        """
        similarity = 0.
        t1_attribute_str = ' '.join(self.dataset[t1])
        t2_attribute_str = ' '.join(self.dataset[t2])
        bow_t1 = self.string_to_bow(t1_attribute_str)
        bow_t2 = self.string_to_bow(t2_attribute_str)

        tfidf_1 = dict(self.tfidf_model[bow_t1])
        tfidf_2 = dict(self.tfidf_model[bow_t2])

        # Normalized vector of tfidf weights
        tfidf_1_norm = math.sqrt(sum([math.pow(w, 2) for w in tfidf_1.values()]))
        tfidf_2_norm = math.sqrt(sum([math.pow(w, 2) for w in tfidf_2.values()]))

        # Calculate similarity
        for w1_id in tfidf_1.keys():
            if w1_id in tfidf_2:
                similarity += tfidf_1[w1_id] * tfidf_2[w1_id]

        return similarity / (tfidf_1_norm * tfidf_2_norm)


    def draw_pair(self):
        if self.N_complete and type(self.N_complete[0]) == set:
            pair = None
            N_bins = self.N_complete
            while not pair and self.npairs > 0:
                bin = np.random.choice(self.choices, p=self.weights)
                N_choice = N_bins[bin]
                try:
                    pair = N_choice.pop()
                except KeyError:
                    pair = None

            self.npairs -= 1
            return pair
        else:
            if self.N_complete:
                return self.N_complete.pop()
            else:
                return None

    def predict(self):
        blocker = {}
        candidates = set()
        window_size = self.window_size

        P = []
        N = []
        if self.verbose:
            logger.info("Blocking by tokens and fields")

        for field in range(0, self.attribute_count):
            blocker[field] = defaultdict(set)
            for r_id, r_attributes in self.dataset.items():
                attribute = r_attributes[field]
                tokens = attribute.split()
                for token in tokens:
                    blocker[field][token].add(r_id)

            if self.verbose:
                logger.info("Blocked field %d" % field)

        if self.verbose:
            logger.info("Moving window to generate candidates")

        for field in range(0, self.attribute_count):
            field_blocks = blocker[field]
            block_filter = filter(lambda t: len(field_blocks[t]) < 100, field_blocks.keys())
            for tokens in block_filter:
                token_block = sorted(field_blocks[tokens])
                index = 0
                # Move window over token block for field
                while len(token_block[index + 1:index + window_size]) > 0:
                    for candidate in token_block[index + 1:index + window_size]:
                        candidates.add((token_block[index], candidate))
                    index += 1

        if self.verbose:
            logger.info("Generated %d candidates" % len(candidates))

        simcount = 0
        if self.gold_pairs:
            for t1, t2 in self.gold_pairs:
                P.append(SimTupel(t1, t2, 1))

            trim_threshold = self.max_negative_pairs * 3
            candidates = candidates.difference(self.gold_pairs)
            self.npairs = len(candidates)
            N_bins = [set() for x in range(self.bins)]
            for index, (t1, t2) in enumerate(candidates):
                sim = self.tfidf_similarity(t1, t2)
                bin = int(sim * self.bins)
                if bin >= self.bins:
                    bin = self.bins - 1

                N_bins[bin].add(SimTupel(t1, t2, sim))

                if index % 50000 == 0:
                    logger.info("Processed %d candidates" % simcount)
                    simcount += 50000

            # Calculate probability distribution
            self.weights = [len(bin) for bin in N_bins]
            if self.verbose:
                logger.info(self.weights)

            wsum = sum(self.weights)
            self.weights[:] = [float(weight)/wsum for weight in self.weights]

            # Select pairs by probability distribution
            paircount = 0
            self.N_complete = N_bins
            while len(N) < self.max_negative_pairs:
                N.append(self.draw_pair())
                if len(N) % 10000 == 0:
                    logger.info("Selected %d negative pairs" % paircount)
                    paircount += 10000

        else:
            trim_threshold = self.max_negative_pairs * 3
            for index, (t1, t2) in enumerate(candidates):
                sim = self.tfidf_similarity(t1, t2)
                if sim >= self.upper_threshold:
                    P.append(SimTupel(t1, t2, sim))
                if sim < self.lower_threshold:
                    N.append(SimTupel(t1, t2, sim))

                if index % trim_threshold == 0:
                    P = heapq.nlargest(self.max_positive_pairs, P, key=lambda pair: pair.sim)

            P = heapq.nlargest(self.max_positive_pairs, P, key=lambda pair: pair.sim)
            self.N_complete = N
            self.N_complete.sort(key=lambda pair: pair.sim) # Sort ascending by similarity

            N = []
            while len(N) < self.max_negative_pairs:
                pair = self.draw_pair()
                if not pair:
                    break

                N.append(pair)

        if self.verbose:
            P_bins = [set() for x in range(20)]
            for t in P:
                sim = self.tfidf_similarity(t[0], t[1])
                bin = int(sim * 20)
                if bin >= 20:
                    bin = 20 - 1

                P_bins[bin].add(t)

            for index, _ in enumerate(P_bins):
                P_bins[index] = len(P_bins[index])

            logger.info(P_bins)

        return P, N

    def filter(self, blocking_scheme, P, N):

        def has_common_block(pair):
            p1_attributes = self.dataset[pair.t1]
            p2_attributes = self.dataset[pair.t2]
            # Add pairs whose attributes have a common block.
            for blocking_key in blocking_scheme:
                p1_bkvs = blocking_key.blocking_key_values(p1_attributes)
                p2_bkvs = blocking_key.blocking_key_values(p2_attributes)
                if not p1_bkvs.isdisjoint(p2_bkvs):
                    return True

            return False

        filtered_P = []
        for pair in P:
            if has_common_block(pair):
                filtered_P.append(pair)

        filtered_N = []
        for pair in N:
            if has_common_block(pair):
                filtered_N.append(pair)

        x = 0
        y = 0
        print("Filtered %d vs Max %d" % (len(filtered_N), self.max_negative_pairs))
        while len(filtered_N) < self.max_negative_pairs:
            pair = self.draw_pair()
            if not pair:
                break

            if has_common_block(pair):
                y += 1
                filtered_N.append(pair)

            x += 1

        print("Tried %d pairs hits %d" % (x, y))

        return filtered_P, filtered_N


BlockingKey = namedtuple('BlockingKey', ['field', 'encoder'])


class Feature:
    def __init__(self, blocking_keys, fsc):
        self.blocking_keys = blocking_keys
        self.fsc = fsc
        self.y_true = None
        self.y_pred = None
        self.illegal_bkvs = set()

    def union(self, other):
        bk_conjunction = self.blocking_keys + other.blocking_keys
        feature = Feature(bk_conjunction, None)
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

        return BKVs.difference(self.illegal_bkvs)

    def covered_fields(self):
        fields = set()
        for blocking_key in self.blocking_keys:
            fields.add(blocking_key.field)

        return fields

    def add_illegal_bkv(self, key_value):
        self.illegal_bkvs.add(key_value)

    def to_json(self):
        my = dict()
        my['fsc'] = self.fsc
        my['illegal_bkvs'] = list(self.illegal_bkvs)
        my['blocking_keys'] = \
            [(b.field, b.encoder.__name__) for b in self.blocking_keys]
        return json.dumps(my)

    @staticmethod
    def from_json(data):
        data = json.loads(data)
        possibles = globals().copy()
        possibles.update(locals())

        self = Feature(None, None)
        self.fsc = data['fsc']
        self.illegal_bkvs = set(data['illegal_bkvs'])
        print(data['blocking_keys'])
        self.blocking_keys = \
            [BlockingKey(b[0], possibles.get(b[1])) for b in data['blocking_keys']]
        return self

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

    def __init__(self, blocking_keys, P, N, indexer, k=2, d=3,
                 max_blocksize=100, min_goodratio=0.9,
                 block_timer=None, verbose=False):
        self.P = [frozenset((p.t1, p.t2)) for p in P]
        self.N = [frozenset((p.t1, p.t2)) for p in N]
        self.frozen_P = set(self.P)
        self.flat_P = set(hp.flatten(self.frozen_P))
        self.features = []
        self.k = k
        self.d = d
        self.max_blocksize = max_blocksize
        self.min_goodratio = min_goodratio
        self.indexer = indexer

        self.verbose = verbose
        self.block_timer = block_timer

        self.blocking_keys = blocking_keys
        self.features = []

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
        for depth in range(1, self.k + 1):
            keycombinations = it.combinations(self.blocking_keys, depth)
            for keycombi in keycombinations:
                feature = Feature(keycombi, None)
                # Calculate score
                feature.fsc = self.block_metrics(feature)
                self.features.append(feature)

    def block_metrics(self, feature):
        if self.verbose:
            logger.info("%r" % feature.signature())
            logger.info("0) Build blocks\t\t\t%s" % hp.memory_usage())

        blocks = self.indexer.blocks(feature, self.dataset)

        # Check frequency distribution of blocks
        block_sizes = []
        nblocks = len(blocks)
        for block_key, block in blocks:
            block_sizes.append(len(block))

        freq_dis = Counter(block_sizes)
        ngood = sum([freq_dis[b] * b for b in freq_dis.keys() if b > 1 and b < self.max_blocksize + 1])
        nbad = sum([freq_dis[b] * b for b in freq_dis.keys() if b > self.max_blocksize])
        # In case the blocking key assigns unique keys
        if ngood == 0:
            return 0

        # If there are too much big blocks disregard
        good_ratio = ngood / (ngood + nbad)
        if self.verbose:
            logger.info("1) Number of Blocks %d" % nblocks)
            logger.info("1) Frequency Ratio is %f\t%s" % (good_ratio, hp.memory_usage()))

        if good_ratio < self.min_goodratio:
            logger.info("2) Skip because of bad ratio")
            logger.info("----------------------------------------------------")
            return 0

        term_candidate_pairs = set()
        TP, FP, FN = 0, 0, 0
        for block_key, block in blocks:
            if len(block) > self.max_blocksize:
                feature.add_illegal_bkv(block_key)
                continue

            candidates = block
            # Generate candidate pairs
            cp_list = it.combinations(candidates, 2)
            block_candidate_pairs = set([frozenset(p) for p in cp_list])
            del cp_list

            # Calculate TP, FP, FN
            block_TP_pairs = block_candidate_pairs.intersection(self.frozen_P)
            block_FP_pairs = block_candidate_pairs.difference(block_TP_pairs)
            FP_canidates = set(hp.flatten(block_FP_pairs))

            TP += len(block_TP_pairs)
            FP += len(block_FP_pairs)
            FN += len(FP_canidates.intersection(self.flat_P))

            term_candidate_pairs.update(block_candidate_pairs)

        if self.verbose:
            logger.info("2) Generated candidates %s" % len(term_candidate_pairs))
            logger.info("3) Building ytrue/ypred\t\t%s" % hp.memory_usage())

        # Create feature vectors based on positive and negative labels
        y_true = np.zeros(len(self.P) + len(self.N), np.bool)
        y_pred = np.zeros(len(self.P) + len(self.N), np.bool)
        for index, pair in enumerate(self.P):
            if pair in term_candidate_pairs:
                y_true[index] = True
                y_pred[index] = True
            else:
                y_true[index] = True
                y_pred[index] = False

        for index, pair in enumerate(self.N):
            if pair in term_candidate_pairs:
                y_true[index + len(self.P)] = False
                y_pred[index + len(self.P)] = True
            else:
                y_true[index + len(self.P)] = False
                y_pred[index + len(self.P)] = False

        feature.y_true = y_true
        feature.y_pred = y_pred

        if self.verbose:
            logger.info("4) Calculate F1-Score\t\t%s" % hp.memory_usage())

        # No need to calculate score if there are no true positives
        if TP == 0:
            if self.verbose:
                logger.info("5) Skip because TP is 0\t%s" % hp.memory_usage())
                logger.info("----------------------------------------------------")

            return 0

        recall = TP / (FN + TP)
        precision = TP / (TP + FP)
        f1_score = 2 * ((precision * recall) / (precision + recall))

        # Force cleanup
        del blocks
        del term_candidate_pairs

        if self.verbose:
            logger.info("5) Calculated block metrics\t%s" % hp.memory_usage())
            logger.info("----------------------------------------------------")

        return f1_score

    def transform(self, dataset):
        self.dataset = dataset
        # Build and score features
        self.terms()

        if self.verbose:
            logger.info("Finding best blocking scheme")

        # Sort features
        Kf = []
        for feature in self.features:
            Kf.append(feature)

        Kf = filter(lambda feature: feature.fsc > 0, Kf)
        Kf = sorted(Kf, key=lambda feature: feature.fsc, reverse=True)
        top_10 = heapq.nlargest(1000, Kf, key=lambda feature: feature.fsc)

        blocking_scheme_candidates = []
        for index in range(self.d):
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


def bigrams(term):
    return set(''.join(x) for x in ngrams(term, 2))


def trigrams(term):
    return set(''.join(x) for x in ngrams(term, 3))


def prefixes(term):
    return set(term[:i+2] for i in np.arange(3))


def suffixes(term):
    return set(term[-(i+2):] for i in np.arange(3))
