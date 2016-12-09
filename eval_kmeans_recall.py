from simindex import DyKMeans
from difflib import SequenceMatcher


def _compare(a, b):
    return SequenceMatcher(None, a, b).ratio()


def _encode(a):
    return a[:1]

datasets = ["dataset1", "restaurant.csv", "ferbl-4k-1k-1"]
dataset_attributes = [
    ["rec_id", "given_name", "suburb", "surname", "postcode"],
    ["id", "name", "addr", "city", "phone"],
    ["rec_id", "given_name", "suburb", "surname", "postcode"]
]
dataset_gold_attributes = [
    ["org_id", "dup_id"],
    ["id_1", "id_2"],
    ["org_id", "dup_id"]
]
for index, dataset in enumerate(datasets):
    dy_kmeans = DyKMeans(_encode, _compare, threshold=1., top_n=3,
                         gold_standard="%s_gold.csv" % dataset,
                         gold_attributes=dataset_gold_attributes[index])
    dy_kmeans.fit_eval("%s.csv" % dataset, dataset_attributes[index])
