from time import time
from simindex import DyLSH
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
    ["id_1", "id_2"]
]
n_thresholds = [0.01, 0.05, 0.08, 0.09, 0.1, 0.2, 0.3]
n_perms = range(2, 256)
for index, dataset in enumerate(datasets):
    fp = open("eval_lsh_%s.txt" % dataset, mode='w')
    for threshold in n_thresholds:
        for perm in n_perms:
            t0 = time()
            dy_lsh = DyLSH(_encode, _compare, threshold=1., top_n=3,
                           gold_standard="%s_gold.csv" % dataset,
                           gold_attributes=dataset_gold_attributes[index],
                           lsh_threshold=threshold, lsh_num_perm=perm)
            dy_lsh.fit("%s.csv" % dataset, dataset_attributes[index])
            fp.write("%f, %d, %f, %f\n" % (threshold,
                                           perm,
                                           dy_lsh.recall(),
                                           time() - t0))
            fp.flush()

    fp.close()
