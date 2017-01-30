# -*- coding: utf-8 -*-
import subprocess
import pandas as pd
import itertools as it


def read_csv(filename, attributes=None, percentage=1.0,
             delimiter=',', dtype='unicode', encoding='iso-8859-1'):
    csv_chunks = pd.read_csv(filename,
                             usecols=attributes,
                             skipinitialspace=True,
                             iterator=True,
                             chunksize=10000,
                             error_bad_lines=False,
                             index_col=False,
                             # dtype=dtype,
                             encoding=encoding)

    for chunk in csv_chunks:
        chunk.fillna('', inplace=True)
        for row in chunk.values:
            yield row


def write_csv(filename, data, mode='w', compression=None):
    df = pd.DataFrame(data)
    df.to_csv(filename, mode=mode, index=False,
              encoding="utf-8", chunksize=10000, compression=compression)


def hdf_record_attributes(store, group, columns=None):
    if not store.is_open:
        store.open()

    if columns:
        frames = store.select_column(group, where="columns == %r" % columns,
                              iterator=True, chunksize=50000)
    else:
        frames = store.select(group, iterator=True, chunksize=50000)
    for df in frames:
        df.fillna('', inplace=True)
        for row in df.values:
            yield row

    store.close()


def hdf_records(store, group, query=None):
    if not store.is_open:
        store.open()

    frames = store.select(group, where=query, iterator=True, chunksize=50000)
    for df in frames:

        df.fillna('', inplace=True)
        for record in df.to_records():
            yield list(record)

    store.close()


def prepare_record_fitting(dataset, ground_truth):
    X = []
    y = []
    for a_id, b_id in ground_truth:
        X.append((a_id, b_id))
        y.append(1)

    return X, y


def calc_micro_scores(q_id, result, y_true, y_score, gold_records):
    # Only consider querys with relevant records
    if q_id in gold_records.keys():
        # Disregards True negatives (TN)
        for result_id in result.keys():
                result_score = result[result_id]
                y_score.append(result_score)
                if result_id in gold_records[q_id]:
                    # True Positive (TP)
                    y_true.append(1)
                else:
                    # False Positive (FP)
                    y_true.append(0)
                    # print("For", q_id, "falsly predicted:", a, "with score", result[a], ".")

        # Fill in False Negatives (FN) with score 0.0
        for fn in gold_records[q_id].difference(result.keys()):
            y_true.append(1)
            y_score.append(0.)

def calc_micro_metrics(q_id, result, y_true, y_pred, gold_records):
    test = False
    # Only consider querys with relevant records
    if q_id in gold_records.keys():
        # Disregards True negatives (TN)
        for result_id in result.keys():
                y_pred.append(1)
                if result_id in gold_records[q_id]:
                    # True Positive (TP)
                    y_true.append(1)
                    test = True
                else:
                    # False Positive (FP)
                    y_true.append(0)

        # Fill in False Negatives (FN)
        for fn in gold_records[q_id].difference(result.keys()):
            y_true.append(1)
            y_pred.append(0)

def grouper(iterable, n, fillvalue=None):
    "Collect data into fixed-length chunks or blocks"
    # grouper('ABCDEFG', 3, 'x') --> ABC DEF Gxx"
    args = [iter(iterable)] * n
    return it.zip_longest(*args, fillvalue=fillvalue)


def flatten(listOfLists):
    "Flatten one level of nesting"
    return it.chain.from_iterable(listOfLists)


def file_len(filename):
    p = subprocess.Popen(['wc', '-l', filename], stdout=subprocess.PIPE,
                                              stderr=subprocess.PIPE)
    result, err = p.communicate()
    if p.returncode != 0:
        raise IOError(err)
    return int(result.strip().split()[0])


def merge_sum(collector, *dicts):
    print(collector)
    for d in dicts:
        for k, v in d.items():
            print(k)
            collector[k] += v
