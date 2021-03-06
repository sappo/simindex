# -*- coding: utf-8 -*-
import os
import subprocess
import numpy as np
import pandas as pd
import itertools as it
import psutil


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


def matches_count(q_id, result, gold_records):
    count = 0
    if q_id in gold_records.keys():
        for r_id in result.keys():
            if r_id in gold_records[q_id]:
                count += 1

    return count


def all_matches_count(id, gold_records):
    if id in gold_records.keys():
        return len(gold_records[id])
    else:
        return 0


def calc_micro_scores(q_id, result, probas, y_true, y_score, gold_records):
    # Only consider querys with relevant records this disregards
    # True negatives (TN)
    if q_id not in gold_records.keys():
        return

    for result_id in result.keys():
        result_score = probas[result_id][1]
        y_score.append(result_score)
        if result_id in gold_records[q_id]:
            # True Positive (TP)
            y_true.append(1)
        else:
            # False Positive (FP)
            y_true.append(0)

    # Fill in False Negatives (FN) with score 0.0
    for fn in gold_records[q_id].difference(result.keys()):
        y_true.append(1)
        y_score.append(0.)


def calc_micro_metrics(q_id, result, y_true, y_pred, gold_records):
    # Only consider querys with relevant records this disregards
    # True negatives (TN)
    if q_id in gold_records.keys():
        for result_id in result.keys():
                y_pred.append(1)
                if result_id in gold_records[q_id]:
                    # True Positive (TP)
                    y_true.append(1)
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
    for d in dicts:
        for k, v in d.items():
            collector[k] += v


def convert_bytes(n):
    symbols = ('K', 'M', 'G', 'T', 'P', 'E', 'Z', 'Y')
    prefix = {}
    for i, s in enumerate(symbols):
        prefix[s] = 1 << (i + 1) * 10

    for s in reversed(symbols):
        if n >= prefix[s]:
            value = float(n) / prefix[s]
            return '%.1f%s' % (value, s)

    return "%sB" % n


def memory_usage():
    p = psutil.Process(os.getpid())
    mem = p.memory_full_info()
    rss = mem.rss
    vms = mem.vms
    uss = mem.uss
    pss = getattr(mem, "pss", "")
    swap = getattr(mem, "swap", "")

    line = "RSS: %s\tVMS: %s\tUSS: %s\tPSS: %s\tSWAP: %s" % (
        convert_bytes(rss),
        convert_bytes(vms),
        convert_bytes(uss),
        convert_bytes(pss) if pss != "" else "",
        convert_bytes(swap) if swap != "" else ""
    )
    return line


def clamp(val, minimum=0, maximum=255):
    if val < minimum:
        return minimum
    if val > maximum:
        return maximum
    return int(val)


def colorscale(hexstr, scalefactor):
    """
    Scales a hex string by ``scalefactor``. Returns scaled hex string.

    To darken the color, use a float value between 0 and 1.
    To brighten the color, use a float value greater than 1.

    >>> colorscale("#DF3C3C", .5)
    #6F1E1E
    >>> colorscale("#52D24F", 1.6)
    #83FF7E
    >>> colorscale("#4F75D2", 1)
    #4F75D2
    """

    hexstr = hexstr.strip('#')

    if scalefactor < 0 or len(hexstr) != 6:
        return hexstr

    r, g, b = int(hexstr[:2], 16), int(hexstr[2:4], 16), int(hexstr[4:], 16)

    r = clamp(r * scalefactor)
    g = clamp(g * scalefactor)
    b = clamp(b * scalefactor)

    return "#%02x%02x%02x" % (r, g, b)
