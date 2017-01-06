# -*- coding: utf-8 -*-
import csv


def read_csv(filename, attributes=[], percentage=1.0, delimiter=','):
    lines = []
    columns = []
    with open(filename, newline='', encoding='utf-8', errors='ignore') as csvfile:
        reader = csv.reader(csvfile, delimiter=delimiter, quotechar='"')
        data = list(reader)
        row_count = len(data)
        threshold = int(row_count * percentage)
        for index, row in enumerate(data[:threshold]):
            if index == 0:
                for x, field in enumerate(row):
                    row[x] = str(field).strip()
                for attribute in attributes:
                    columns.append(row.index(attribute))
            else:
                if len(columns) > 0:
                    line = []
                    for col in columns:
                        line.append(str(row[col]).strip())
                    lines.append(line)
                else:
                    lines.append(row)

    return lines

def calc_micro_scores(q_id, result, y_true, y_score, gold_records):
    # Disregards True negatives (TN)
    for a in result.keys():
        # Only consider querys with relevant records
        if q_id in gold_records.keys():
            y_score.append(result[a])
            if a in gold_records[q_id]:
                # True Positive (TP)
                y_true.append(1)
            else:
                # False Positive (FP)
                y_true.append(0)
                # print("For", q_id, "falsly predicted:", a, "with score", result[a], ".")

    # Fill in False Negatives (FN) with score 0.0
    if q_id in gold_records:
        for fn in gold_records[q_id].difference(result.keys()):
            y_true.append(1)
            y_score.append(0.)

def calc_micro_metrics(q_id, result, y_true, y_pred, gold_records):
    # Disregards True negatives (TN)
    for a in result.keys():
        # Only consider querys with relevant records
        if q_id in gold_records.keys():
            y_pred.append(1)
            if a in gold_records[q_id]:
                # True Positive (TP)
                y_true.append(1)
            else:
                # False Positive (FP)
                y_true.append(0)

    # Fill in False Negatives (FN)
    if q_id in gold_records:
        for fn in gold_records[q_id].difference(result.keys()):
            y_true.append(1)
            y_pred.append(0)