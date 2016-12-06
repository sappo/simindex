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
