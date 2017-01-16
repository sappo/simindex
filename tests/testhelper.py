#!/usr/bin/env python
# -*- coding: utf-8 -*-

from simindex.weak_labels import stoplist


def has_common_token(t1, t2):
    t1_token = t1.lower().split()
    t2_token = t2.lower().split()
    t1_tokens = set(word for word in t1_token if word not in stoplist)
    t2_tokens = set(word for word in t2_token if word not in stoplist)

    if len(t1_tokens.intersection(t2_tokens)) > 0:
        return 1
    else:
        return 0
