#!/usr/bin/env python
# -*- coding: utf-8 -*-
def has_common_token(t1, t2):
    t1_tokens = set(t1.split())
    t2_tokens = set(t2.split())

    if len(t1_tokens.intersection(t2_tokens)) > 0:
        return 1
    else:
        return 0
