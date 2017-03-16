#!/usr/bin/env sh

# ###################################################### #
#  Evaluation effects of different similarity functions  #
# ###################################################### #

# Baseline
rm -f ./.engine/.ferbl-*
EVAL_FLAGS="--gt-labels --no-classifier" ./analyze.sh -f

# BAG distance
EVAL_FLAGS="--similarity=bag --gt-labels" ./analyze.sh -f
EVAL_FLAGS="--similarity=bag --gt-labels --full-simvector" ./analyze.sh -f

# LEVENSHTEIN distance
EVAL_FLAGS="--similarity=levenshtein --gt-labels" ./analyze.sh -f
EVAL_FLAGS="--similarity=levenshtein --gt-labels --full-simvector" ./analyze.sh -f

# JARO distance
EVAL_FLAGS="--similarity=jaro --gt-labels" ./analyze.sh -f
EVAL_FLAGS="--similarity=jaro --gt-labels --full-simvector" ./analyze.sh -f

# Python RATIO distance
EVAL_FLAGS="--similarity=ratio --gt-labels" ./analyze.sh -f
EVAL_FLAGS="--similarity=ratio --gt-labels --full-simvector" ./analyze.sh -f
