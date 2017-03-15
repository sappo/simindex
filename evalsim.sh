#!/usr/bin/env sh
EVAL_FLAGS="--similarity=bag --gt-labels" ./analyze.sh -f
EVAL_FLAGS="--similarity=bag --gt-labels --full-simvector" ./analyze.sh -f
EVAL_FLAGS="--similarity=levenshtein --gt-labels" ./analyze.sh -f
EVAL_FLAGS="--similarity=levenshtein --gt-labels --full-simvector" ./analyze.sh -f
EVAL_FLAGS="--similarity=jaro --gt-labels" ./analyze.sh -f
EVAL_FLAGS="--similarity=jaro --gt-labels --full-simvector" ./analyze.sh -f
EVAL_FLAGS="--similarity=ratio --gt-labels" ./analyze.sh -f
EVAL_FLAGS="--similarity=ratio --gt-labels --full-simvector" ./analyze.sh -f
