#!/usr/bin/env sh

# ###################################################### #
#  Evaluation effects of different similarity functions  #
# ###################################################### #

CLF=$1

# Baseline
rm -f ./.engine/.ncvoter*
EVAL_FLAGS="--gt-labels --no-classifier" ./analyze.sh -n > /its/ksapp002/nohup.log

# BAG distance
EVAL_FLAGS="$CLF --similarity=bag --gt-labels" ./analyze.sh -n > /its/ksapp002/nohup.log
EVAL_FLAGS="$CLF --similarity=bag --gt-labels --full-simvector" ./analyze.sh -n > /its/ksapp002/nohup.log

# LEVENSHTEIN distance
EVAL_FLAGS="$CLF --similarity=levenshtein --gt-labels" ./analyze.sh -n > /its/ksapp002/nohup.log
EVAL_FLAGS="$CLF --similarity=levenshtein --gt-labels --full-simvector" ./analyze.sh -n > /its/ksapp002/nohup.log

# JARO distance
EVAL_FLAGS="$CLF --similarity=jaro --gt-labels" ./analyze.sh -n > /its/ksapp002/nohup.log
EVAL_FLAGS="$CLF --similarity=jaro --gt-labels --full-simvector" ./analyze.sh -n > /its/ksapp002/nohup.log

# Python RATIO distance
EVAL_FLAGS="$CLF --similarity=ratio --gt-labels" ./analyze.sh -n > /its/ksapp002/nohup.log
EVAL_FLAGS="$CLF --similarity=ratio --gt-labels --full-simvector" ./analyze.sh -n > /its/ksapp002/nohup.log
