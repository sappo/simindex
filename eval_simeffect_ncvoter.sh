#!/usr/bin/env sh

# ###################################################### #
#  Evaluation effects of different similarity functions  #
# ###################################################### #

CLF=$1

# Cleanup
rm -f ./.engine/.ncvoter*

# BAG distance
EVAL_FLAGS="$CLF --similarity=bag --gt-labels" ./analyze.sh -n > /its/ksapp002/nohup.log
EVAL_FLAGS="$CLF --similarity=bag --gt-labels --full-simvector" ./analyze.sh -n > /its/ksapp002/nohup.log

# LEVENSHTEIN distance
EVAL_FLAGS="$CLF --similarity=levenshtein --gt-labels" ./analyze.sh -n > /its/ksapp002/nohup.log
EVAL_FLAGS="$CLF --similarity=levenshtein --gt-labels --full-simvector" ./analyze.sh -n > /its/ksapp002/nohup.log

# JARO-WINKLER distance
EVAL_FLAGS="$CLF --similarity=jaro --gt-labels" ./analyze.sh -n > /its/ksapp002/nohup.log
EVAL_FLAGS="$CLF --similarity=jaro --gt-labels --full-simvector" ./analyze.sh -n > /its/ksapp002/nohup.log

# LEE distance
EVAL_FLAGS="$CLF --similarity=ratio --gt-labels" ./analyze.sh -n > /its/ksapp002/nohup.log
EVAL_FLAGS="$CLF --similarity=ratio --gt-labels --full-simvector" ./analyze.sh -n > /its/ksapp002/nohup.log

# JACCARD distance
EVAL_FLAGS="$CLF --similarity=jaccard --gt-labels" ./analyze.sh -n > /its/ksapp002/nohup.log
EVAL_FLAGS="$CLF --similarity=jaccard --gt-labels --full-simvector" ./analyze.sh -n > /its/ksapp002/nohup.log
