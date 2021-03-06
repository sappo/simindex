#!/bin/bash

# ###################################################### #
#  Evaluation effects of different similarity functions  #
# ###################################################### #

CLF=$1

# Cleanup
rm -f ./.engine/.ncvoter*

if [[ "$2" == '-1' || "$2" == '' ]]; then
# BAG distance
EVAL_FLAGS="$CLF --similarity=bag --gt-labels" ./analyze.sh -n > /its/ksapp002/nohup.log
EVAL_FLAGS="$CLF --similarity=bag --gt-labels --full-simvector" ./analyze.sh -n > /its/ksapp002/nohup.log
EVAL_FLAGS="$CLF --similarity=bag --gt-labels --parfull-simvector" ./analyze.sh -n > /its/ksapp002/nohup.log

# COMPRESSION distance
EVAL_FLAGS="$CLF --similarity=compression --gt-labels" ./analyze.sh -n > /its/ksapp002/nohup.log
EVAL_FLAGS="$CLF --similarity=compression --gt-labels --full-simvector" ./analyze.sh -n > /its/ksapp002/nohup.log
EVAL_FLAGS="$CLF --similarity=compression --gt-labels --parfull-simvector" ./analyze.sh -n > /its/ksapp002/nohup.log
fi

if [[ "$2" == '-2' || "$2" == '' ]]; then
# HAMMING distance
EVAL_FLAGS="$CLF --similarity=hamming --gt-labels" ./analyze.sh -n > /its/ksapp002/nohup.log
EVAL_FLAGS="$CLF --similarity=hamming --gt-labels --full-simvector" ./analyze.sh -n > /its/ksapp002/nohup.log
EVAL_FLAGS="$CLF --similarity=hamming --gt-labels --parfull-simvector" ./analyze.sh -n > /its/ksapp002/nohup.log

# LEVENSHTEIN distance
EVAL_FLAGS="$CLF --similarity=levenshtein --gt-labels" ./analyze.sh -n > /its/ksapp002/nohup.log
EVAL_FLAGS="$CLF --similarity=levenshtein --gt-labels --full-simvector" ./analyze.sh -n > /its/ksapp002/nohup.log
EVAL_FLAGS="$CLF --similarity=levenshtein --gt-labels --parfull-simvector" ./analyze.sh -n > /its/ksapp002/nohup.log
fi

if [[ "$2" == '-3' || "$2" == '' ]]; then
# JACCARD distance
EVAL_FLAGS="$CLF --similarity=jaccard --gt-labels" ./analyze.sh -n > /its/ksapp002/nohup.log
EVAL_FLAGS="$CLF --similarity=jaccard --gt-labels --full-simvector" ./analyze.sh -n > /its/ksapp002/nohup.log
EVAL_FLAGS="$CLF --similarity=jaccard --gt-labels --parfull-simvector" ./analyze.sh -n > /its/ksapp002/nohup.log

# JARO distance
EVAL_FLAGS="$CLF --similarity=jaro --gt-labels" ./analyze.sh -n > /its/ksapp002/nohup.log
EVAL_FLAGS="$CLF --similarity=jaro --gt-labels --full-simvector" ./analyze.sh -n > /its/ksapp002/nohup.log
EVAL_FLAGS="$CLF --similarity=jaro --gt-labels --parfull-simvector" ./analyze.sh -n > /its/ksapp002/nohup.log
fi

if [[ "$2" == '-4' || "$2" == '' ]]; then
# JARO-WINKLER distance
EVAL_FLAGS="$CLF --similarity=jarowinkler --gt-labels" ./analyze.sh -n > /its/ksapp002/nohup.log
EVAL_FLAGS="$CLF --similarity=jarowinkler --gt-labels --full-simvector" ./analyze.sh -n > /its/ksapp002/nohup.log
EVAL_FLAGS="$CLF --similarity=jarowinkler --gt-labels --parfull-simvector" ./analyze.sh -n > /its/ksapp002/nohup.log

# PYTHON RATIO distance
EVAL_FLAGS="$CLF --similarity=ratio --gt-labels" ./analyze.sh -n > /its/ksapp002/nohup.log
EVAL_FLAGS="$CLF --similarity=ratio --gt-labels --full-simvector" ./analyze.sh -n > /its/ksapp002/nohup.log
EVAL_FLAGS="$CLF --similarity=ratio --gt-labels --parfull-simvector" ./analyze.sh -n > /its/ksapp002/nohup.log
fi
