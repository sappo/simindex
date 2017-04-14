#!/usr/bin/env sh

# ###################################################### #
#  Evaluation effects of different similarity functions  #
# ###################################################### #

# Baseline
rm -f ./.engine/.ncvoter*
EVAL_FLAGS="--gt-labels --no-classifier" ./analyze.sh -n > /its/ksapp002/nohup.log
EVAL_FLAGS="--gt-labels --no-classifier --full-simvector" ./analyze.sh -n > /its/ksapp002/nohup.log
EVAL_FLAGS="--gt-labels --no-classifier --parfull-simvector" ./analyze.sh -n > /its/ksapp002/nohup.log
