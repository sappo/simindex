#!/usr/bin/env sh

# ############################################# #
#  Evaluate influence of different classifiers  #
# ############################################# #

CLF=$1

# Cleanup
rm -f ./.engine/.ncvoter*

# Evaluation with different scoring methods
EVAL_FLAGS="$CLF --classifier-scoring=average_precision" ./analyze.sh -n > /its/ksapp002/nohup.log
EVAL_FLAGS="$CLF --classifier-scoring=f1" ./analyze.sh -n > /its/ksapp002/nohup.log
EVAL_FLAGS="$CLF --classifier-scoring=recall" ./analyze.sh -n > /its/ksapp002/nohup.log
