#!/usr/bin/env sh

# ########################################## #
#  Evaluate Ground Truth vs No Ground Truth  #
# ########################################## #

#Evaluation with ground truth
rm -f ./.engine/.ncvoter*
EVAL_FLAGS="--gt-labels" ./analyze.sh -n > /its/ksapp002/nohup.log
EVAL_FLAGS="--gt-labels --full-simvector" ./analyze.sh -n > /its/ksapp002/nohup.log
EVAL_FLAGS="--gt-labels --no-classifier" ./analyze.sh -n > /its/ksapp002/nohup.log

# Evaluation without ground truth
rm -f ./.engine/.ncvoter*
EVAL_FLAGS="--no-gt-labels" ./analyze.sh -n > /its/ksapp002/nohup.log
EVAL_FLAGS="--no-gt-labels --full-simvector" ./analyze.sh -n > /its/ksapp002/nohup.log
EVAL_FLAGS="--no-gt-labels --no-classifier" ./analyze.sh -n > /its/ksapp002/nohup.log
