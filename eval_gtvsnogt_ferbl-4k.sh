#!/usr/bin/env sh

# ########################################## #
#  Evaluate Ground Truth vs No Ground Truth  #
# ########################################## #

# Evaluation with ground truth
rm -f ./.engine/.ferbl-*
EVAL_FLAGS="--gt-labels" ./analyze.sh -f
EVAL_FLAGS="--gt-labels --full-simvector" ./analyze.sh -f
EVAL_FLAGS="--gt-labels --no-classifier" ./analyze.sh -f

# Evaluation without ground truth
rm -f ./.engine/.ferbl-*
EVAL_FLAGS="--no-gt-labels" ./analyze.sh -f
EVAL_FLAGS="--no-gt-labels --full-simvector" ./analyze.sh -f
EVAL_FLAGS="--no-gt-labels --no-classifier" ./analyze.sh -f
