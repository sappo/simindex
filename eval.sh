#!/usr/bin/env sh
# Evaluation with ground truth
rm ./.engine/.*
EVAL_FLAGS="--gt-labels" ./analyze.sh -n > /its/ksapp002/nohup.log
EVAL_FLAGS="--gt-labels --full-simvector" ./analyze.sh -n > /its/ksapp002/nohup.log
EVAL_FLAGS="--gt-labels --no-classifier" ./analyze.sh -n > /its/ksapp002/nohup.log
# Evaluation without ground truth
rm ./.engine/.*
EVAL_FLAGS="--no-gt-labels" ./analyze.sh -n > /its/ksapp002/nohup.log
EVAL_FLAGS="--no-gt-labels --full-simvector" ./analyze.sh -n > /its/ksapp002/nohup.log
EVAL_FLAGS="--no-gt-labels --no-classifier" ./analyze.sh -n > /its/ksapp002/nohup.log
