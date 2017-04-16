#!/bin/bash

# ########################################## #
#  Evaluate Ground Truth vs No Ground Truth  #
# ########################################## #

if [ "$1" == 'gt' ]; then
    #Evaluation with ground truth
    rm -f ./.engine/.ncvoter*
    EVAL_FLAGS="--gt-labels --parfull-simvector --no-classifier" ./analyze.sh -n -x test test &> /its/ksapp002/nohup.log
    EVAL_FLAGS="--gt-labels --parfull-simvector" ./analyze.sh -n -x test test &> /its/ksapp002/nohup.log
    EVAL_FLAGS="--gt-labels --parfull-simvector" ./analyze.sh -n -x test validate &> /its/ksapp002/nohup.log
fi

if [ "$1" == 'nogt' ]; then
    # Evaluation without ground truth
    rm -f ./.engine/.ncvoter*
    EVAL_FLAGS="--no-gt-labels --parfull-simvector --no-classifier" ./analyze.sh -n -x test test &> /its/ksapp002/nohup.log
    EVAL_FLAGS="--no-gt-labels --parfull-simvector" ./analyze.sh -n -x test test &> /its/ksapp002/nohup.log
    EVAL_FLAGS="--no-gt-labels --parfull-simvector" ./analyze.sh -n -x test validate &> /its/ksapp002/nohup.log
fi
