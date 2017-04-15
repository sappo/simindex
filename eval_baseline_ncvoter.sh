#!/usr/bin/env sh

# ###################################################### #
#  Evaluation effects of different similarity functions  #
# ###################################################### #

# Baseline
if [ "$1" == 'trtr' ]; then
    rm -f ./.engine/.ncvoter*
    EVAL_FLAGS="--gt-labels --parfull-simvector" ./analyze.sh -n -x train train > /its/ksapp002/nohup.log
    rm -f ./.engine/.ncvoter*
    EVAL_FLAGS="--gt-labels --parfull-simvector" ./analyze.sh -n -b train train > /its/ksapp002/nohup.log
fi

if [ "$1" == 'tete' ]; then
    rm -f ./.engine/.ncvoter*
    EVAL_FLAGS="--gt-labels --parfull-simvector" ./analyze.sh -n -x test test > /its/ksapp002/nohup.log
    rm -f ./.engine/.ncvoter*
    EVAL_FLAGS="--gt-labels --parfull-simvector" ./analyze.sh -n -b test test > /its/ksapp002/nohup.log
fi

if [ "$1" == 'teva' ]; then
    rm -f ./.engine/.ncvoter*
    EVAL_FLAGS="--gt-labels --parfull-simvector" ./analyze.sh -n -x test test > /its/ksapp002/nohup.log
    rm -f ./.engine/.ncvoter*
    EVAL_FLAGS="--gt-labels --parfull-simvector" ./analyze.sh -n -b test test > /its/ksapp002/nohup.log
fi
