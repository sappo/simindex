#!/bin/bash

# ###################################################### #
#  Evaluation effects of different similarity functions  #
# ###################################################### #

# Baseline
if [ "$1" == 'trtr' ]; then
    if [ "$2" != '-b' ]; then
        rm -f ./.engine/.ncvoter*
        EVAL_FLAGS="--gt-labels --parfull-simvector" ./analyze.sh -n -x train train &> /its/ksapp002/nohup.log
    else
        rm -f ./.engine/.ncvoter*
        EVAL_FLAGS="--gt-labels --parfull-simvector" ./analyze.sh -n -b train train &> /its/ksapp002/nohup.log
    fi
fi

if [ "$1" == 'tete' ]; then
    if [ "$2" != '-b' ]; then
        rm -f ./.engine/.ncvoter*
        EVAL_FLAGS="--gt-labels --parfull-simvector" ./analyze.sh -n -x test test &> /its/ksapp002/nohup.log
    else
        rm -f ./.engine/.ncvoter*
        EVAL_FLAGS="--gt-labels --parfull-simvector" ./analyze.sh -n -b test test &> /its/ksapp002/nohup.log
    fi
fi

if [ "$1" == 'teva' ]; then
    if [ "$2" != '-b' ]; then
        rm -f ./.engine/.ncvoter*
        EVAL_FLAGS="--gt-labels --parfull-simvector" ./analyze.sh -n -x test validate &> /its/ksapp002/nohup.log
    else
        rm -f ./.engine/.ncvoter*
        EVAL_FLAGS="--gt-labels --parfull-simvector" ./analyze.sh -n -b test validate &> /its/ksapp002/nohup.log
    fi
fi
