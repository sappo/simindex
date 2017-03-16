#!/usr/bin/env sh

# ############################################# #
#  Evaluate influence of different classifiers  #
# ############################################# #

# Evaluation with SVM linear
rm -f ./.engine/.ncvoter*
EVAL_FLAGS="--clf=svmlinear --gt-labels" ./analyze.sh -n > /its/ksapp002/nohup.log
EVAL_FLAGS="--clf=svmlinear --gt-labels --full-simvector" ./analyze.sh -n > /its/ksapp002/nohup.log

# Evaluation with SVM rbf
rm -f ./.engine/.ncvoter*
EVAL_FLAGS="--clf=svmrbf --gt-labels" ./analyze.sh -n > /its/ksapp002/nohup.log
EVAL_FLAGS="--clf=svmrbf --gt-labels --full-simvector" ./analyze.sh -n > /its/ksapp002/nohup.log

# Evaluation with SVM decision tree
rm -f ./.engine/.ncvoter*
EVAL_FLAGS="--clf=decisiontree --gt-labels" ./analyze.sh -n > /its/ksapp002/nohup.log
EVAL_FLAGS="--clf=decisiontree --gt-labels --full-simvector" ./analyze.sh -n > /its/ksapp002/nohup.log
