#!/usr/bin/env sh

# ############################################# #
#  Evaluate influence of different classifiers  #
# ############################################# #

# Cleanup
rm -f ./.engine/.ncvoter*

# Evaluation with SVM decision tree
EVAL_FLAGS="--clf=decisiontree --gt-labels" ./analyze.sh -n > /its/ksapp002/nohup.log
EVAL_FLAGS="--clf=decisiontree --gt-labels --parfull-simvector" ./analyze.sh -n > /its/ksapp002/nohup.log
EVAL_FLAGS="--clf=decisiontree --gt-labels --full-simvector" ./analyze.sh -n > /its/ksapp002/nohup.log

# Evaluation with SVM linear
EVAL_FLAGS="--clf=svmlinear --gt-labels" ./analyze.sh -n > /its/ksapp002/nohup.log
EVAL_FLAGS="--clf=svmlinear --gt-labels --parfull-simvector" ./analyze.sh -n > /its/ksapp002/nohup.log
EVAL_FLAGS="--clf=svmlinear --gt-labels --full-simvector" ./analyze.sh -n > /its/ksapp002/nohup.log

# Evaluation with SVM rbf
EVAL_FLAGS="--clf=svmrbf --gt-labels" ./analyze.sh -n > /its/ksapp002/nohup.log
EVAL_FLAGS="--clf=svmrbf --gt-labels --parfull-simvector" ./analyze.sh -n > /its/ksapp002/nohup.log
EVAL_FLAGS="--clf=svmrbf --gt-labels --full-simvector" ./analyze.sh -n > /its/ksapp002/nohup.log
