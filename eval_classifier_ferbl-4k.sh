#!/usr/bin/env sh

# ############################################# #
#  Evaluate influence of different classifiers  #
# ############################################# #

# Evaluation with SVM linear
rm -f ./.engine/.ferblr*
EVAL_FLAGS="--clf=svmlinear --gt-labels" ./analyze.sh -f
EVAL_FLAGS="--clf=svmlinear --gt-labels --full-simvector" -f

# Evaluation with SVM rbf
rm -f ./.engine/.ferblr*
EVAL_FLAGS="--clf=svmrbf --gt-labels" ./analyze.sh -f
EVAL_FLAGS="--clf=svmrbf --gt-labels --full-simvector" ./analyze.sh -f

# Evaluation with SVM decision tree
rm -f ./.engine/.ferblr*
EVAL_FLAGS="--clf=decisiontree --gt-labels" ./analyze.sh -f
EVAL_FLAGS="--clf=decisiontree --gt-labels --full-simvector" ./analyze.sh -f

