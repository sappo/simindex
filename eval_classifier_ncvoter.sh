#!/usr/bin/env sh

# ############################################# #
#  Evaluate influence of different classifiers  #
# ############################################# #

# Cleanup
rm -f ./.engine/.ncvoter*

# Evaluation with SVM decision tree
if [[ "$1" == '-1' || "$1" == '' ]]; then
EVAL_FLAGS="--clf=decisiontree --gt-labels" ./analyze.sh -n > /its/ksapp002/nohup.log
EVAL_FLAGS="--clf=decisiontree --gt-labels --parfull-simvector" ./analyze.sh -n > /its/ksapp002/nohup.log
EVAL_FLAGS="--clf=decisiontree --gt-labels --full-simvector" ./analyze.sh -n > /its/ksapp002/nohup.log
fi

# Evaluation with SVM linear
if [[ "$1" == '-2' || "$1" == '' ]]; then
EVAL_FLAGS="--clf=svmlinear --gt-labels" ./analyze.sh -n > /its/ksapp002/nohup.log
EVAL_FLAGS="--clf=svmlinear --gt-labels --parfull-simvector" ./analyze.sh -n > /its/ksapp002/nohup.log
EVAL_FLAGS="--clf=svmlinear --gt-labels --full-simvector" ./analyze.sh -n > /its/ksapp002/nohup.log
fi

# Evaluation with SVM rbf
if [[ "$1" == '-3' || "$1" == '' ]]; then
EVAL_FLAGS="--clf=svmrbf --gt-labels" ./analyze.sh -n > /its/ksapp002/nohup.log
EVAL_FLAGS="--clf=svmrbf --gt-labels --parfull-simvector" ./analyze.sh -n > /its/ksapp002/nohup.log
EVAL_FLAGS="--clf=svmrbf --gt-labels --full-simvector" ./analyze.sh -n > /its/ksapp002/nohup.log
fi
