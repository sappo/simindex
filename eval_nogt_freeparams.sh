 #!/usr/bin/env sh

 # ################################################## #
 #  Evaluation effects of different no gt thresholds  #
 # ################################################## #

# Thresholds
UPPER=$1
WINDOW=$2
MAX_P=$3
MAX_N=$4
CLF=$5
for i in $(seq 1 10);
do
    LOWER=`echo - | awk "{print $i / 10}"`
    if [ `echo - | awk "{print $LOWER <= $UPPER}"` -ne 0 ]; then
        rm -f ./.engine/.ncvoter*
        EVAL_FLAGS="$CLF --no-classifier --no-gt-labels --gt-thresholds $LOWER $UPPER $WINDOW $MAX_P $MAX_N" ./analyze.sh -n
    fi
done
