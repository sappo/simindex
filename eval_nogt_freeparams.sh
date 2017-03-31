 #!/usr/bin/env sh

 # ################################################## #
 #  Evaluation effects of different no gt thresholds  #
 # ################################################## #

# Thresholds
UPPER=$1
WINDOW=$2
CLF=$3
for i in $(seq 1 10);
do
    LOWER=`echo - | awk "{print $i / 10}"`
    if [ `echo - | awk "{print $LOWER <= $UPPER}"` -ne 0 ]; then
        rm -f ./.engine/.ncvoter*
        EVAL_FLAGS="$CLF --no-classifier --no-gt-labels --gt-thresholds $LOWER $UPPER $WINDOW 0.1 0.25" ./analyze.sh -n
    fi
done
