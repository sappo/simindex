#!/bin/bash
timestamp=$(date +%s)

# Run ferbl datasets
for indexer in "MDySimII" "MDySimIII" "MDyLSH"
do
##for dataset in "ferbl-4k-1k-1" "ferbl-9k-1k-1" "ferbl-90k-10k-1"
for dataset in "ferbl-90k-1k-1"
do
    prefix="../../master_thesis/datasets/febrl/"
    result_output="${timestamp}_${indexer}_${dataset}"
    # Measure insert/query times on restaurant dataset
    # Calculate metrics on dataset
    python -W ignore analyzer.py \
        -i rec_id -i given_name -i surname -i suburb -i state \
        -q rec_id -q given_name -q surname -q suburb -q state \
        -t rec_id -t given_name -t surname -t suburb -t state \
        -s ${prefix}${dataset}_train_gold.csv -g id_1 -g id_2 \
        --run-type evaluation -o $result_output \
        -m ${indexer} \
        ${prefix}${dataset}_index.csv \
        ${prefix}${dataset}_train_query.csv \
        ${prefix}${dataset}_train.csv \

    ## Measure peak memory and index build time
    #PEAK=$(./memusg.sh python measures.py \
        #-i rec_id -i given_name -i suburb -i surname \
        #-q rec_id -q given_name -q suburb -q surname \
        #-e soundex -e soundex -e soundex \
        #-c default -c default -c default \
        #-t index -o $result_output \
        #-m ${indexer} \
        #${prefix}${dataset}_index.csv ${prefix}${dataset}_train_query.csv)

    #sed -i '$ d' $result_output
    #echo "    ,\"memory_usage\":" $PEAK >> $result_output
    #echo "}" >> $result_output
done
done
