#!/bin/bash
timestamp=$(date +%s)
mkdir -p ./reports

# Run ferbl datasets
#for dataset in "ferbl-4k-1k-1" "ferbl-9k-1k-1" "ferbl-90k-10k-1"
for dataset in "ferbl-4k-1k-1"
do
    prefix="../../master_thesis/datasets/febrl/"

    # Fit model once!
    mprof run analyzer.py \
        -t rec_id -t given_name -t surname -t suburb -t state \
        --run-type fit -r "${timestamp}" \
        ${prefix}${dataset}_index.csv \
        ${prefix}${dataset}_train_query.csv \
        ${prefix}${dataset}_train.csv \

    for indexer in "MDySimII" "MDySimIII" "MDyLSH"
    do
        result_output="./reports/${timestamp}_${indexer}_${dataset}"

        # Build and Query without metrics to get precice memory usage
        mprof run analyzer.py \
            -i rec_id -i given_name -i surname -i suburb -i state \
            -q rec_id -q given_name -q surname -q suburb -q state \
            -t rec_id -t given_name -t surname -t suburb -t state \
            -s ${prefix}${dataset}_train_gold.csv -g id_1 -g id_2 \
            --run-type build -r "${timestamp}" -m ${indexer} \
            ${prefix}${dataset}_index.csv \
            ${prefix}${dataset}_train_query.csv \
            ${prefix}${dataset}_train.csv \

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

    done
done

mv mprofile* ./reports
