#!/bin/bash
timestamp=$(date +%Y%m%d-%H%M%S)
reportprefix="./reports"
mkdir -p $reportprefix

# Run ferbl datasets
for dataset in "ferbl-4k-1k-1" "ferbl-9k-1k-1" "ferbl-90k-10k-1"
do
    datasetprefix="../../master_thesis/datasets/febrl/"

    if [ "$1" == '-b' ]; then
        # Fit baseline model once!
        mprof run analyzer.py \
            --run-type fit -r "${timestamp}" \
            -o "${reportprefix}/${timestamp}_fit_${dataset}" \
            -s ${datasetprefix}${dataset}_train_gold.csv -g id_1 -g id_2 \
            -t rec_id -t given_name -t surname -t suburb -t state -t postcode -t address_1 -t date_of_birth -t phone_number \
            -b "0 surname term_id" \
            -b "0 address_1 tokens" \
            -b "0 postcode term_id" \
            -b "1 given_name term_id" \
            -b "1 date_of_birth term_id" \
            -b "2 phone_number term_id" \
            ${datasetprefix}${dataset}_index.csv \
            ${datasetprefix}${dataset}_train_query.csv \
            ${datasetprefix}${dataset}_train.csv
    else
        # Fit model once!
        mprof run analyzer.py \
            --run-type fit -r "${timestamp}" \
            -o "${reportprefix}/${timestamp}_fit_${dataset}" \
            -s ${datasetprefix}${dataset}_train_gold.csv -g id_1 -g id_2 \
            -t rec_id -t given_name -t surname -t suburb -t state -t postcode -t address_1 -t date_of_birth -t phone_number \
            ${datasetprefix}${dataset}_index.csv \
            ${datasetprefix}${dataset}_train_query.csv \
            ${datasetprefix}${dataset}_train.csv
    fi

    for indexer in "MDySimII" "MDySimIII" "MDyLSH"
    do
        result_output="${reportprefix}/${timestamp}_${indexer}_${dataset}"

        # Build and Query without metrics to get precice memory usage
        mprof run analyzer.py \
            -i rec_id -i given_name -i surname -i suburb -i state -i postcode -i address_1 -i date_of_birth -i phone_number \
            -q rec_id -q given_name -q surname -q suburb -q state -q postcode -q address_1 -q date_of_birth -q phone_number \
            -t rec_id -t given_name -t surname -t suburb -t state -t postcode -t address_1 -t date_of_birth -t phone_number \
            -s ${datasetprefix}${dataset}_train_gold.csv -g id_1 -g id_2 \
            --run-type build -r "${timestamp}" -m ${indexer} \
            ${datasetprefix}${dataset}_index.csv \
            ${datasetprefix}${dataset}_train_query.csv \
            ${datasetprefix}${dataset}_train.csv

        # Calculate metrics on dataset
        python -W ignore analyzer.py \
            -i rec_id -i given_name -i surname -i suburb -i state -i postcode -i address_1 -i date_of_birth -i phone_number \
            -q rec_id -q given_name -q surname -q suburb -q state -q postcode -q address_1 -q date_of_birth -q phone_number \
            -t rec_id -t given_name -t surname -t suburb -t state -t postcode -t address_1 -t date_of_birth -t phone_number \
            -s ${datasetprefix}${dataset}_train_gold.csv -g id_1 -g id_2 \
            --run-type evaluation -o $result_output \
            -m ${indexer} \
            ${datasetprefix}${dataset}_index.csv \
            ${datasetprefix}${dataset}_train_query.csv \
            ${datasetprefix}${dataset}_train.csv

    done
done

mv mprofile* $reportprefix
