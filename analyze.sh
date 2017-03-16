#!/bin/bash
timestamp=$(date +%Y%m%d-%H%M%S)
reportprefix="./reports"
mkdir -p $reportprefix

if [ "$1" == '-f' ]; then
# Run ferbl datasets
#for dataset in "ferbl-4k-1k-1" "ferbl-9k-1k-1" "ferbl-90k-10k-1"
for dataset in "ferbl-4k-1k-1"
do
    ###########################################################################
    datasetprefix="../../master_thesis/datasets/febrl/"
    ###########################################################################

    #for indexer in "MDySimII" "MDySimIII" "MDyLSH"
    for indexer in "MDySimIII"
    do
        if [ "$2" == '-b' ]; then
            # Fit baseline model once!
            (
            mprof run analyzer.py \
                --run-type fit -r "${timestamp}" \
                -o "${reportprefix}/${timestamp}_fit_${dataset}" \
                -s ${datasetprefix}${dataset}_train_gold.csv -g id_1 -g id_2 \
                -t rec_id -t date_of_birth -t given_name -t surname -t state -t suburb -t postcode -t address_1 -t phone_number \
                -b "0 surname term_id" \
                -b "0 address_1 tokens" \
                -b "0 postcode term_id" \
                -b "1 given_name term_id" \
                -b "1 date_of_birth term_id" \
                -b "2 phone_number term_id" \
                -m ${indexer} \
                $EVAL_FLAGS \
                ${datasetprefix}${dataset}_index.csv \
                ${datasetprefix}${dataset}_train_query.csv \
                ${datasetprefix}${dataset}_train.csv
            ) || exit 1
        else
            # Fit model once!
            (
            mprof run analyzer.py \
                --run-type fit -r "${timestamp}" \
                -o "${reportprefix}/${timestamp}_fit_${dataset}" \
                -s ${datasetprefix}${dataset}_train_gold.csv -g id_1 -g id_2 \
                -t rec_id -t date_of_birth -t given_name -t surname -t state -t suburb -t postcode -t address_1 -t phone_number \
                -m ${indexer} \
                $EVAL_FLAGS \
                ${datasetprefix}${dataset}_index.csv \
                ${datasetprefix}${dataset}_train_query.csv \
                ${datasetprefix}${dataset}_train.csv
            ) || exit 1
        fi

        result_output="${reportprefix}/${timestamp}_${indexer}_${dataset}"

        # Build and Query without metrics to get precice memory usage
        (
        mprof run analyzer.py \
            -i rec_id -i given_name -i surname -i suburb -i state -i postcode -i address_1 -i date_of_birth -i phone_number \
            -q rec_id -q given_name -q surname -q suburb -q state -q postcode -q address_1 -q date_of_birth -q phone_number \
            -t rec_id -t given_name -t surname -t suburb -t state -t postcode -t address_1 -t date_of_birth -t phone_number \
            -s ${datasetprefix}${dataset}_train_gold.csv -g id_1 -g id_2 \
            --run-type build -r "${timestamp}" -m ${indexer} \
            $EVAL_FLAGS \
            ${datasetprefix}${dataset}_index.csv \
            ${datasetprefix}${dataset}_train_query.csv \
            ${datasetprefix}${dataset}_train.csv
        ) || exit 1

        # Calculate metrics on dataset
        (
        python -W ignore analyzer.py \
            -i rec_id -i given_name -i surname -i suburb -i state -i postcode -i address_1 -i date_of_birth -i phone_number \
            -q rec_id -q given_name -q surname -q suburb -q state -q postcode -q address_1 -q date_of_birth -q phone_number \
            -t rec_id -t given_name -t surname -t suburb -t state -t postcode -t address_1 -t date_of_birth -t phone_number \
            -s ${datasetprefix}${dataset}_train_gold.csv -g id_1 -g id_2 \
            --run-type evaluation -o $result_output \
            -m ${indexer} \
            $EVAL_FLAGS \
            ${datasetprefix}${dataset}_index.csv \
            ${datasetprefix}${dataset}_train_query.csv \
            ${datasetprefix}${dataset}_train.csv
        ) || exit 1

    done
done
elif [ "$1" == '-n' ]; then
###############################################################################
dataset="ncvoter"
datasetprefix="../../master_thesis/datasets/ncvoter/"
###############################################################################

for indexer in "MDySimIII"
do
    if [ "$2" == '-b' ]; then
        # Fit baseline model once!
        mprof run analyzer.py \
            --run-type fit -r "${timestamp}" \
            -o "${reportprefix}/${timestamp}_fit_${dataset}" \
            -s ${datasetprefix}${dataset}_train_gold.csv -g id_1 -g id_2 \
            -t id -t first_name -t middle_name -t last_name -t street_address -t city -t state -t zip_code -t full_phone_num \
            -b "0 last_name term_id" \
            -b "0 street_address tokens" \
            -b "0 zip_code term_id" \
            -b "1 first_name term_id" \
            -b "1 middle_name term_id" \
            -b "2 full_phone_num term_id" \
            $EVAL_FLAGS \
            -m ${indexer} \
            ${datasetprefix}${dataset}_index.csv \
            ${datasetprefix}${dataset}_train_query.csv \
            ${datasetprefix}${dataset}_train.csv
    else
        # Fit model once!
        mprof run analyzer.py \
            --run-type fit -r "${timestamp}" \
            -o "${reportprefix}/${timestamp}_fit_${dataset}" \
            -s ${datasetprefix}${dataset}_train_gold.csv -g id_1 -g id_2 \
            -t id -t first_name -t middle_name -t last_name -t street_address -t city -t state -t zip_code -t full_phone_num \
            $EVAL_FLAGS \
        -m ${indexer} \
            ${datasetprefix}${dataset}_index.csv \
            ${datasetprefix}${dataset}_train_query.csv \
            ${datasetprefix}${dataset}_train.csv
    fi

    result_output="${reportprefix}/${timestamp}_${indexer}_${dataset}"
    # Build and Query without metrics to get precice memory usage
    mprof run analyzer.py \
        -i id -i first_name -i middle_name -i last_name -i city -i state -i zip_code -i street_address -i full_phone_num \
        -q id -q first_name -q middle_name -q last_name -q city -q state -q zip_code -q street_address -q full_phone_num \
        -t id -t first_name -t middle_name -t last_name -t city -t state -t zip_code -t street_address -t full_phone_num \
        -s ${datasetprefix}${dataset}_train_gold.csv -g id_1 -g id_2 \
        --run-type build -r "${timestamp}" -m ${indexer} \
        $EVAL_FLAGS \
        ${datasetprefix}${dataset}_index.csv \
        ${datasetprefix}${dataset}_train_query.csv \
        ${datasetprefix}${dataset}_train.csv

    # Calculate metrics on dataset
    python -W ignore analyzer.py \
        -i id -i first_name -i middle_name -i last_name -i city -i state -i zip_code -i street_address -i full_phone_num \
        -q id -q first_name -q middle_name -q last_name -q city -q state -q zip_code -q street_address -q full_phone_num \
        -t id -t first_name -t middle_name -t last_name -t city -t state -t zip_code -t street_address -t full_phone_num \
        -s ${datasetprefix}${dataset}_train_gold.csv -g id_1 -g id_2 \
        --run-type evaluation -o $result_output \
        -m ${indexer} \
        $EVAL_FLAGS \
        ${datasetprefix}${dataset}_index.csv \
        ${datasetprefix}${dataset}_train_query.csv \
        ${datasetprefix}${dataset}_train.csv

done
fi

mv mprofile* $reportprefix
