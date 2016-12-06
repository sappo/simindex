#!/bin/bash
timestamp=$(date +%s)

# Run ferbl datasets
for dataset in "dataset1" "ferbl-4k-1k-1" "ferbl-9k-1k-1" "ferbl-90k-10k-1"
do
    # Measure insert/query times on restaurant dataset
    # Calculate metrics on dataset
    python measures.py \
        -i rec_id -i given_name -i suburb -i surname -i postcode \
        -q rec_id -q given_name -q suburb -q surname -q postcode \
        -e soundex -e soundex -e soundex -e first3 \
        -c default -c default -c default -c default \
        -s ${dataset}_gold.csv \
        -g org_id -g dup_id \
        -t evaluation -o "${timestamp}_${dataset}" \
        ${dataset}.csv ${dataset}.csv

    # Measure peak memory and index build time
    PEAK=$(./memusg.sh python measures.py \
        -i rec_id -i given_name -i suburb -i surname -i postcode \
        -q rec_id -q given_name -q suburb -q surname -q postcode \
        -e soundex -e soundex -e soundex -e first3 \
        -c default -c default -c default -c default \
        -t index -o $ "${timestamp}_${dataset}" \
        ${dataset}.csv ${dataset}.csv)

    sed -i '$ d' "${timestamp}_${dataset}"
    echo "    ,\"memory_usage\":" $PEAK >> "${timestamp}_${dataset}"
    echo "}" >> "${timestamp}_${dataset}"
done

# Draw results
python measures.py \
    -t plot -r "${timestamp}" \
    benchmark.sh benchmark.sh

## Measure insert/query times on restaurant dataset
## Calculate metrics on restaurant dataset
#python measures.py \
    #-i id -i name -i addr -i city -i phone \
    #-q id -q name -q addr -q city -q phone \
    #-e first3 -e soundex -e soundex -e first3 \
    #-c default -c default -c default -c default \
    #-s restaurant_gold.csv \
    #-g id_1 -g id_2 \
    #-t evaluation \
    #restaurant.csv restaurant.csv

#./memusg.sh python measures.py \
    #-i id -i name -i addr -i city -i phone \
    #-q id -q name -q addr -q city -q phone \
    #-e first3 -e soundex -e soundex -e first3 \
    #-c default -c default -c default -c default \
    #-t index \
    #restaurant.csv restaurant.csv

## Measure insert/query times on DBLP/ACM dataset
## Calculate metrics on DBLP/ACM dataset
#python measures.py \
    #-i id -i title -i authors \
    #-q id -q title -q authors \
    #-e first3 -e first3 \
    #-c default -c default \
    #-s DBLP-ACM_perfectMapping.csv \
    #-g idDBLP -g idACM \
    #-t evaluation \
    #ACM.csv DBLP2.csv

#./memusg.sh python measures.py \
    #-i id -i title -i authors \
    #-q id -q title -q authors \
    #-e first3 -e first3 \
    #-c default -c default \
    #-t index \
    #ACM.csv DBLP2.csv
