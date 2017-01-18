#!/bin/bash
timestamp=$(date +%s)

# Run ferbl datasets
#for indexer in "DySimII" "DyLSH" "DyKMeans"
for indexer in "DySimII"
do
##for dataset in "dataset1" "ferbl-4k-1k-1" "ferbl-9k-1k-1" "ferbl-90k-10k-1"
for dataset in "ferbl-90k-10k-1"
do
    prefix="../../master_thesis/datasets/febrl/"
    result_output="${timestamp}_${indexer}_${dataset}"
    # Measure insert/query times on restaurant dataset
    # Calculate metrics on dataset
    python measures.py \
        -i rec_id -i given_name -i suburb -i surname \
        -q rec_id -q given_name -q suburb -q surname \
        -e soundex -e soundex -e soundex \
        -c default -c default -c default \
        -s ${prefix}${dataset}_train_gold.csv \
        -g id_1 -g id_2 \
        -t evaluation -o $result_output \
        -m ${indexer} \
        ${prefix}${dataset}_index.csv ${prefix}${dataset}_train_query.csv

    # Measure peak memory and index build time
    PEAK=$(./memusg.sh python measures.py \
        -i rec_id -i given_name -i suburb -i surname \
        -q rec_id -q given_name -q suburb -q surname \
        -e soundex -e soundex -e soundex \
        -c default -c default -c default \
        -t index -o $result_output \
        -m ${indexer} \
        ${prefix}${dataset}_index.csv ${prefix}${dataset}_train_query.csv)

    sed -i '$ d' $result_output
    echo "    ,\"memory_usage\":" $PEAK >> $result_output
    echo "}" >> $result_output
done

## Measure insert/query times on restaurant dataset
## Calculate metrics on restaurant dataset
#python measures.py \
    #-i id -i first_name -i middle_name -i last_name -i street_address \
    #-q id -q first_name -q middle_name -q last_name -q street_address \
    #-e soundex -e soundex -e soundex -e soundex \
    #-c default -c default -c default -c default \
    #-s ../../master_thesis/datasets/ncvoter/ncvoter_train_gold.csv \
    #-g id_1 -g id_2 \
    #-t evaluation -o "${timestamp}_${indexer}_ncvoter" \
    #-m ${indexer} \
    #../../master_thesis/datasets/ncvoter/ncvoter_index.csv \
    #../../master_thesis/datasets/ncvoter/ncvoter_train_query.csv

#PEAK=$(./memusg.sh python measures.py \
    #-i id -i first_name -i middle_name -i last_name -i street_address \
    #-q id -q first_name -q middle_name -q last_name -q street_address \
    #-e soundex -e soundex -e soundex -e soundex \
    #-c default -c default -c default -c default \
    #-t index -o "${timestamp}_${indexer}_ncvoter" \
    #-m ${indexer} \
    #../../master_thesis/datasets/ncvoter/ncvoter_index.csv \
    #../../master_thesis/datasets/ncvoter/ncvoter_train_query.csv)

## Measure insert/query times on restaurant dataset
## Calculate metrics on restaurant dataset
#python measures.py \
    #-i id -i name -i addr -i city \
    #-q id -q name -q addr -q city \
    #-e first3 -e soundex -e soundex \
    #-c default -c default -c default \
    #-s restaurant_gold.csv \
    #-g id_1 -g id_2 \
    #-t evaluation -o "${timestamp}_${indexer}_restaurant" \
    #-m ${indexer} \
    #restaurant.csv restaurant.csv

#PEAK=$(./memusg.sh python measures.py \
    #-i id -i name -i addr -i city \
    #-q id -q name -q addr -q city \
    #-e first3 -e soundex -e soundex \
    #-c default -c default -c default \
    #-t index -o "${timestamp}_${indexer}_restaurant" \
    #-m ${indexer} \
    #restaurant.csv restaurant.csv)

#sed -i '$ d' "${timestamp}_${indexer}_restaurant"
#echo "    ,\"memory_usage\":" $PEAK >> "${timestamp}_${indexer}_restaurant"
#echo "}" >> "${timestamp}_${indexer}_restaurant"

    ## Measure insert/query times on DBLP/ACM dataset
    ## Calculate metrics on DBLP/ACM dataset
    #python measures.py \
        #-i id -i title -i authors \
        #-q id -q title -q authors \
        #-e first3 -e first3 \
        #-c default -c default \
        #-s DBLP-ACM_perfectMapping.csv \
        #-g idDBLP -g idACM \
        #-t evaluation -o "${timestamp}_${indexer}_acmdblp2" \
        #-m ${indexer} \
        #ACM.csv DBLP2.csv

    #./memusg.sh python measures.py \
        #-i id -i title -i authors \
        #-q id -q title -q authors \
        #-e first3 -e first3 \
        #-c default -c default \
        #-t index -o "${timestamp}_${indexer}_acmdblp2" \
        #-m ${indexer} \
        #ACM.csv DBLP2.csv

    #sed -i '$ d' "${timestamp}_${indexer}_acmdblp2"
    #echo "    ,\"memory_usage\":" $PEAK >> "${timestamp}_${indexer}_acmdblp2"
    #echo "}" >> "${timestamp}_${indexer}_acmdblp2"
done

# Draw results
#python measures.py \
    #-t plot -r "${timestamp}" \
    #benchmark.sh benchmark.sh
