#!/bin/bash
timestamp=$(date +%s)

# Measure insert/query times on restaurant dataset
# Calculate metrics on restaurant dataset
python measures.py \
    -i rec_id -i given_name -i suburb -i surname -i postcode \
    -q rec_id -q given_name -q suburb -q surname -q postcode \
    -e soundex -e soundex -e soundex -e first3 \
    -c default -c default -c default -c default \
    -s dataset1_gold.csv \
    -g org_id -g dup_id \
    -t evaluation -o "${timestamp}_dataset1" \
    dataset1.csv dataset1.csv

#python -m cProfile measures.py \
PEAK=$(./memusg.sh python measures.py \
    -i rec_id -i given_name -i suburb -i surname -i postcode \
    -q rec_id -q given_name -q suburb -q surname -q postcode \
    -e soundex -e soundex -e soundex -e first3 \
    -c default -c default -c default -c default \
    -t index -o $ "${timestamp}_dataset1" \
    dataset1.csv dataset1.csv)

sed -i '$ d' "${timestamp}_dataset1"
echo "    ,\"memory_usage\":" $PEAK >> "${timestamp}_dataset1"
echo "}" >> "${timestamp}_dataset1"

# Measure insert/query times on restaurant dataset
# Calculate metrics on restaurant dataset
python measures.py \
    -i rec_id -i given_name -i suburb -i surname -i postcode \
    -q rec_id -q given_name -q suburb -q surname -q postcode \
    -e soundex -e soundex -e soundex -e first3 \
    -c default -c default -c default -c default \
    -s ./ferbl-9k-1k-1_gold.csv \
    -g org_id -g dup_id \
    -t evaluation -o "${timestamp}_febrl-9k-1k-1" \
    ./ferbl-9k-1k-1.csv ./ferbl-9k-1k-1.csv

#python -m cProfile measures.py \
PEAK=$(./memusg.sh python measures.py \
    -i rec_id -i given_name -i suburb -i surname -i postcode \
    -q rec_id -q given_name -q suburb -q surname -q postcode \
    -e soundex -e soundex -e soundex -e first3 \
    -c default -c default -c default -c default \
    -t index -o $ "${timestamp}_febrl-9k-1k-1" \
    ./ferbl-9k-1k-1.csv ./ferbl-9k-1k-1.csv)

sed -i '$ d' "${timestamp}_febrl-9k-1k-1"
echo "    ,\"memory_usage\":" $PEAK >> "${timestamp}_febrl-9k-1k-1"
echo "}" >> "${timestamp}_febrl-9k-1k-1"

python measures.py \
    -t plot -r "${timestamp}" \
    benchmark.sh benchmark.sh

## Measure insert/query times on restaurant dataset
## Calculate metrics on restaurant dataset
#python measures.py \
    #-i rec_id -i given_name -i suburb -i surname -i postcode \
    #-q rec_id -q given_name -q suburb -q surname -q postcode \
    #-e soundex -e soundex -e soundex -e first3 \
    #-c default -c default -c default -c default \
    #-s ./ferbl-4k-1k-1_gold.csv \
    #-g org_id -g dup_id \
    #-t evaluation \
    #./ferbl-4k-1k-1.csv ./ferbl-4k-1k-1.csv

#./memusg.sh python measures.py \
    #-i rec_id -i given_name -i suburb -i surname -i postcode \
    #-q rec_id -q given_name -q suburb -q surname -q postcode \
    #-e soundex -e soundex -e soundex -e first3 \
    #-c default -c default -c default -c default \
    #-t index \
    #./ferbl-4k-1k-1.csv ./ferbl-4k-1k-1.csv

## Measure insert/query times on restaurant dataset
## Calculate metrics on restaurant dataset
#python measures.py \
    #-i rec_id -i given_name -i suburb -i surname -i postcode \
    #-q rec_id -q given_name -q suburb -q surname -q postcode \
    #-e soundex -e soundex -e soundex -e first3 \
    #-c default -c default -c default -c default \
    #-s ./ferbl-9k-1k-1_gold.csv \
    #-g org_id -g dup_id \
    #-t evaluation \
    #./ferbl-9k-1k-1.csv ./ferbl-9k-1k-1.csv

#./memusg.sh python measures.py \
    #-i rec_id -i given_name -i suburb -i surname -i postcode \
    #-q rec_id -q given_name -q suburb -q surname -q postcode \
    #-e soundex -e soundex -e soundex -e first3 \
    #-c default -c default -c default -c default \
    #-t index \
    #./ferbl-9k-1k-1.csv ./ferbl-9k-1k-1.csv

## Measure insert/query times on restaurant dataset
## Calculate metrics on restaurant dataset
#python measures.py \
    #-i rec_id -i given_name -i suburb -i surname -i postcode \
    #-q rec_id -q given_name -q suburb -q surname -q postcode \
    #-e soundex -e soundex -e soundex -e first3 \
    #-c default -c default -c default -c default \
    #-s ./ferbl-90k-10k-1_gold.csv \
    #-g org_id -g dup_id \
    #-t evaluation \
    #./ferbl-90k-10k-1.csv ./ferbl-90k-10k-1.csv

#./memusg.sh python measures.py \
    #-i rec_id -i given_name -i suburb -i surname -i postcode \
    #-q rec_id -q given_name -q suburb -q surname -q postcode \
    #-e soundex -e soundex -e soundex -e first3 \
    #-c default -c default -c default -c default \
    #-t index \
    #./ferbl-90k-10k-1.csv ./ferbl-90k-10k-1.csv

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
