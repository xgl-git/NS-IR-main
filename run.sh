#!/bin/bash

for s in "fiqa" "nfcorpus" "scifact" "trec-covid" "arguana" "dbpedia-entity"
do
    python main.py --dataset $s
done
