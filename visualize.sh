#!/bin/bash

# for f in parse/MSRPA_source/*.xml;
# do
# 	python3 ../ucca/scripts/visualize.py "$f" -o "./img/MSRPA_source";
# done;

for f in parse/MSRPA_reference/*.xml;
do
	python3 ../ucca/scripts/visualize.py "$f" -o "./img/MSRPA_reference1";
done;

# python3 ../ucca/scripts/visualize.py parse/p4p_source/$1.xml -o "./img/p4p_source"
# python3 ../ucca/scripts/visualize.py parse/p4p_reference/$1.xml -o "./img/p4p_reference"

for f in parse/p4p_source/*.xml;
do
	python3 ../ucca/scripts/visualize.py "$f" -o "./img/p4p_source";
done;

for f in parse/p4p_reference/*.xml;
do
	python3 ../ucca/scripts/visualize.py "$f" -o "./img/p4p_reference";
done;