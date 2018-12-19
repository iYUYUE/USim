#!/bin/bash

for f in demo/*.xml
do
	python3 ../ucca/scripts/visualize.py "$f" -o "./img/demo";
done;

#for f in new_parse/MSRPA_reference/*.xml;
#do
#	python3 ../ucca/scripts/visualize.py "$f" -o "./img/new_parse/MSRPA_reference";
#done;