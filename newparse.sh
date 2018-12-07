#!/bin/bash

for f in new_parse/MSRPA_source/*.xml
do
	python3 ../ucca/scripts/visualize.py "$f" -o "./img/new_parse/MSRPA_source";
done;

for f in new_parse/MSRPA_reference/*.xml;
do
	python3 ../ucca/scripts/visualize.py "$f" -o "./img/new_parse/MSRPA_reference";
done;