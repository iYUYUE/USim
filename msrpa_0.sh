#!/bin/bash

for f in parse/msr_source_0/*.xml;
do
	python3 ../ucca/scripts/visualize.py "$f" -o "./img/msr_source_0";
done;

for f in parse/msr_reference_0/*.xml;
do
	python3 ../ucca/scripts/visualize.py "$f" -o "./img/msr_reference_0";
done;