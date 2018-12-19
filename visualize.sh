#!/bin/bash


#for f in new_parse/msr_source_0/*.xml;
#do
#	python3 ../ucca/scripts/visualize.py "$f" -o "./img/msr_source_0";
#done;

#for f in new_parse/msr_reference_0/*.xml;
#do
#	python3 ../ucca/scripts/visualize.py "$f" -o "./img/msr_reference_0";
#done;

for f in new_parse/MSRPA_source/*.xml;
do
	python3 ../ucca/scripts/visualize.py "$f" -o "./img/MSRPA_source";
done;

for f in new_parse/MSRPA_reference/*.xml;
do
	python3 ../ucca/scripts/visualize.py "$f" -o "./img/MSRPA_reference";
done;

#for f in parse/p4p_source/*.xml;
#do
#	python3 ../ucca/scripts/visualize.py "$f" -o "./img/p4p_source";
#done;

#for f in parse/p4p_reference/*.xml;
#do
#	python3 ../ucca/scripts/visualize.py "$f" -o "./img/p4p_reference";
#done;