#!/bin/bash

python ./ucca/scripts/visualize.py parse/p4p_source/$1.xml -o "./img/p4p_source"
python ./ucca/scripts/visualize.py parse/p4p_reference/$1.xml -o "./img/p4p_reference"
