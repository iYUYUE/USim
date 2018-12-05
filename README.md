# USim
monolingual sentence similarity measure

Please cite our [NAACL2018 paper](http://www.aclweb.org/anthology/N18-2020) if you use our measure or annotations.

```@inproceedings{choshen2018reference,
  title={Reference-less Measure of Faithfulness for Grammatical Error Correction},
  author={Choshen, Leshem and Abend, Omri},
  booktitle={Proceedings of the 2018 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies, Volume 2 (Short Papers)},
  volume={2},
  pages={124--129},
  year={2018}
}
```

As USim uses currently the TUPA parser it should be installed

`pip install tupa`

you should also have a trained model, pretrained ones could be acquired from [here](https://github.com/danielhers/tupa).

In addition ucca and the modules for required align submodule should also be installed

```
pip install ucca
pip install zss
pip install distance
pip install munkres
```


Usage example (assuming parser module was changed in the code, otherwise add -p flag):
python USim.py parse out.out -ss "I love rusty spoons", "nothing matters" -rs "he shares pretty cars", "nothing indeed"

== Scripts ==

python3 USim.py parse output/p4p.align.out -sf p4p_source.txt -rf p4p_reference.txt -p ../tupa/models/ucca-bilstm -a p4p_alignment.txt

python3 USim.py parse output/p4p.out -sf p4p_source.txt -rf p4p_reference.txt -p ../tupa/models/ucca-bilstm

python3 USim.py parse output/MSRPA.align.out -sf MSRPA_source.txt -rf MSRPA_reference.txt -p ../tupa/models/ucca-bilstm -a MSRPA_alignment.txt

python3 USim.py parse output/MSRPA.out -sf MSRPA_source.txt -rf MSRPA_reference.txt -p ../tupa/models/ucca-bilstm

python3 USim.py parse output/msr_0.out -sf msr_source_0.txt -rf msr_reference_0.txt -p ../tupa/models/ucca-bilstm

python3 USim.py parse output/msr_0_r.out -sf msr_reference_0.txt -rf msr_source_0.txt -p ../tupa/models/ucca-bilstm

python3 USim.py parse output/msr_1.out -sf msr_source_1.txt -rf msr_reference_1.txt -p ../tupa/models/ucca-bilstm

python3 USim.py parse output/msr_1_r.out -sf msr_reference_1.txt -rf msr_source_1.txt -p ../tupa/models/ucca-bilstm

to visualize, run "bash p4p.sh [file id]"

without alignment from paraphrase annonatation:

python3 stats.py output/p4p.out

with alignment from paraphrase annonatation:

python3 stats.py output/p4p.align.out

python3 rank.py [+/-] output/p4p.out

to get corpus wide analysis:

python USim_corpus.py parse output/p4p.corpus.out -sf p4p_source.txt -rf p4p_reference.txt -p ../tupa/models/ucca-bilstm  -a p4p_alignment.txt