# FZR-Enhancing-Knowledge-Transfer-via-Shared-Factors-Composition-in-Zero-Shot-Relational-Learning


Code and Data for the paper: "FZR: Enhancing Knowledge Transfer via Shared Factors Composition in Zero-Shot Relational Learning"

## Requirements
- ``python 3.7``
- ``pytorch 1.13.1``

## Dataset
### glove
You need to download pretrained [Glove](http://nlp.stanford.edu/data/glove.6B.zip) word embedding dictionary, uncompress it and put all files to the folder ``data/glove/``.

### NELL-ZS and Wiki-ZS
You can download these two datasets from [here](https://github.com/Panda0406/Zero-shot-knowledge-graph-relational-learning) and put them to the corresponding data folder.

## Preparation
- ``python factor_clustering.py``
- ``python process_score.py``

## Training
You can follow the commands at ``code/README.md``

> Note: you can skip the preparation step if you just want to use the enhanced representation we learned, the files are provided in the corresponding directories.
