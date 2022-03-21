# NFC

This repository is the PyTorch implementation of our ACL-2022 [paper]

Source code for Investigating Non-local Features for Neural Constituency Parsing.

## Contents
1. [Introduction](#Introduction)
2. [Usage](#usage)
3. [Citation](#citation)
4. [Credits](#credits)

## Introduction

We inject non-local sub-tree structure features to chart-based constituency parser by introducing two auxiliary training objectives.

### Instance-level Pattern Loss
<img src="figure/intro.png" width="1000">
We define pattern as the n-gram constituents sharing the same parent, and ask model to predict the pattern based on span representation.

### Corpus-level Consistency Loss
The consistency loss regularizes the co-occurrence between constituents and pattern by collecting corpus-level statistics.



NFC is an extended version of [Self-Attentive Parser](https://arxiv.org/abs/1805.01052), whose code base is at [this url](https://github.com/nikitakit/self-attentive-parser).

## Usage

To run our code:
```bash
$ pip install -r requirement.txt
```

We release our model checkpoints at Google Drive. (Models for [PTB](https://drive.google.com/file/d/1GD-u3CIxRz20bFjhaYGOVJtyt8Patwz-/view?usp=sharing) and for [CTB5.1](https://drive.google.com/file/d/1vpGTii40PgOllAen43CzNNeWprO4fMCn/view?usp=sharing).)

For test, run
```
python src/export test \
    --pretrained-model-path "bert-large-uncased" \
    --model-path path-to-the-checkpoint\
    --test-path path-to-the-test-file
```
If for Chinese, add '''--text-processing chinese''' and change the --pretrained-model-path.




## Citation

If you use this software for research, please cite our papers as follows:

```

```

## Credits

The code in this repository and portions of this README are based on https://github.com/mitchellstern/minimal-span-parser and https://github.com/nikitakit/self-attentive-parser.
