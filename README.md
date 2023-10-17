# Multi3WOZ

**Please note that this branch may be behind the main branch.**

This repository reproduces the results reported in our paper, which may not incorporate changes or fixes we introduced to the dataset after the publication of this paper. If possible, please use the data and code in the main branch for your research.

Code repository for the paper: <br>

[***A Systematic Study of Performance Disparities in Multilingual Task-Oriented Dialogue Systems***](https://openreview.net/forum?id=aY4avQ0ItI)
by [Songbo Hu](https://songbohu.github.io), [Han Zhou](https://hzhou.top), [Zhangdie Yuan](https://www.moyyuan.com), [Milan Gritta](https://github.com/milangritta), [Guchun Zhang](), [Ignacio Iacobacci](https://iiacobac.wordpress.com),  [Anna Korhonen](https://sites.google.com/site/annakorhonen/), and [Ivan Vulić](https://sites.google.com/site/ivanvulic/).


## This Repository

- **data.zip** contains the Multi3WOZ dataset in four languages: Arabic (Afro-Asiatic), English (Indo-European), French (Indo-European), and Turkish (Turkic). Each language includes 9,160 multi-parallel dialogues.

- **code** directory contains the baseline code to reproduce our experimental results in the paper. We provide our baseline code for all the popular ToD tasks: natural language understanding (NLU), dialogue state tracking (DST), and natural language generation (NLG).

## Baselines

Before running the experiments, please run the following command to uncompress the data 

```bash
>> unzip data.zip
```

Then follow each baseline directory's instructions to reproduce our reported results. For example, please follow [./code/nlu/README.md](./code/nlu/README.md) to reproduce our reported NLU results.

## Annotation Protocol

Please visit the following website for our annotation instruction: [https://cambridgeltl.github.io/multi3woz/](https://cambridgeltl.github.io/multi3woz/).


## Issue Report

If you have found any issue in this repository, please contact: [sh2091@cam.ac.uk](mailto:sh2091@cam.ac.uk).
