# Multi3WOZ

**Please note that this branch may be behind the main branch.**

This repository reproduces the results reported in our paper, which may not incorporate changes or fixes we introduced to the dataset after the publication of this paper. If possible, please use the data and code in the main branch for your research.

Data and Code to reproduce the results for the TACL paper:

[***Multi3WOZ: A Multilingual, Multi-Domain, Multi-Parallel Dataset for Training and Evaluating Culturally Adapted Task-Oriented Dialog Systems***](https://arxiv.org/abs/2307.14031)
by [Songbo Hu](https://songbohu.github.io),&ast; [Han Zhou](https://hzhou.top),&ast; [Mete Hergul](), [Milan Gritta](https://github.com/milangritta), [Guchun Zhang](), [Ignacio Iacobacci](https://iiacobac.wordpress.com), [Ivan Vulić](https://sites.google.com/site/ivanvulic/), &ast;&ast; and [Anna Korhonen](https://sites.google.com/site/annakorhonen/). &ast;&ast;

Multi3WOZ is a novel multilingual, multi-domain, multi-parallel task-oriented dialogue (ToD) dataset. It is large-scale and offers culturally adapted dialogues in 4 languages to enable training and evaluation of multilingual and cross-lingual ToD systems. This dataset is collected via a complex bottom-up data collection process, as shown in the following figure.

<p float="middle">
  <img src="./media/figure1.png" width="800" />
</p>



## This Repository

- **data.zip** contains the Multi3WOZ dataset in four languages: Arabic (Afro-Asiatic), English (Indo-European), French (Indo-European), and Turkish (Turkic). Each language includes 9,160 multi-parallel dialogues.

- **code** directory contains the baseline code to reproduce our experimental results in the paper. We provide our baseline code for all the popular ToD tasks: natural language understanding (NLU), dialogue state tracking (DST), natural language generation (NLG), and end-to-end modelling (E2E).

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
