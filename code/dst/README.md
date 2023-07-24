# DST Baseline

This repository is based on the source code of [Leveraging Slot Descriptions for Zero-Shot Cross-Domain Dialogue State Tracking](https://github.com/facebookresearch/Zero-Shot-DST).

## Environment

To create the conda environment: 

```bash
>> conda env create -f environment.yml
```

To activate the conda environment: 

```bash
>> conda activate dst
```

## Preprocessing

To preprocess the original Multi3WOZ data and create the DST dataset compatible with the T5DST model:

```bash
>> bash create_data.sh
```

The preprocessed DST data for each language will be created under ./data/

## Training

In the following, we provide example scripts to run the Arabic experiments. To run experiments in other languages, please modify the training script accordingly.

To train the DST model:

For training the DST model, run the following script:

```bash
>> bash train_dst.sh
```
You can customize the scripts for your need including:
  - `model_checkpoint`: use different LLMs (we report the performance using [mt5-small](https://huggingface.co/google/mt5-small/tree/main)).
  - `data_dir`: define the source language.
  - `few_shots_percent`: specify the percent of training data.
- Use different hyperparameters such as `train_batch_size` and `n_epochs`.

