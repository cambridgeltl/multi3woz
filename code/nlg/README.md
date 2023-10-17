# NLU Baseline

## Environment

To create the conda environment: 

```bash
>> conda env create -f environment.yml
```

To activate the conda environment: 

```bash
>> conda activate nlg
```

## Experiment

Before running experiments, edit the config files accordingly. For example, change the project_root_path to the path at the root of this repository.

In the config file, we use the `train_languages` and `val_languages` to select different data collection strategies in Section 5 RQ4. These keys are in the format of "strategy" + "proportion" of the data. For example, random0.1 denotes we randomly select 10% of the training or validation data in Multi3WOZ. "trigram0.1" means that we select 10% of the data that maximise the trigram coverage of the chosen dialogues. Currently, we support five "strategies": "random", "domain", "slot", "length", and "trigram". We support proportions from 0.01, 0.02, 0.03 to 0.09 and 0.1, 0.2 to 1.0. To find how these subsets of dialogues are selected exactly, please check the jupyter notebook at ./data/build_parallel_dic.ipynb
In the following, we provide example scripts to run the Arabic experiments. To run experiments in other languages, please modify the config file accordingly.

To train and evaluate the natural langauge generation models:

```bash
>> bash train_nlg.sh
```