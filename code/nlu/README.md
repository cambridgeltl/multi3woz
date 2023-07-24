# NLU Baseline

## Environment

To create the conda environment: 

```bash
>> conda env create -f environment.yml
```

To activate the conda environment: 

```bash
>> conda activate nlu
```

## Experiment

Before running experiments, edit the config files accordingly. For example, change the project_root_path to the path at the root of this repository.

In the following, we provide example scripts to run the Arabic experiments. To run experiments in other languages, please modify the config file accordingly.

To train and evaluate the intent detection models:

```bash
>> bash train_intent.sh
```

To train and evaluate the slot labelling models:

```bash
>> bash train_labelling.sh
```