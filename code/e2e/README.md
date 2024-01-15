# E2E Baseline

🚨 We have released a new and arguably better end-to-end baseline system. It is easier to use than the code here. Please check our new [codebase](https://github.com/cambridgeltl/multi3woz/tree/main).

This repository provides end-to-end ToD baselines on Multi3WOZ using seq2seq (i.e., encoder-decoder) models.

## Environment

We need two separate environments to train and evaluate our systems: *end2end* and *soloist*.

The *end2end* environment is used for system training.

To create the conda environment: 

```bash
>> conda env create -f environment.yml
```

To activate the conda environment: 

```bash
>> conda activate end2end
```

The *soloist* environment is used for data preprocessing and system evaluation.


To create the conda environment: 

```bash
>> conda create -n soloist Python=3.6
>> conda activate soloist
>> cd soloist
>> pip install -r requirements.txt
```

To activate the conda environment: 

```bash
>> conda activate soloist
```

## Experiment

In the following, we provide example scripts to run the Arabic experiments. To run experiments in other languages, please use the data preprocessing script in the corresponding directory (./soloist/English) and modify the training script accordingly.

### Preprocessing

Run the preprocessing script, i.e.:

```bash
>> conda activate soloist
>> cd soloist/Arabic
>> bash fetch_data_and_preprocessing.sh
```

For belief state prediction, please run the following script under ./code/e2e/:

```bash
>> python src/belief_state_prediction/linearize.py
```

For response prediction, please run the following script under ./code/e2e/:

```bash
>> python src/response_prediction/linearize.py
```

### Training

To reproduce our experimental results:

```bash
>> conda activate soloist
>> bash scripts/all.sh
```

For belief state prediction:

```bash 
>> bash scripts/training_bsp.sh
```
For response prediction:
```bash 
>> bash scripts/training_rp.sh
```

You can customize the scripts for your need including:
- Use different LLMs (we report the performance using [mt5-small](https://huggingface.co/google/mt5-small/tree/main)) by specifying `model_name_or_path`
- Use your own dataset by specifying `train_file` and `validation_file`
  - All files should contain exactly two columns for the seq2seq setup.
  - For belief state prediction, specify `history_column` (default: `history`) to match the dialogue history and `bs_column` (default: `bs`) to match the belief state.
  - For response prediction, specify `context_column` (default: `context`) to match the entire context (dialogue history, belief state, and database state) and `response_column` (default: `response`) o match the belief state.
- Use different hyperparameters such as `learning_rate`.

### Evaluation

Inference and postprocessing:

```bash 
>> conda activate end2end
>> bash scripts/postprocessing.sh
```

Evaluation using the soloist evaluation script:

```bash 
>> conda activate soloist
>> cd soloist/Arabic
>> bash eval.sh
```