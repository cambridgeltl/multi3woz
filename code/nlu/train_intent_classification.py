import shutil
import sys

import torch
from torch import Tensor

from my_dataset import MultilingualMultiWoZDataset
from transformers import AutoModelForSequenceClassification, AutoTokenizer, EarlyStoppingCallback
from transformers import DataCollatorWithPadding
from transformers import TrainingArguments, Trainer
import evaluate
from argparse import ArgumentParser
import numpy as np
import configparser
import argparse
import json
import os

import numpy as np
from transformers import set_seed
from tqdm import tqdm

from sklearn.metrics import f1_score, roc_auc_score, accuracy_score, precision_score, recall_score
from transformers import EvalPrediction
import torch

from transformers import AutoModelForTokenClassification, TrainingArguments, Trainer

result_dic = {}

def run_experiment():
    global result_dic
    global prediction_dic

    parser = argparse.ArgumentParser(description="Config Loader")
    parser.add_argument("-C","-c", "--config", help="set config file", required=True, type=argparse.FileType('r'))
    parser.add_argument("-s", "--seed", help="set random seed", type=int)
    parser.add_argument("--do_train", action='store_true')
    args = parser.parse_args()

    config = None

    config_file_path = args.config.name
    if config_file_path is not None:
        try:
            config = configparser.ConfigParser(allow_no_value=True)
            config.read(config_file_path)
        except Exception as inst:
            print('Failed to parse file', inst)
    else:
        config = configparser.ConfigParser(allow_no_value=True)

    config.set("project", "config_path", args.config.name)

    result_save_path = os.path.join(config["experiment"]["output_dir"], "evaluation_result.json")

    if args.do_train:
        train(config)

    with open(result_save_path, 'w', encoding='utf-8') as f:
        json.dump(result_dic, f, ensure_ascii=False, indent=4)


    config_save_path = os.path.join(config["experiment"]["output_dir"], "config.cfg")
    shutil.copyfile(config["project"]["config_path"], config_save_path)


def train(config):
    global result_dic

    model_name = config["experiment"]["model_name"]
    set_seed(int(config["experiment"]["seed"]))

    tokenizer = AutoTokenizer.from_pretrained(model_name)


    dataset = MultilingualMultiWoZDataset(config)
    data_dic = dataset.load_data()
    label2id = dataset.label_to_index
    id2label = dataset.index_to_label

    model = AutoModelForSequenceClassification.from_pretrained(
        model_name, num_labels=len(id2label), id2label=id2label, label2id=label2id,
        problem_type="multi_label_classification",
        ignore_mismatched_sizes=True
    )

    tokenizer.add_tokens(dataset.special_token_list, special_tokens=True)
    model.resize_token_embeddings(len(tokenizer))

    assert len(label2id) ==    len(id2label)
    num_of_label = len(label2id)

    def preprocess_function(example):
        text = example["text"]
        one_hot_labels = [0.0 for _ in range(num_of_label)]
        encoding = tokenizer(text, truncation=True)

        for idx in example["intent_idx"]:
            one_hot_labels[idx] = 1.0

        encoding["labels"] = one_hot_labels
        return encoding

    tokenized_dataset = data_dic.map(preprocess_function, batched=False)

    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    # source: https://jesusleal.io/2021/04/21/Longformer-multilabel-classification/
    def multi_label_metrics(predictions, labels, threshold=0.5):
        sigmoid = torch.nn.Sigmoid()
        probs = sigmoid(torch.Tensor(predictions))
        y_pred = np.zeros(probs.shape)
        y_pred[np.where(probs >= threshold)] = 1
        y_true = labels

        f1_micro_average = f1_score(y_true=y_true, y_pred=y_pred, average='micro')
        roc_auc = roc_auc_score(y_true, y_pred, average='micro')
        precision = precision_score(y_true, y_pred, average='micro')
        recall = recall_score(y_true, y_pred, average='micro')
        accuracy = accuracy_score(y_true, y_pred)
        metrics = {'f1': f1_micro_average,
                   'roc_auc': roc_auc,
                   'accuracy': accuracy,
                   "precision": precision,
                   "recall" : recall,
                   }
        return metrics

    def compute_metrics(p: EvalPrediction):
        preds = p.predictions[0] if isinstance(p.predictions,
                                               tuple) else p.predictions
        result = multi_label_metrics(
            predictions=preds,
            labels=p.label_ids)
        return result

    training_args = TrainingArguments(
        output_dir=config["experiment"]["output_dir"],
        learning_rate=float(config["experiment"]["learning_rate"]),
        per_device_train_batch_size=int(config["experiment"]["batch_size"]),
        per_device_eval_batch_size=int(config["experiment"]["batch_size"]),
        max_steps = int(config["experiment"]["max_training_steps"]),
        save_steps=int(config["experiment"]["eval_and_save_steps"]),
        eval_steps=int(config["experiment"]["eval_and_save_steps"]),
        save_strategy="steps",
        evaluation_strategy="steps",
        weight_decay=float(config["experiment"]["weight_decay"]),
        load_best_model_at_end=True,
        push_to_hub=False,
        save_total_limit=int(config["experiment"]["save_total_limit"]),
        metric_for_best_model="f1",
        greater_is_better=True,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset["train"],
        eval_dataset=tokenized_dataset["val"],
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
        callbacks = [EarlyStoppingCallback(early_stopping_patience=int(config["experiment"]["early_stopping_patience"]))]

    )

    trainer.train()

    dev_result = trainer.evaluate()
    result_dic["dev_result"] = dev_result
    print(dev_result)

    test_result = trainer.evaluate(tokenized_dataset["test"])
    result_dic["test_result"] = test_result
    print(test_result)

    trainer.save_model(os.path.join(config["experiment"]["output_dir"], "checkpoint-best"))

def main():
    run_experiment()

if __name__ == '__main__':
    main()