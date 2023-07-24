import shutil
import torch
from my_dataset import MultilingualMultiWoZDataset
from transformers import AutoTokenizer
import configparser
import argparse
import json
import os
from transformers import DataCollatorForTokenClassification
import evaluate
import numpy as np
from transformers import set_seed
from tqdm import tqdm

from transformers import AutoModelForTokenClassification, TrainingArguments, Trainer

result_dic = {}
prediction_dic = {}

def run_experiment():
    global result_dic
    global prediction_dic

    parser = argparse.ArgumentParser(description="Config Loader")
    parser.add_argument("-C","-c", "--config", help="set config file", required=True, type=argparse.FileType('r'))
    parser.add_argument("-s", "--seed", help="set random seed", type=int)
    parser.add_argument("--do_train", action='store_true')
    parser.add_argument("--do_test", action='store_true')
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

    if config["project"]["overwrite_eval_result"].lower() != "true":
        if os.path.isfile(result_save_path) and os.access(result_save_path, os.R_OK):
            with open(result_save_path, "r", encoding="utf-8") as f:
                result_dic = json.load(f)

    if args.do_train:
        train(config)
    if args.do_test:
        test(config)

    with open(result_save_path, 'w', encoding='utf-8') as f:
        json.dump(result_dic, f, ensure_ascii=False, indent=4)

    prediction_save_path = os.path.join(config["experiment"]["output_dir"], "predictions.json")
    with open(prediction_save_path, 'w', encoding='utf-8') as f:
        json.dump(prediction_dic, f, ensure_ascii=False, indent=4)

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

    data_collator = DataCollatorForTokenClassification(tokenizer=tokenizer)

    model = AutoModelForTokenClassification.from_pretrained(
        model_name, num_labels=len(id2label), id2label=id2label, label2id=label2id
    )

    tokenizer.add_tokens(dataset.special_token_list, special_tokens=True)
    model.resize_token_embeddings(len(tokenizer))

    seqeval = evaluate.load("seqeval")

    def compute_metrics(p):

        labels = p.label_ids
        predictions = p.predictions
        predictions = np.argmax(predictions, axis=2)

        true_predictions = [
            [id2label[p] for (p, l) in zip(prediction, label) if l != -100]
            for prediction, label in zip(predictions, labels)
        ]
        true_labels = [
            [id2label[l] for (p, l) in zip(prediction, label) if l != -100]
            for prediction, label in zip(predictions, labels)
        ]

        results = seqeval.compute(predictions=true_predictions, references=true_labels)

        return {
            "precision": results["overall_precision"],
            "recall": results["overall_recall"],
            "f1": results["overall_f1"],
            "accuracy": results["overall_accuracy"],
        }

    training_args = TrainingArguments(
        output_dir=config["experiment"]["output_dir"],
        learning_rate=float(config["experiment"]["learning_rate"]),
        per_device_train_batch_size=int(config["experiment"]["batch_size"]),
        per_device_eval_batch_size=int(config["experiment"]["batch_size"]),
        num_train_epochs=int(config["experiment"]["training_epoch"]),
        weight_decay=float(config["experiment"]["weight_decay"]),
        evaluation_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        push_to_hub=False,
        save_total_limit=int(config["experiment"]["save_total_limit"]),
        metric_for_best_model="f1",
        greater_is_better=True,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=data_dic["train"],
        eval_dataset=data_dic["val"],
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
    )

    trainer.train()
    dev_result = trainer.evaluate()

    result_dic["dev_result_token"] = dev_result
    print(dev_result)


    test_result = (trainer.evaluate(data_dic["test"]))
    print(test_result)
    result_dic["test_result_token"] = test_result

    trainer.save_model(os.path.join(config["experiment"]["output_dir"], "checkpoint-best"))


def test(config):
    global result_dic
    global prediction_dic

    model_path = os.path.join(config["experiment"]["output_dir"], "checkpoint-best")

    dataset = MultilingualMultiWoZDataset(config)
    data_dic = dataset.load_data()

    label2id = dataset.label_to_index
    id2label = dataset.index_to_label

    seqeval = evaluate.load("seqeval")
    test_data = data_dic["test"]

    model = AutoModelForTokenClassification.from_pretrained(model_path).to("cuda")

    assert label2id == model.config.label2id

    true_predictions = []
    true_labels = []

    for data_entry in tqdm(test_data):

        input_ids = torch.tensor(data_entry["input_ids"]).unsqueeze(0).to("cuda")
        attention_mask = torch.tensor(data_entry["attention_mask"]).unsqueeze(0).to("cuda")

        with torch.no_grad():
            logits = model(input_ids=input_ids,attention_mask=attention_mask).logits

        predictions = torch.argmax(logits, dim=2)
        prediction = (predictions[0])
        prediction= prediction.cpu().detach().numpy()

        mapped_result = dataset.map_token_bio_to_char_bio(data_entry, prediction)
        gold_result =  data_entry["char_bio_tag"]

        prediction_entry = {}
        prediction_entry["gold_bio_tag"] = gold_result
        prediction_entry["pred_bio_tag"] = mapped_result
        prediction_entry["text"] = data_entry["text"]
        prediction_entry["turn_id"] = data_entry["turn_id"]
        prediction_entry["dail_id"] = data_entry["dail_id"]
        if data_entry["dail_id"] not in prediction_dic:
            prediction_dic[data_entry["dail_id"]] = {}
        prediction_dic[data_entry["dail_id"]][data_entry["turn_id"]] = prediction_entry

        true_predictions.append(gold_result)
        true_labels.append(mapped_result)

    results = seqeval.compute(predictions=true_predictions, references=true_labels)

    char_level_test_result  = {}
    char_level_test_result["precision"] = results["overall_precision"]
    char_level_test_result["recall"] = results["overall_recall"]
    char_level_test_result["f1"] = results["overall_f1"]
    char_level_test_result["accuracy"] = results["overall_accuracy"]
    print(char_level_test_result)
    result_dic["test_result_char"] = char_level_test_result


def main():
    run_experiment()


if __name__ == '__main__':
    main()
