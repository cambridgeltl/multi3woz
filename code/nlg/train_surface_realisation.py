import shutil
from my_dataset import MultilingualMultiWoZDataset
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, \
    DataCollatorForSeq2Seq, Seq2SeqTrainer, Seq2SeqTrainingArguments, EarlyStoppingCallback

import configparser
import argparse
import json
import os

import evaluate
import numpy as np
from transformers import set_seed

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

    current_languages = list(map(lambda x: x.lower(), json.loads(config["experiment"]["test_languages"])))

    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        truncation_side="left")

    dataset = MultilingualMultiWoZDataset(config)
    data_dic = dataset.load_data()

    prefix = "generation dialogue utterance in "

    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
    num_added_tokens = tokenizer.add_tokens(dataset.special_token_list, special_tokens=True)
    model.resize_token_embeddings(len(tokenizer))

    def preprocess_function(examples):
        inputs = [prefix + lang.split("_")[0] + ":" + example for (example, lang) in zip(examples["source"], examples["dail_id"])]
        targets = [example for example in examples["target"]]

        model_inputs = tokenizer(inputs, text_target=targets, max_length=512, truncation=True)
        return model_inputs

    tokenized_dataset = data_dic.map(preprocess_function, batched=True)

    data_collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, model=model)


    bleu = evaluate.load("sacrebleu")
    rouge = evaluate.load("rouge")
    meteor = evaluate.load("meteor")

    def postprocess_text(preds, labels):
        preds = [pred.strip() for pred in preds]
        labels = [[label.strip()] for label in labels]

        return preds, labels

    def compute_metrics(eval_preds):
        preds, labels = eval_preds
        if isinstance(preds, tuple):
            preds = preds[0]
        decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)

        labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
        decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

        decoded_preds, decoded_labels = postprocess_text(decoded_preds, decoded_labels)

        result = bleu.compute(predictions=decoded_preds, references=decoded_labels)
        result = {"bleu": result["score"]}

        rouge_result = rouge.compute(predictions=decoded_preds, references=decoded_labels)
        meteor_result = meteor.compute(predictions=decoded_preds, references=decoded_labels)

        for key, score in rouge_result.items():
            result[key] = score
        result["meteor"] = meteor_result["meteor"]


        prediction_lens = [np.count_nonzero(pred != tokenizer.pad_token_id) for pred in preds]
        result["gen_len"] = np.mean(prediction_lens)
        result = {k: round(v, 4) for k, v in result.items()}
        return result


    training_args = Seq2SeqTrainingArguments(
        output_dir=config["experiment"]["output_dir"],
        learning_rate=float(config["experiment"]["learning_rate"]),
        per_device_train_batch_size=int(config["experiment"]["batch_size"]),
        per_device_eval_batch_size=int(config["experiment"]["batch_size"]),
        weight_decay=float(config["experiment"]["weight_decay"]),
        save_total_limit=int(config["experiment"]["save_total_limit"]),
        predict_with_generate=True,
        max_steps=int(config["experiment"]["max_training_steps"]),
        save_steps=int(config["experiment"]["eval_and_save_steps"]),
        eval_steps=int(config["experiment"]["eval_and_save_steps"]),
        save_strategy="steps",
        evaluation_strategy="steps",
        load_best_model_at_end=True,
        push_to_hub=False,
        fp16=config["experiment"]["fp16"].lower()=="true",
        metric_for_best_model="bleu",
        greater_is_better=True,
    )

    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset["train"],
        eval_dataset=tokenized_dataset["val"],
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=int(config["experiment"]["early_stopping_patience"]))]
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