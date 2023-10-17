# Copyright (c) Facebook, Inc. and its affiliates

import os, random
import torch
import argparse
import pytorch_lightning as pl
from pytorch_lightning import Trainer, seed_everything
from transformers import (AdamW, T5Tokenizer, BartTokenizer, BartForConditionalGeneration, T5ForConditionalGeneration,
                          WEIGHTS_NAME,CONFIG_NAME,
                          MT5ForConditionalGeneration)
from data_loader import prepare_data
from config import get_args
from evaluate import evaluate_metrics
import json
from tqdm import tqdm
from copy import deepcopy
import numpy as np
from collections import Counter

# def consistency_cross_entropy(lm_logits1, lm_logits2, threshold=0.4):
#     logsoftmax = torch.nn.LogSoftmax(dim=1)
#     softmax = torch.nn.Softmax(dim=1)

#     lm_logits1 = lm_logits1.squeeze()
#     lm_logits2 = lm_logits2.squeeze()
#     # (batch, vocab_size)
#     # give threshold
#     prob2 = softmax(lm_logits2)
#     # the result tuple of two output tensors (max, max_indices)
#     # print(torch.max(prob2, dim=1))
#     prob2_max, prob2_index = torch.max(prob2, dim=1)
#     valid = []
#     for i in range(prob2_max.shape[0]):
#         if (prob2_index[i]==5839 and prob2_max[i]>0.9) or (prob2_index[i]!=5839 and prob2_max[i]>threshold):
#             valid.append(1)
#         else:
#             valid.append(0)

#     #sharpening
#     soft_targets = softmax(lm_logits2/0.5)

#     loss_temp = torch.sum(- soft_targets * logsoftmax(lm_logits1), 1)
#     for i in range(prob2_max.shape[0]):
#         if valid[i]==0:
#             loss_temp[i]=0

#     return torch.mean(loss_temp)



class DST_Seq2Seq(pl.LightningModule):

    def __init__(self,args, tokenizer, model):
        super().__init__()
        self.tokenizer = tokenizer
        self.model = model
        self.lr = args["lr"]


    def training_step(self, batch, batch_idx):
        self.model.train()
        outputs = self.model(input_ids=batch["encoder_input"],
                            attention_mask=batch["attention_mask"],
                            labels=batch["decoder_output"]
                            )
        loss = outputs[0]
        # result = pl.TrainResult(loss)
        # result.log('train_loss', loss, on_epoch=True)
        return {'loss': loss, 'log': {'train_loss': loss}}
        # return result

    def validation_step(self, batch, batch_idx):
        self.model.eval()
        outputs = self.model(input_ids=batch["encoder_input"],
                            attention_mask=batch["attention_mask"],
                            labels=batch["decoder_output"]
                            )
        loss = outputs[0]

        return {'val_loss': loss, 'log': {'val_loss': loss}}
        # return result

    def validation_epoch_end(self, outputs):
        print(outputs)
        val_loss_mean = sum([o['val_loss'] for o in outputs]) / len(outputs)
        # show val_loss in progress bar but only log val_loss
        results = {'progress_bar': {'val_loss': val_loss_mean.item()}, 'log': {'val_loss': val_loss_mean.item()},
                   'val_loss': val_loss_mean.item()}
        self.log('val_loss', results['val_loss'])
        return results

    def configure_optimizers(self):
        return AdamW(self.parameters(), lr=self.lr, correct_bias=True)



def train(args, *more):
    args = vars(args)
    args["model_name"] = args["model_checkpoint"]+args["model_name"]+"_except_domain_"+args["except_domain"]+ "_slotlang_" +str(args["slot_lang"]) + "_lr_" +str(args["lr"]) + "_epoch_" + str(args["n_epochs"]) + "_seed_" + str(args["seed"])
    # train!
    seed_everything(args["seed"])

    print('0')

    model = MT5ForConditionalGeneration.from_pretrained(args["model_checkpoint"])
    print('1')
    tokenizer = T5Tokenizer.from_pretrained(args["model_checkpoint"], bos_token="[bos]", eos_token="[eos]",
                                            sep_token="[sep]")
    print('2')
    model.resize_token_embeddings(new_num_tokens=len(tokenizer))
    print('3')
    task = DST_Seq2Seq(args, tokenizer, model)

    train_loader, val_loader, test_loader, ALL_SLOTS, fewshot_loader_dev, fewshot_loader_test = prepare_data(args, task.tokenizer)

    #save model path
    save_path = os.path.join(args["saving_dir"],args["model_name"])
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    trainer = Trainer(
                    default_root_dir=save_path,
                    accumulate_grad_batches=args["gradient_accumulation_steps"],
                    gradient_clip_val=args["max_norm"],
                    max_epochs=args["n_epochs"],
                    callbacks=[pl.callbacks.EarlyStopping(monitor='val_loss',min_delta=0.00, patience=2,verbose=False, mode='min')],
                    gpus=args["GPU"],
                    deterministic=True,
                    num_nodes=1,
                    #precision=16,
                    # accelerator="ddp"
                    )

    trainer.fit(task, train_loader, val_loader)

    task.model.save_pretrained(save_path)
    task.tokenizer.save_pretrained(save_path)

    print("test start...")
    #evaluate model
    _ = evaluate_model(args, task.tokenizer, task.model, test_loader, save_path, ALL_SLOTS)

def evaluate_model(args, tokenizer, model, test_loader, save_path, ALL_SLOTS, prefix="zeroshot"):
    save_path = os.path.join(save_path,"results")
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    predictions = {}
    # to gpu
    # gpu = args["GPU"][0]
    device = torch.device("cuda:0")
    model.to(device)
    model.eval()

    slot_logger = {slot_name:[0,0,0] for slot_name in ALL_SLOTS}

    for batch in tqdm(test_loader):
        dst_outputs = model.generate(input_ids=batch["encoder_input"].to(device),
                                attention_mask=batch["attention_mask"].to(device),
                                eos_token_id=tokenizer.eos_token_id,
                                max_length=200,
                                )

        value_batch = tokenizer.batch_decode(dst_outputs, skip_special_tokens=True)

        for idx, value in enumerate(value_batch):
            dial_id = batch["ID"][idx]
            if dial_id not in predictions:
                predictions[dial_id] = {}
                predictions[dial_id]["domain"] = batch["domains"][idx][0]
                predictions[dial_id]["turns"] = {}
            if batch["turn_id"][idx] not in predictions[dial_id]["turns"]:
                predictions[dial_id]["turns"][batch["turn_id"][idx]] = {"turn_belief":batch["turn_belief"][idx], "pred_belief":[]}

            if value!="none":
                predictions[dial_id]["turns"][batch["turn_id"][idx]]["pred_belief"].append(str(batch["slot_text"][idx])+'-'+str(value))

            # analyze slot acc:
            if str(value)==str(batch["value_text"][idx]):
                slot_logger[str(batch["slot_text"][idx])][1]+=1 # hit
            slot_logger[str(batch["slot_text"][idx])][0]+=1 # total

    for slot_log in slot_logger.values():
        slot_log[2] = slot_log[1]/slot_log[0]

    with open(os.path.join(save_path, f"{prefix}_slot_acc.json"), 'w') as f:
        json.dump(slot_logger,f, indent=4)

    with open(os.path.join(save_path, f"{prefix}_prediction.json"), 'w') as f:
        json.dump(predictions,f, indent=4)

    joint_acc_score, F1_score, turn_acc_score = evaluate_metrics(predictions, ALL_SLOTS)

    evaluation_metrics = {"Joint Acc":joint_acc_score, "Turn Acc":turn_acc_score, "Joint F1":F1_score}
    print(f"{prefix} result:",evaluation_metrics)

    with open(os.path.join(save_path, f"{prefix}_result.json"), 'w') as f:
        json.dump(evaluation_metrics,f, indent=4)

    return predictions



if __name__ == "__main__":
    args = get_args()
    print(args)
    if args.mode=="train":
        train(args)