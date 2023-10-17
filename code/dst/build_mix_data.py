import os
import sys
import json

lang_list = ['english', 'arabic', 'french', 'turkish']
target_lang = 'turkish'

source_lang = lang_list.copy()
source_lang.remove(target_lang)

train_dials = []
dev_dials = []

for lang in source_lang:
    f = open('data/new_dst_{}/train_dials.json'.format(lang))
    train_file = json.load(f)
    for item in train_file:
        item['dial_id'] = lang + '-' + item['dial_id']
        train_dials.append(item)
    f = open('data/new_dst_{}/dev_dials.json'.format(lang))
    dev_file = json.load(f)
    for item in dev_file:
        item['dial_id'] = lang + '-' + item['dial_id']
        dev_dials.append(item)

f = open('data/new_dst_{}/test_dials.json'.format(target_lang))
test_file = json.load(f)

# if mix_dst_{} folder does not exist, create it
if not os.path.exists('data/mix_dst_{}'.format(target_lang)):
    os.makedirs('data/mix_dst_{}'.format(target_lang))

# save train dials into mix_dst_english
with open('data/mix_dst_{}/train_dials.json'.format(target_lang), 'w', encoding='utf-8') as f:
    json.dump(train_dials, f, indent=2, ensure_ascii=False)
with open('data/mix_dst_{}/dev_dials.json'.format(target_lang), 'w', encoding='utf-8') as f:
    json.dump(dev_dials, f, indent=2, ensure_ascii=False)
with open('data/mix_dst_{}/test_dials.json'.format(target_lang), 'w', encoding='utf-8') as f:
    json.dump(test_file, f, indent=2, ensure_ascii=False)
