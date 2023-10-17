#!/bin/bash

cap_lang=Turkish
lang=turkish
python create_data.py \
  --main_dir data/$cap_lang \
  --target_dir data/dst_$lang

cap_lang=English
lang=english
python create_data.py \
  --main_dir data/$cap_lang \
  --target_dir data/dst_$lang

cap_lang=French
lang=french
python create_data.py \
  --main_dir data/$cap_lang \
  --target_dir data/dst_$lang

cap_lang=Arabic
lang=arabic
python create_data.py \
  --main_dir data/$cap_lang \
  --target_dir data/dst_$lang