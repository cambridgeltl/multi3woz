lang=arabic
saving_dir=output/woz_ar_dst_mt5_small_seed_1/

python T5.py \
  --model_checkpoint 'google/mt5-small' \
  --model_name 'mt5' \
  --train_batch_size 4 \
  --few_shots_percent 100 \
  --GPU 1 \
  --seed 1 \
  --slot_lang slottype \
  --n_epochs 5 \
  --saving_dir $saving_dir \
  --data_dir data/dst_$lang