TASKS=(english arabic turkish french)
lang=english
few_shots_key=random1.0

saving_dir=output/woz/$lang/mt5_small/new_percent-$few_shots_key/

python T5.py \
  --model_checkpoint 'google/mt5-small' \
  --model_name 'mt5' \
  --train_batch_size 8 \
  --few_shots_percent 100 \
  --GPU 1 \
  --dial_ids_description $few_shots_key \
  --dial_ids_dir ./data/parallel_dic.json  \
  --seed 457\
  --slot_lang slottype \
  --n_epochs 8 \
  --saving_dir $saving_dir \
  --data_dir data/new_dst_$lang