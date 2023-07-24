import json
import os
import sys
import numpy as np
lang_list = ['english', 'arabic', 'french', 'turkish']
seed_list = ['357','457','557']
directory = "./output/"
mean_jga = []
mean_ta = []
mean_f1 = []
for lang in lang_list:
    jga_list = []
    ta_list = []
    f1_list = []
    for seed in seed_list:
        target_directory = directory + lang + '/mt5/small-' + seed +'/5epochs/google/mt5-smallmt5_except_domain_none_slotlang_slottype_lr_0.0001_epoch_5_seed_' +seed
        results_dir = target_directory + '/results'
        f = open(os.path.join(results_dir, 'zeroshot_result.json'))
        results = json.load(f)
        jga_list.append(results["Joint Acc"])
        ta_list.append(results["Turn Acc"])
        f1_list.append(results["Joint F1"])
    print(lang)
    print(np.mean(jga_list))
    print(np.mean(ta_list))
    print(np.mean(f1_list))
    mean_jga.append(np.mean(jga_list))
    mean_ta.append(np.mean(ta_list))
    mean_f1.append(np.mean(f1_list))
print(np.mean(mean_jga))
print(np.mean(mean_ta))
print(np.mean(mean_f1))