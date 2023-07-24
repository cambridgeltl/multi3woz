import json
import os

if not os.path.exists("data/rp/"):
    os.makedirs("data/rp/")


for filename in ['train.soloist.json', 'valid.soloist.json', 'test.soloist.json']:
    with open(f'raw_data/' + filename, encoding='utf-8') as input_f, \
        open(f'data/rp/' + filename, 'w', encoding='utf-8') as output_f:
        data = json.load(input_f)
        for entry in data:
            history = entry['history']
            history = ' '.join(history)
            belief = entry['belief']
            reply = entry['reply']
            kb = entry['kb']
            context = '[SEP]'.join([history, belief, kb])
            output_f.write(json.dumps({'context': context, 'response': reply}, ensure_ascii=False))
            output_f.write('\n')