import json
import os

if not os.path.exists("data/bsp/"):
    os.makedirs("data/bsp/")

for filename in ['train.soloist.json', 'valid.soloist.json', 'test.soloist.json']:


    with open(f'raw_data/' + filename, encoding='utf-8') as input_f, \
        open(f'data/bsp/' + filename, 'w', encoding='utf-8') as output_f:
        data = json.load(input_f)
        for entry in data:
            history = entry['history']
            history = ' '.join(history)
            belief = entry['belief']
            output_f.write(json.dumps({'history': history, 'bs': belief}, ensure_ascii=False))
            output_f.write('\n')
                