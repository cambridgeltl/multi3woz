import json

with open('data/multi-woz/data.json') as input ,\
    open('data/multi-woz/dialogue_acts.json', 'w') as output:
        data = json.load(input)
        dialog_acts = {}
        for id in data:
            dialog_acts[id.replace('.json','')] = {}
            log = data[id]['log']
            for i in range(len(log)):
                if i % 2 == 0:
                     pass
                else:
                    entry = log[i]
                    i = int((i + 1) / 2)
                    if 'dialog_act' in entry:
                        dialog_act = entry['dialog_act']
                        if dialog_act != {}:
                            dialog_acts[id.replace('.json','')][i] = dialog_act
        json.dump(dialog_acts, output, indent=4)
