from dataclasses import dataclass, field
from typing import Optional
from transformers import pipeline, HfArgumentParser
from transformers import MT5Tokenizer, MT5ForConditionalGeneration
import sys
import os
import json
import sqlite3
from tqdm import tqdm
domains = ['restaurant', 'hotel', 'attraction', 'train'] #, 'taxi', 'hospital']#, 'police']
dbs = {}


@dataclass
class PostprocessingArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune from.
    """

    bsp_model_name_or_path: str = field(
        metadata={"help": "Path to pretrained bsp model or model identifier from huggingface.co/models"}
    )
    rp_model_name_or_path: str = field(
        metadata={"help": "Path to pretrained rp model or model identifier from huggingface.co/models"}
    )
    input_file_path: str = field(
        metadata={"help": "The input test file (in the soloist preprocessed format)."}
    )
    output_file_path: str = field(
        metadata={"help": "The output file destination (in the MultiWOZ Context-to-Response Evaluation format)."}
    )
    db_path: str = field(
        metadata={"help": "The output file destination (in the MultiWOZ Context-to-Response Evaluation format)."}
    )

def queryResult(domain, bs):
    """Returns the list of entities for a given domain
    based on the annotation of the belief state"""
    # query the db
    sql_query = "select * from {}".format(domain)

    flag = True
    #print turn['metadata'][domain]['semi']
    for key, val in bs.items():
        if val == "" or val == "dont care" or val == 'not mentioned' or val == "don't care" or val == "dontcare" or val == "do n't care":
            pass
        else:
            if flag:
                sql_query += " where "
                val2 = val.replace("'", "''")
                #val2 = normalize(val2)
                # change query for trains
                if key == 'leaveAt':
                    sql_query += r" " + key + " > " + r"'" + val2 + r"'"
                elif key == 'arriveBy':
                    sql_query += r" " + key + " < " + r"'" + val2 + r"'"
                else:
                    sql_query += r" " + key + "=" + r"'" + val2 + r"'"
                flag = False
            else:
                val2 = val.replace("'", "''")
                #val2 = normalize(val2)
                if key == 'leaveAt':
                    sql_query += r" and " + key + " > " + r"'" + val2 + r"'"
                elif key == 'arriveBy':
                    sql_query += r" and " + key + " < " + r"'" + val2 + r"'"
                else:
                    sql_query += r" and " + key + "=" + r"'" + val2 + r"'"

    #try:  # "select * from attraction  where name = 'queens college'"
    #print sql_query
    #print domain
    num_entities = len(dbs[domain].execute(sql_query).fetchall())

    return num_entities

def main():
    global dbs
    parser = HfArgumentParser((PostprocessingArguments))
    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        # If we pass only one argument to the script and it's the path to a json file,
        # let's parse it to get our arguments.
        postprocessing_args, = parser.parse_json_file(json_file=os.path.abspath(sys.argv[1]))
    else:
        postprocessing_args, = parser.parse_args_into_dataclasses()

    bsp_tokenizer = MT5Tokenizer.from_pretrained(postprocessing_args.bsp_model_name_or_path)
    bsp_model = MT5ForConditionalGeneration.from_pretrained(postprocessing_args.bsp_model_name_or_path).to('cuda')

    rp_tokenizer = MT5Tokenizer.from_pretrained(postprocessing_args.rp_model_name_or_path)
    rp_model = MT5ForConditionalGeneration.from_pretrained(postprocessing_args.rp_model_name_or_path).to('cuda')

    # pipe_bsp = pipeline(task='text2text-generation', model=postprocessing_args.bsp_model_name_or_path, device=0, args_parser={'prefix':'predict the belief state: '})
    # pipe_rp = pipeline(task='text2text-generation', model=postprocessing_args.rp_model_name_or_path, device=0, args_parser={'prefix':'predict the response: '})

    for domain in domains:
        db_file_name = '{}-dbase.db'.format(domain)
        db = os.path.join(postprocessing_args.db_path, db_file_name)
        conn = sqlite3.connect(db)
        c = conn.cursor()
        dbs[domain] = c


    with open(postprocessing_args.input_file_path) as input_f, \
        open(postprocessing_args.output_file_path, 'w', encoding='utf-8') as output_f:
        data = json.load(input_f)
        output_list = []

        for entry in tqdm(data):
            history = entry['history']
            history = ' '.join(history)

            bs_inputs = bsp_tokenizer.batch_encode_plus(["predict the belief state: " + history], max_length=1024, return_tensors="pt", pad_to_max_length=True, truncation=True).to('cuda')
            # bs_outputs = bsp_model.generate(bs_inputs['input_ids'], num_beams=4, max_length=512, early_stopping=True)
            bs_outputs = bsp_model.generate(bs_inputs['input_ids'], do_sample=True, max_length=512, top_p=0.7)
            
            pred_bs = [bsp_tokenizer.decode(g, skip_special_tokens=True, clean_up_tokenization_spaces=False) for g in bs_outputs][0].strip().split('belief : ')[1]

            # pred_bs = pipe_bsp(history)[0]['generated_text']
            # pred_bs = pred_bs.strip().split('belief : ')[1]

            # example: 
            # belief : taxi destination = pipasha restaurant ; departure = holy trinity church ; arriveby = 18:00 | restaurant food = indian ; pricerange = expensive ; area = east | booking day = monday ; people = 7 ; time = 18:00 | attraction name = king's college

            states_json = {}

            try:
                for state in pred_bs.split('|'):
                    state_json = {}
                    substates = state.split(";")
                    state_key = ""
                    for i in range(len(substates)):
                        try:
                            substate = substates[i]
                            if i == 0:
                                state_key = substate.split(" ")[0]
                                substate = substate.replace(state_key, "")
                                substate_key, substate_value = substate.split('=')
                                state_json[substate_key.strip()] = substate_value.strip()
                            else:
                                substate_key, substate_value = substate.split('=')
                                state_json[substate_key.strip()] = substate_value.strip()
                        except:
                            pass
                    states_json[state_key] = state_json
            except:
                pass

            if not states_json:
                kb = f'kb: '

            for active_domain in states_json:
                try:
                    if active_domain not in ['none','taxi','hospital']:
                        kb_nums = queryResult(state_key, states_json[state_key])
                        if active_domain != 'train':
                            if kb_nums > 5:
                                kb_nums = 'more than five'
                            elif kb_nums == 0:
                                kb_nums = 'zero'
                            elif kb_nums == 1:
                                kb_nums = 'one'
                            elif kb_nums == 2:
                                kb_nums = 'two'
                            elif kb_nums == 3:
                                kb_nums = 'three'
                            elif kb_nums == 4:
                                kb_nums = 'four'
                        else:
                            if kb_nums > 40:
                                kb_nums = 'more than five'
                            elif kb_nums == 0:
                                kb_nums = 'zero'
                            elif kb_nums <= 2:
                                kb_nums = 'one'
                            elif kb_nums <= 5:
                                kb_nums = 'two'
                            elif kb_nums <= 10:
                                kb_nums = 'three'
                            elif kb_nums <= 40:
                                kb_nums = 'four'
                        kb = f'kb : {active_domain} {kb_nums}'
                    else:
                        kb = f'kb : {active_domain}'
                except:
                    kb = f'kb: '
            
            context = '[SEP]'.join([history, pred_bs, kb])

            r_inputs = rp_tokenizer.batch_encode_plus(["predict the response: " + context], max_length=1024, return_tensors="pt", pad_to_max_length=True, truncation=True).to('cuda')
            r_outputs = rp_model.generate(bs_inputs['input_ids'], num_beams=4, max_length=512, early_stopping=True)

            pred_response = [rp_tokenizer.decode(g, skip_special_tokens=True, clean_up_tokenization_spaces=False) for g in r_outputs][0].strip()

            # pred_response = pipe_rp(context)[0]['generated_text']

            output_entry = pred_bs + pred_response
            
            output_list.append([output_entry])
        
        json.dump(output_list, output_f, indent=2,ensure_ascii=False)
               

if __name__ == "__main__":
    main()