import json
from datasets import Dataset, DatasetDict
import pandas as pd
import os

class MultilingualMultiWoZDataset():
	def __init__(self, config, language = None):

		assert config
		self.config = config

		self.language = self.config["experiment"]["language"].lower()

		if language:
			self.language = language.lower()

		assert self.language in ["arabic", "english", "french", "turkish"]

		project_root_path = config["project"]["project_root_path"]

		self.is_dev = False
		if "is_development" in self.config["project"]:
			self.is_dev = config["project"]["is_development"].lower() == "true"

		self.data_path = os.path.join(project_root_path, config["data"][self.language + "_data_path"])

		self.raw_train_dic, self.raw_val_dic, self.raw_test_dic = self._load_raw_dataset()

		self.raw_data_dic = {
			"train": self.raw_train_dic,
			"val": self.raw_val_dic,
			"test": self.raw_test_dic,
		}

		self.task = None

		self.selected_model = config["experiment"]["model_name"]

		self.tokenizer = None

		self.special_token_list = ["[turn_sep]","[history_sep]", "[act_sep]"]


	def _load_raw_dataset(self):

		with open(os.path.join(self.data_path, "data.json"), "r", encoding="utf-8") as f:
			data = json.load(f)

		f = open(os.path.join(self.data_path, "valListFile.txt"))
		val_list = f.read().splitlines()
		f.close()
		f = open(os.path.join(self.data_path, "testListFile.txt"))
		test_list = f.read().splitlines()
		f.close()

		train_dic = {}
		val_dic = {}
		test_dic = {}

		for dial_id, dial in data.items():
			if dial_id in test_list:
				test_dic[dial_id] = dial
			elif dial_id in val_list:
				val_dic[dial_id] = dial
			else:
				train_dic[dial_id] = dial

		assert len(train_dic) + len(val_dic) + len(test_dic) == len(data)
		return train_dic, val_dic, test_dic

	def load_data(self, task = None):

		if task is not None:
			self.task = task
		else:
			self.task = self.config["experiment"]["task"]

		# we only support the following tasks
		assert self.task in ["realisation", "language_modelling"]

		dataset_dict = None
		if self.task == "realisation":
			processed_data = self._preprocess_realisation_dataset()
			for data_key, data in processed_data.items():
				data = pd.DataFrame.from_dict(data)
				data = Dataset.from_pandas(data)
				processed_data[data_key] = data
			dataset_dict = DatasetDict(processed_data)

		elif self.task == "language_modelling":
			processed_data = self._preprocess_language_modelling_dataset()
			for data_key, data in processed_data.items():
				data = pd.DataFrame.from_dict(data)
				data = Dataset.from_pandas(data)
				processed_data[data_key] = data
			dataset_dict = DatasetDict(processed_data)

		return dataset_dict

	def _preprocess_language_modelling_dataset(self):

		self.process_mode = self.config["experiment"]["process_mode"]
		assert self.process_mode in ["all", "user", "system"]

		self.context_window = int(self.config["experiment"]["context_window"])
		self.max_context_char_length = int(self.config["experiment"]["max_context_char_length"])

		processed_data = {}
		for data_key, dataset in self.raw_data_dic.items():
			processed_data[data_key] = []
			for dial_id, dial in list(dataset.items())[:]:
				context = []
				for turn_id, turn in enumerate(dial['log']):
					if self.process_mode == 'user' and turn_id % 2 == 1:
						context.append(turn['text'])
						continue
					elif self.process_mode == 'system' and turn_id % 2 == 0:
						context.append(turn['text'])
						continue

					assert self.context_window > 1

					context_text = " [turn_sep] ".join(context[-(self.context_window - 1):])[:self.max_context_char_length]

					source_text = context_text


					target_text = turn["text"]
					data_entry = {}
					data_entry["source"] = source_text
					data_entry["target"] = target_text
					data_entry["language"] = self.language
					data_entry["turn_id"] = turn_id
					data_entry["dail_id"] = dial_id
					data_entry["context"] = context.copy()
					data_entry["utterance"] = turn["text"]

					processed_data[data_key].append(data_entry)
					context.append(turn['text'])

		return processed_data

	def _preprocess_realisation_dataset(self):


		self.process_mode = self.config["experiment"]["process_mode"]
		assert self.process_mode in ["all", "user", "system"]

		self.context_window = int(self.config["experiment"]["context_window"])
		self.max_context_char_length = int(self.config["experiment"]["max_context_char_length"])

		processed_data = {}
		for data_key, dataset in self.raw_data_dic.items():
			processed_data[data_key] = []
			for dial_id, dial in list(dataset.items())[:]:

				context = []
				for turn_id, turn in enumerate(dial['log']):
					if self.process_mode == 'user' and turn_id % 2 == 1:
						context.append(turn['text'])
						continue
					elif self.process_mode == 'system' and turn_id % 2 == 0:
						context.append(turn['text'])
						continue

					if self.context_window <= 1:
						context_text = ""
					else:
						context_text = " [turn_sep] ".join(context[-(self.context_window - 1):])[:self.max_context_char_length]

					dialogue_act = turn["dialog_act"]



					da_string = self._linearise_act(dialogue_act)

					if len(context_text) > 0:
						source_text = context_text + " [history_sep] " + da_string
					else:
						source_text = da_string

					target_text = turn["text"]
					data_entry = {}
					data_entry["source"] = source_text
					data_entry["target"] = target_text
					data_entry["language"] = self.language
					data_entry["turn_id"] = turn_id
					data_entry["dail_id"] = dial_id
					data_entry["context"] = context.copy()
					data_entry["utterance"] = turn["text"]
					data_entry["da_string"] = da_string

					processed_data[data_key].append(data_entry)
					context.append(turn['text'])

		return processed_data



	def _linearise_act(self, dial_act):

		da_string_list = []
		first_inten = True
		for domain_intent_pair, slot_val_list in dial_act.items():

			if not first_inten:
				da_string_list.append(";")
			first_inten = False

			domain, intent = domain_intent_pair.split("-")
			domain, intent = domain.lower(), intent.lower()


			da_string_list.append("["+domain+"]")
			da_string_list.append("["+intent+"]")
			da_string_list.append("(")

			is_first = True
			for slot_val in slot_val_list:
				slot, val = slot_val
				slot, val = slot.lower(), val.lower()
				if not is_first:
					da_string_list.append(",")
				is_first = False
				da_string_list.append("["+slot+"]")

				da_string_list.append("["+val+"]")
			da_string_list.append(")")
		da_string =  "".join(da_string_list)
		return da_string

