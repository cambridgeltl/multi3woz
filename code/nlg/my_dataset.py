import json

from datasets import Dataset, DatasetDict
import pandas as pd
import os

class MultilingualMultiWoZDataset():
	def __init__(self, config, language = None):

		assert config
		self.config = config

		project_root_path = config["project"]["project_root_path"]

		self.is_dev = False
		if "is_development" in self.config["project"]:
			self.is_dev = config["project"]["is_development"].lower() == "true"

		self.parallel_dic_path = os.path.join(project_root_path, config["data"]["parallel_dic_path"])
		with open(self.parallel_dic_path, "r", encoding="utf-8") as f:
			self.parallel_dic = json.load(f)


		self.train_languages_json = json.loads(self.config["experiment"]["train_languages"])
		self.train_languages = []

		for this_language, dial_key, in self.train_languages_json.items():
			assert this_language.lower() in ["arabic", "english", "french", "turkish"]
			assert dial_key in self.parallel_dic["train"]
			this_data_path = os.path.join(project_root_path, config["data"][this_language + "_data_path"])

			self.train_languages.append((this_language.lower(), dial_key, this_data_path))

		self.val_languages_json = json.loads(self.config["experiment"]["val_languages"])
		self.val_languages = []

		for this_language, dial_key, in self.val_languages_json.items():
			assert this_language.lower() in ["arabic", "english", "french", "turkish"]
			this_data_path = os.path.join(project_root_path, config["data"][this_language + "_data_path"])

			self.val_languages.append((this_language.lower(), dial_key, this_data_path))

		if language:
			assert language.lower() in ["arabic", "english", "french", "turkish"]

			self.test_languages_json = {language: "random1.0"}
		else:
			self.test_languages_json = json.loads(self.config["experiment"]["test_languages"])
		self.test_languages = []

		for this_language, dial_key, in self.test_languages_json.items():
			assert this_language.lower() in ["arabic", "english", "french", "turkish"]
			assert dial_key in self.parallel_dic["test"]
			this_data_path = os.path.join(project_root_path, config["data"][this_language + "_data_path"])
			self.test_languages.append((this_language.lower(), dial_key, this_data_path))

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

		language_split_data_dic = {}
		for languages in [self.train_languages, self.val_languages, self.test_languages]:
			for language, dial_key , data_path in languages:

				if not language_split_data_dic.get(language):

					split_dic = {
						"train": {},
						"val": {},
						"test": {},
					}
					with open(os.path.join(data_path, "data.json"), "r", encoding="utf-8") as f:
						data = json.load(f)
					f = open(os.path.join(data_path, "valListFile.txt"))
					val_list = f.read().splitlines()
					f.close()
					f = open(os.path.join(data_path, "testListFile.txt"))
					test_list = f.read().splitlines()
					f.close()
					train_list = list(filter(lambda x : x not in test_list + val_list, data.keys()))
					for dial_id, dial in data.items():
						if dial_id in test_list:
							split_dic["test"][dial_id] = dial
						elif dial_id in val_list:
							split_dic["val"][dial_id] = dial
						elif dial_id in train_list:
							split_dic["train"][dial_id] = dial
					language_split_data_dic[language] = split_dic


		data_dic = {
			"train": {},
			"val": {},
			"test": {}
		}

		for data_split, languages  in \
			[("train", self.train_languages), ("val", self.val_languages), ("test", self.test_languages)]:

			for language, dial_key, data_path in languages:


				target_dic = data_dic[data_split]
				target_data = language_split_data_dic[language][data_split]
				target_list = self.parallel_dic[data_split][dial_key]

				for dial_id in target_list:
					target_dic[language + "_"+ dial_id] = target_data[dial_id]

		return data_dic["train"], data_dic["val"], data_dic["test"]


	def load_data(self, task = None):

		if task is not None:
			self.task = task
		else:
			self.task = self.config["experiment"]["task"]


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


	def load_data_from_file(self, data_path , task = None):
		if task is not None:
			self.task = task
		else:
			self.task = self.config["experiment"]["task"]

		assert self.task in ["realisation", "language_modelling"]

		self.raw_data_dic = {}

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
		if len(train_dic) > 0:
			self.raw_data_dic["train"] = train_dic
		if len(val_dic) > 0:
			self.raw_data_dic["val"] = val_dic
		if len(test_dic) > 0:
			self.raw_data_dic["test"] = test_dic

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