import json
from datasets import Dataset, DatasetDict
import pandas as pd
import os
from transformers import AutoTokenizer
from transformers import set_seed


class MultilingualMultiWoZDataset():
	def __init__(self, config, language = None):

		assert config
		self.config = config


		set_seed(int(config["experiment"]["seed"]))

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
			assert dial_key in self.parallel_dic["val"]
			this_data_path = os.path.join(project_root_path, config["data"][this_language + "_data_path"])

			self.val_languages.append((this_language.lower(), dial_key, this_data_path))


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

		selected_mode = config["experiment"]["model_name"]

		self.tokenizer = AutoTokenizer.from_pretrained(selected_mode)

		self.special_token_list = ["[turn_sep]","[history_sep]"]

		num_added_tokens = self.tokenizer.add_tokens(self.special_token_list, special_tokens=True)
		self.turn_sep_id, self.history_sep_id = self.tokenizer.convert_tokens_to_ids(["[turn_sep]", "[history_sep]"])



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

		assert self.task in ["labelling", "intent"]

		dataset_dict = None
		if self.task == "labelling":

			processed_data = self._preprocess_labelling_dataset()
			for data_key, data in processed_data.items():
				data = pd.DataFrame.from_dict(data)
				data = Dataset.from_pandas(data)
				processed_data[data_key] = data

			dataset_dict = DatasetDict(processed_data)

		elif self.task == "intent":
			processed_data = self._preprocess_intent_dataset()
			for data_key, data in processed_data.items():
				data = pd.DataFrame.from_dict(data)
				data = Dataset.from_pandas(data)
				processed_data[data_key] = data
			dataset_dict = DatasetDict(processed_data)


		return dataset_dict

	def _preprocess_intent_dataset(self):

		self.process_mode = self.config["experiment"]["process_mode"]
		self.context_window = int(self.config["experiment"]["context_window"])
		self.max_context_char_length = int(self.config["experiment"]["max_context_char_length"])

		assert self.process_mode in ["all", "user", "system"]

		processed_data = {}
		label_to_index = {}
		index_to_label = {}
		label_counter = -1
		for data_key, dataset in self.raw_data_dic.items():
			processed_data[data_key] = []
			for dial_id, dial in list(dataset.items()):
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
						context_text = " [turn_sep] ".join(context[-(self.context_window - 1):])[
									   :self.max_context_char_length]

					if len(context_text) > 0:
						context_text = context_text + " [history_sep] "
					text = context_text + turn["text"]


					data_entry = {}

					data_entry["text"] = text
					data_entry["context"] = context.copy()
					data_entry["utterance"] = turn["text"]
					data_entry["turn_id"] = turn_id
					data_entry["dail_id"] = dial_id

					temp_label = []
					for temp_intent in turn["dialog_act"].keys():
						if temp_intent not in label_to_index:
							label_counter += 1
							index_to_label[label_counter] = temp_intent
							label_to_index[temp_intent] = label_counter
						temp_label.append(label_to_index[temp_intent])
					data_entry["intent_idx"] = temp_label



					processed_data[data_key].append(data_entry)
					context.append(turn['text'])

		self.label_to_index = label_to_index
		self.index_to_label = index_to_label
		return processed_data

	def _build_bio_tag_sequence(self):
		all_label_set = set()
		all_label_set.add("O")

		tokenizer = self.tokenizer

		label_to_index = {}
		index_to_label = {}
		label_to_index["O"] = 0
		index_to_label[0] = "O"
		label_counter = 0

		for data_key, data_set in self.raw_data_dic.items():
			for dial_id, dial in data_set.items():

				for turn_id, turn in enumerate(dial['log']):
					text = turn["text"]
					tokenized_result = tokenizer(text)
					tokens = tokenized_result.tokens()
					bio_token_tag_seq = ["O" for _ in tokens]
					bio_char_tag_seq = ["O" for _ in text]

					all_token_idx = list(map(lambda x: x[0], filter(lambda x: x[1] is not None,
																	list(enumerate(tokenized_result.word_ids())))))

					char_token_map = {}
					for token_idx in all_token_idx:
						result = tokenized_result.token_to_chars(token_idx)
						for char_idx in range(result.start, result.end):
							char_token_map[char_idx] = token_idx

					for item in turn["span_info"]:
						key = item[0] + "+" + item[1]
						start = item[3]
						end = item[4]
						all_label_set.add("B-" + key)
						all_label_set.add("I-" + key)

						if "B-" + key not in label_to_index:
							label_counter += 1
							index_to_label[label_counter] = "B-" + key
							label_to_index["B-" + key] = label_counter

						if "I-" + key not in label_to_index:
							label_counter += 1
							index_to_label[label_counter] = "I-" + key
							label_to_index["I-" + key] = label_counter

						for char_idx in range(start, end):
							if char_idx < len(bio_char_tag_seq):
								bio_char_tag_seq[char_idx] = key
								if char_idx in char_token_map:
									bio_token_tag_seq[char_token_map[char_idx]] = key


					new_bio_token_tag_seq = []
					pre_tag = None
					for tag, tk_word_id in zip(bio_token_tag_seq, tokenized_result.word_ids()):
						if tk_word_id == None:
							new_bio_token_tag_seq.append(-100)
						elif tag == "O":
							new_bio_token_tag_seq.append(label_to_index["O"])
						elif tag != pre_tag:
							new_bio_token_tag_seq.append(label_to_index["B-" + tag])
						else:
							new_bio_token_tag_seq.append(label_to_index["I-" + tag])
						pre_tag = tag

					new_bio_char_tag_seq = []
					pre_tag = None
					for tag in bio_char_tag_seq:
						if tag == "O":
							new_bio_char_tag_seq.append("O")
						elif tag != pre_tag:
							new_bio_char_tag_seq.append("B-" + tag)
						else:
							new_bio_char_tag_seq.append("I-" + tag)
						pre_tag = tag

					turn["token_bio_tag"] = new_bio_token_tag_seq
					turn["char_bio_tag"] = new_bio_char_tag_seq
					turn["tokenized_result"] = tokenized_result

		return  self.raw_data_dic, label_to_index, index_to_label

	def _preprocess_labelling_dataset(self):

		processed_raw_dic, label_to_index, index_to_label = self._build_bio_tag_sequence()

		self.label_to_index = label_to_index
		self.index_to_label = index_to_label
		self.processed_data_dic = processed_raw_dic

		self.process_mode = self.config["experiment"]["process_mode"]
		self.context_window = int(self.config["experiment"]["context_window"])
		self.max_context_char_length = int(self.config["experiment"]["max_context_char_length"])

		assert self.process_mode in ["all", "user", "system"]


		processed_data = {}

		all_da = []
		all_intent = []
		all_tag = []




		for data_key, data_set in self.raw_data_dic.items():
			processed_data[data_key] = []

			for dial_id, dial in list(data_set.items())[: 1 if self.is_dev else len(data_set)]:
				context = []

				for turn_id, turn in enumerate(dial['log']):
					if self.process_mode == 'user' and turn_id % 2 == 1:
						context.append(turn['text'])
						continue
					elif self.process_mode == 'system' and turn_id % 2 == 0:
						context.append(turn['text'])
						continue

					text = turn["text"]
					tokenized_result = turn["tokenized_result"]
					tokens = tokenized_result.tokens()

					data_entry = {}

					data_entry["tokens"] = tokens
					data_entry["text"] = text
					data_entry["char_bio_tag"] = turn["char_bio_tag"]
					data_entry["context"] = context.copy()
					data_entry["turn_id"] = turn_id
					data_entry["dail_id"] = dial_id


					data_entry["input_ids"] = tokenized_result["input_ids"]
					data_entry["labels"] = turn["token_bio_tag"]
					data_entry["attention_mask"] = tokenized_result["attention_mask"]


					if self.context_window <= 1:
						context_text = ""
					else:
						context_text = " [turn_sep] ".join(context[-(self.context_window - 1):])[
									   :self.max_context_char_length]

					context_tokenized_result = self.tokenizer(context_text)
					char_offset = len(context_text)
					token_offset = len(context_tokenized_result["input_ids"])

					context_labels = [-100 for _ in context_tokenized_result["input_ids"]]

					tokenized_result["input_ids"][0] = self.history_sep_id
					context_tokenized_result["input_ids"][-1] = self.history_sep_id

					data_entry["input_ids"] = context_tokenized_result["input_ids"] + tokenized_result["input_ids"]
					data_entry["labels"] = context_labels +  turn["token_bio_tag"]

					data_entry["attention_mask"] = context_tokenized_result["attention_mask"] + tokenized_result["attention_mask"]
					data_entry["char_offset"] = char_offset
					data_entry["token_offset"] = token_offset

					assert len(data_entry["input_ids"]) == len(data_entry["labels"])
					assert len(data_entry["input_ids"]) == len(data_entry["attention_mask"])

					processed_data[data_key].append(data_entry)
					context.append(turn['text'])

		return processed_data

	def map_token_bio_to_char_bio(self, data_entry, prediction_seq):

		assert self.tokenizer
		tokenized_result = self.tokenizer(data_entry["text"])
		assert tokenized_result.tokens() == data_entry["tokens"]

		prediction_seq = prediction_seq[data_entry["token_offset"]:]
		all_token_idx = list(map(lambda x: x[0], filter(lambda x: x[1] is not None,
														list(enumerate(tokenized_result.word_ids())))))

		id2label = self.index_to_label

		new_char_bio_tag = ["O" for _ in data_entry["text"]]
		pre_slot_val = "O"
		pre_bio = "O"
		current_start_idx = -1
		current_end_idx = -1
		for token_idx in all_token_idx:
			result = tokenized_result.token_to_chars(token_idx)
			current_tag = (id2label[prediction_seq[token_idx]])
			current_slot_val = "O" if current_tag == "O" else current_tag[2:]
			current_bio = current_tag[0]
			if current_bio == "I":
				if current_start_idx != -1 and pre_slot_val != "O" and pre_bio != "O":

					current_end_idx = result.end
			else:
				if pre_slot_val != "O":
					assert pre_bio != "O"
					assert current_start_idx != -1
					assert current_start_idx < current_end_idx
					for i in range(current_start_idx, current_end_idx):
						new_char_bio_tag[i] = "I-" + pre_slot_val
					new_char_bio_tag[current_start_idx] = "B-" + pre_slot_val

				if current_bio == "B":
					pre_slot_val = current_slot_val
					pre_bio = "B"
					current_start_idx = result.start
					current_end_idx = result.end
				else:
					pre_slot_val = "O"
					pre_bio = "O"
					current_start_idx = -1
					current_end_idx = -1

		if pre_slot_val != "O":
			assert pre_bio != "O"
			assert current_start_idx != -1
			assert current_end_idx != -1
			for i in range(current_start_idx, current_end_idx):
				new_char_bio_tag[i] = "I-" + pre_slot_val
			new_char_bio_tag[current_start_idx] = "B-" + pre_slot_val
		return new_char_bio_tag
