from typing import Any
from src.processing_stage import ProcessingStage
from transformers import AutoModel
from src.validator import FunctionDefinitionSchema
from src.state import JSONState
from llm_sdk.llm_sdk import Small_LLM_Model
import numpy as np
import json
import torch


class JSONGenerator(Small_LLM_Model, ProcessingStage):
	"""JSONGenerator
	"""
	def __init__(self) -> None:
		super().__init__()
		self.__json_results = list()
		self.current_state = JSONState.IN_START

	def load_model_vocabulary(self) -> None:
		path_to_vocabulary = self.get_path_to_vocab_file()
		with open(path_to_vocabulary) as file:
			self.__vocabulary = json.load(file)

	def get_allowed_tokens(self) -> list[str]:
		if self.current_state == JSONState.IN_START:
			return ['{']
		return []

	def get_allowed_logits(self, logits) -> list[int]:

		allowed_strs = self.get_allowed_tokens()
		# ['{'] or None
		masked_logits = np.array(["inf"] * len(logits)) 
		for token_str in allowed_strs:
			token_id = self.encode(token_str).tolist()[0][0]
			print("token_id HERE!!!:", token_id)
			masked_logits[token_id] = logits[token_id]
		return list(masked_logits)

	def execute(self, data: Any) -> Any:

		self.load_model_vocabulary()

		print("pipeline generator:")
		self.__functions_definition_name: list[str] = [
			function.name for function in data['functions_definition']
		]
		self.__functions_definition_name.append("null")

		prompt = data['prompts'][0]
		clean_prompt = self.build_clean_prompt(data['functions_definition'], prompt)
		input_ids_as_list = self.encode(clean_prompt).tolist()

		while self.current_state != JSONState.IN_END :
			logits = self.get_logits_from_input_ids(input_ids_as_list[0])
			allowed_strs = self.get_allowed_tokens()
   			# ['{'] or None
			masked_logits = self.get_allowed_logits(logits)
			print("MASKED:")
			print(masked_logits)
			print("*" * 40)
			max_token = np.argmax(masked_logits).tolist()
			print("input_ids before:", input_ids_as_list)
			input_ids_as_list[0].append(max_token)
			print("input_ids after:", input_ids_as_list)
			print("*" * 60)
			print(self.__vocabulary['{'])

			break

			# numpy_logits = logits.numpy()
			# probabilities = self.__softmax_function(logits.numpy())

			# self.current_state = JSONState.IN_END
			# i += 1
			# continue
		return None

	def __softmax_function(self, logits) -> list[float]:
		""" apply softmax function on logits with numpy arrays 
			and return probabilities of scores or logits as np.array 
		"""
		max_logits = np.max(logits)
		exp_values = np.exp(logits - max_logits)
		probabilities = exp_values / np.sum(exp_values)

		return probabilities

	def build_clean_prompt(
    	self, functions_definition: list[FunctionDefinitionSchema], prompt: str
    ) -> Any:
		available_functions = ''
		for function in functions_definition:
			available_functions += function.model_dump_json() + '\n'

		clean_prompt = f"""
		You are a strict function_calling AI assistant.
		your only job is to analyze the user prompt and decide if can use any
		function from THE AVAILABLE FUNCTIONS

		AVAILABLE FUNCTIONS:
			{available_functions}

		USER PROMPT:
			{prompt}
    		
		INSTRUCTIONS:
			choose one matching function
		RULES:
			- output ONLY valid JSON. NO EXTRA TEXT, NO EXPLANATION, NO MARKDOWN.
			- never invent functions that are not listed above
			- extract values from the user prompt accurately
			- ONLY JSON STRUCTURE	
		EXAMPLE OUTPUT:
		   if match one in AVAILABLE FUNCTIONS:
			{{
				"prompt": "<USER PROMPT HERE>",
				"name": "<function_name>",
				"parameters": {{"a": 2.0, "b": 3.0}}
			}}
		   else:
			{{
				"prompt": "<USER PROMPT HERE>",
				"name": null,
				"parameters": null,
			}}
		"""
		return clean_prompt

	def set_token_hugging_face(self) -> None:
		token_hf = "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"
		model = AutoModel.from_pretrained("private/model", token=token_hf)
