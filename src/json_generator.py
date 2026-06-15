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
		if self.current_state == JSONState.IN_PROMPT_KEY:
			return ["\"prompt\""]
		if self.current_state == JSONState.IN_PROMPT_COLON:
			return [':']
		if self.current_state == JSONState.IN_PROMPT_VALUE:
			return [self.current_prompt]
		if self.current_state == JSONState.IN_COMMA_AFTER_PROMPT:
			return [","]
		if self.current_state == JSONState.IN_NAME_KEY:
			return ["\"name\""]
		if self.current_state == JSONState.IN_NAME_COLON:
			return [":"]
		if self.current_state == JSONState.IN_NAME_VALUE:
			return self.__functions_definition_name
		if self.current_state == JSONState.IN_COMMA_AFTER_NAME:
			return [","]
		if self.current_state == JSONState.IN_PARAMETERS_KEY:
			return["\"parameters\""]
		if self.current_state == JSONState.IN_PARAMETERS_COLON:
			return [":"]
		if self.current_state == JSONState.IN_PARAMETERS_VALUE:
			return []
		if self.current_state == JSONState.IN_CLOSE_BRACE:
			return ['}']
		
		

	def get_allowed_logits(self, logits, generated_str: str, target: str) -> list[int]:

		masked_logits = np.full_like(logits, -np.inf)
		for item in target:
			for k, v in self.__vocabulary.items():
				# print(f"decode [v]: {generated_str + self.decode([v])}")
				if item.startswith((generated_str + self.decode([v]))):
					print(f"token_str: {generated_str + k} {v} v here ===> {v}")
					#  find how to put logits with correctly
					masked_logits[v] = logits[v]
				# if generated_str + self.decode([v]) == target:
				# 	break
		return masked_logits

	def execute(self, data: Any) -> Any:

		self.load_model_vocabulary()
		self.__swapped_vocabulary = {v: k for k,v in self.__vocabulary.items()}
		
		print("pipeline generator:")
		self.__functions_definition_name: list[str] = [
			function.name for function in data['functions_definition']
		]
		self.__functions_definition_name.append("null")

		prompt = data['prompts'][0]
		self.current_prompt = prompt.prompt['prompt']
		clean_prompt = self.build_clean_prompt(data['functions_definition'], prompt)
		input_ids_as_list = self.encode(clean_prompt).tolist()[0]
		self.__start_json = len(input_ids_as_list)
		generated_str = ''
		while self.current_state != JSONState.IN_PARAMETERS_VALUE:
			logits = self.get_logits_from_input_ids(input_ids_as_list)
			masked_logits = self.get_allowed_logits(logits, generated_str, self.get_allowed_tokens())
			index_max_logit = np.argmax(masked_logits)
			generated_str += self.__swapped_vocabulary[int(index_max_logit)]
			# print("input_ids before:", input_ids_as_list)
			input_ids_as_list.append(index_max_logit)
			# print("input_ids after:", input_ids_as_list)
			print("*" * 60)

			print(f"generated_str now: {generated_str}")
			print(f"allowed_tokens: {self.get_allowed_tokens()}")
			if generated_str in self.get_allowed_tokens(): 
				print("The end token.generated str: {generated_str}")
				self.goto_next_state()
				generated_str = ''
			print("self.__current_state:", self.current_state)
			# print("this max_token by argmax:|", self.decode(int(max_token)), "|", end="")
			# print()
		
		return None

	def goto_next_state(self) -> None:
		if self.current_state == JSONState.IN_START:
			self.current_state = JSONState.IN_PROMPT_KEY
		elif self.current_state == JSONState.IN_PROMPT_KEY:
			self.current_state = JSONState.IN_PROMPT_COLON
		elif self.current_state == JSONState.IN_PROMPT_COLON:
			self.current_state = JSONState.IN_PROMPT_VALUE
		elif self.current_state == JSONState.IN_PROMPT_VALUE:
			self.current_state = JSONState.IN_COMMA_AFTER_PROMPT
		elif self.current_state == JSONState.IN_COMMA_AFTER_PROMPT:
			self.current_state = JSONState.IN_NAME_KEY
		elif self.current_state == JSONState.IN_NAME_KEY:
			self.current_state = JSONState.IN_NAME_COLON
		elif slef.current_state == JSONState.IN_NAME_COLON:
			self.current_state = JSONState.IN_NAME_VALUE
		elif self.current_state == JSONState.IN_NAME_VALUE:
			self.current_state = JSONState.IN_COMMA_AFTER_NAME
		elif self.current_state == JSONState.IN_COMMA_AFTER_NAME:
			self.current_state = JSONState.IN_PARAMETERS_KEY
		elif self.current_state == JSONState.IN_PARAMETERS_KEY:
			self.current_state = JSONState.IN_PARAMETERS_COLON
		elif self.current_state == JSONState.IN_PARAMETERS_COLON:
			self.current_state = JSONState.IN_PARAMETERS_VALUE
		elif self.current_state == JSONState.IN_PARAMETERS_VALUE:
			self.current_state = JSONState.IN_END
		else:
			print("Error finished and continue")
			
	# def __softmax_function(self, logits) -> list[float]:
	# 	""" apply softmax function on logits with numpy arrays 
	# 		and return probabilities of scores or logits as np.array 
	# 	"""
	# 	max_logits = np.max(logits)
	# 	exp_values = np.exp(logits - max_logits)
	# 	probabilities = exp_values / np.sum(exp_values)

	# 	return probabilities

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
