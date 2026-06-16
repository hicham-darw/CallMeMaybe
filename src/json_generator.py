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
			parameters = self.get_parameters_as_str()
			print("parameters:", parameters)
			print("type parameters:", type(parameters))
			return []
		if self.current_state == JSONState.IN_CLOSE_BRACE:
			return ['}']
		
	def get_parameters_as_str(self) -> dict[str, str]:
		for function_definition in self.__functions_definition:
			if function_definition.name == self.current_function_call:
				return function_definition.parameters
		return {}

	def get_allowed_logits(self, logits, generated_str: str, target: list[str]) -> list[int]:

		if not target:
			return logits
		masked_logits = np.full_like(logits, -np.inf)
		for item in target:
			for k, v in self.__vocabulary.items():
				# print(f"decode [v]: {generated_str + self.decode([v])}")
				if item.startswith((generated_str + self.decode([v]))):
					# print(f"token_str: {generated_str + k} {v} v here ===> {v}")
					#  find how to put logits with correctly
					masked_logits[v] = logits[v]
# maybe line under me
				if generated_str + self.decode([v]) == target:
					break
		return masked_logits

	def execute(self, data: Any) -> Any:

		self.load_model_vocabulary()
		self.__swapped_vocabulary = {v: k for k,v in self.__vocabulary.items()}
		self.__functions_definition = data['functions_definition']
		self.__prompts = data['prompts']
		
		self.__functions_definition_name: list[str] = list(map(lambda s: "\"" + s.name + "\"" , self.__functions_definition))
		self.__functions_definition_name.append("'null'")

		print("pipeline generator:")
		
		for prompt_schema in self.__prompts:
			json_result = ''
			generated_str = ''

			self.current_prompt = "\"" + prompt_schema.prompt['prompt'] + "\""
			clean_prompt = self.build_clean_prompt()

			input_ids_as_list = self.encode(clean_prompt).tolist()[0]
			self.__start_json = len(input_ids_as_list)

			while self.current_state != JSONState.IN_END:

				logits = self.get_logits_from_input_ids(input_ids_as_list)
				masked_logits = self.get_allowed_logits(logits, generated_str, self.get_allowed_tokens())
				index_max_logit = np.argmax(masked_logits)

				generated_str += self.decode([int(index_max_logit)])
				input_ids_as_list.append(index_max_logit)

				if self.current_state == JSONState.IN_PARAMETERS_VALUE\
						and self.is_closed_json(generated_str):
					print(f"The end state generated str: {generated_str}")
					self.goto_next_state()
					json_result += generated_str
				elif generated_str in self.get_allowed_tokens(): 
					print(f"The end token.generated str: {generated_str}")
					if generated_str in self.__functions_definition_name:
						self.current_function_call = generated_str.strip('"')

					self.goto_next_state()
					json_result += generated_str
					generated_str = ''
				print("self.__current_state:", self.current_state)
			self.__json_results.append(json_result)
			self.current_state = JSONState.IN_START

			print("json_result: ==>", json_result)		
		return None
	
	def is_closed_json(self, json_str: str) -> bool:
		json_str = json_str.strip()
		if json_str[0] == '{' and json_str[-1] == '}':
			return True
		return False

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
		elif self.current_state == JSONState.IN_NAME_COLON:
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

	def build_clean_prompt(self) -> str:
		available_functions = ''
		for function in self.__functions_definition:
			available_functions += function.model_dump_json() + '\n'

		clean_prompt = f"""
		You are a strict function_calling AI assistant.
		your only job is to analyze the user prompt and decide if can use any
		function from THE AVAILABLE FUNCTIONS

		AVAILABLE FUNCTIONS:
			{available_functions}

		USER PROMPT:
			{self.current_prompt}
    		
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
