from typing import Any
from src.processing_stage import ProcessingStage
from transformers import AutoModel
from src.validator import FunctionDefinitionSchema
from src.prompt_builder import PromptBuilder
from src.state import JSONState
from llm_sdk.llm_sdk import Small_LLM_Model
import numpy as np
from src.finite_state_machine import FiniteStateMachine
from src.filter_decoder import FilterDecoder
from src.state import JSONStatic, ParameterState
import json
import re

import sys
import time


class JSONGenerator(Small_LLM_Model, ProcessingStage):
	"""JSONGenerator generate json by llm with constrained decoding
	"""
	def __init__(self) -> None:
		"""init method for loading model init attribute for generatiing json"""
		super().__init__()
		self.__json_results: list[str] = list()

		self.__prompt_builder = PromptBuilder()

		self.__fsm = FiniteStateMachine()
		self.__filter_decoder = FilterDecoder()
		self.__ids_for_strings = []
		self.__ids_for_numbers = []


		self.__index_key_param: int = 0
		self.__current_function_name: str = ''
		self.__dynamic_generated: str = ''
		self.__dynamic_ids: list[int] = []
		self.__json_result: str = ''

	def __prepare_function_names(self) -> None:
		"""prepare function names in set for duplicating"""
		self.__function_names: list[str] = list(
			function.name + "\", " for function in self.__functions_definition
		)
		self.__ids_function_names_set: set[int] = set()
		for name in self.__function_names:
			function_ids = self.encode(name).tolist()[0]
			for token_id in function_ids:
				self.__ids_function_names_set.add(token_id)

	def __load_vocabulary(self) -> None:
		vocab_path = self.get_path_to_vocab_file()
		with open(vocab_path) as file:
			self.__vocabulary = json.load(file)

	def __init_ids_for_parameters(self) -> None:

		for token_id in self.__vocabulary.values():
			decoded_id = self.decode(token_id)
			if decoded_id.isascii() and not decoded_id in ",}":
				self.__ids_for_strings.append(token_id)
			if decoded_id.isdigit() or decoded_id in ".\"":
				self.__ids_for_numbers.append(token_id)

	def __prepare_data(self, data: Any) -> None:
		"""prepare data for generating json file"""		

		self.__functions_definition = data['functions_definition']
		self.__prompts = data['prompts']
		self.__prepare_function_names()

		self.__load_vocabulary()
		self.__prompt_builder.set_available_functions(self.__functions_definition)

		tokens_before_prompt = self.encode(JSONStatic.STR_BEFORE_PROMPT.value).tolist()[0]
		tokens_before_name = self.encode(JSONStatic.STR_BEFORE_NAME.value).tolist()[0]
		tokens_before_parameters = self.encode(JSONStatic.STR_BEFORE_PARAMETERS.value).tolist()[0]

		self.__filter_decoder.set_tokens_before_prompt(tokens_before_prompt)
		self.__filter_decoder.set_tokens_before_name(tokens_before_name)
		self.__filter_decoder.set_tokens_before_parameters(tokens_before_parameters)
		
		self.__init_ids_for_parameters()
		
		self.__prefix_ids: list[int] = self.encode(self.__prompt_builder()).tolist()[0]

	# generating
	def __generate_tokens_before_prompt(self) -> None:
		"""generate tokens in state BEFORE_PROMPT"""
		self.__ids_current_prompt += self.__filter_decoder.get_tokens_before_prompt()
		self.__json_result += self.__fsm.get_static_json()
		self.__fsm.set_static_json(JSONStatic.STR_BEFORE_NAME)
		self.__fsm.set_state(JSONState.IN_PROMPT)

	def __generate_tokens_in_prompt(self) -> None:
		"""generate token in state IN_PROMPT"""
		self.__ids_current_prompt += self.encode(self.__current_prompt + "\", ").tolist()[0]
		self.__json_result += self.__current_prompt.replace("\"", "'") + "\", "
		self.__fsm.set_state(JSONState.BEFORE_NAME)

	def __generate_tokens_before_name(self) -> None:
		"""generate tokens in state BEFORE_NAME"""
		self.__ids_current_prompt += self.__filter_decoder.get_tokens_before_name()
		self.__json_result += self.__fsm.get_static_json()
		self.__fsm.set_static_json(JSONStatic.STR_BEFORE_PARAMETERS)
		self.__fsm.set_state(JSONState.IN_NAME)
	
	def __is_at_least_one_function(self, generated: str) -> bool:
		"""found at least function name start with arguments"""
		for function_name in self.__function_names:
			if function_name.startswith(generated):
				return True
		return False
		
	def __generate_tokens_in_name(self) -> None:
		"""generate tokens in state IN_NAME"""
		full_ids = self.__ids_current_prompt + self.__dynamic_ids
		logits = self.get_logits_from_input_ids(full_ids)
		masked_logits = np.full(len(logits), -np.inf)
		for index_id in self.__ids_function_names_set:
			if self.__is_at_least_one_function(self.__dynamic_generated + self.decode([index_id])):
				masked_logits[index_id] = logits[index_id]
				
		index_max_logit = np.argmax(masked_logits)
		self.__dynamic_ids.append(int(index_max_logit))
		self.__dynamic_generated += self.decode([int(index_max_logit)])
		if self.__filter_decoder.is_found_only_one_function(self.__dynamic_generated, self.__function_names):
			self.__dynamic_generated = self.__get_only_available_function(self.__dynamic_generated)
			self.__dynamic_generated = self.__dynamic_generated
			self.__json_result += self.__dynamic_generated
			self.__current_function_name = self.__dynamic_generated
			self.__dynamic_ids = self.encode(self.__dynamic_generated).tolist()[0]
			self.__ids_current_prompt += self.__dynamic_ids
			self.__dynamic_ids = []
			self.__dynamic_generated = ''
			
			self.__fsm.set_state(JSONState.BEFORE_PARAMETERS)

	
	def __generate_tokens_before_parameters(self) -> None:
		"""generate tokens instate BEFORE_PARAMETERS"""
		self.__ids_current_prompt += self.__filter_decoder.get_tokens_before_parameters()
		self.__json_result += self.__fsm.get_static_json()
		self.__fsm.set_state(JSONState.IN_PARAMETERS)
		self.__dynamic_generated = ""
		self.__dynamic_ids = []

	def __generate_tokens_in_key_parameters(self, parameters: dict[str, dict[str, str]]) -> None:
		"""generate tokens in state IN_PARAMETERS IN_KEY"""
		for index, key_name in enumerate(parameters.keys()):
			if index == self.__index_key_param:
				break
		
		self.__json_result += f'"{key_name}": '
		self.__ids_current_prompt += self.encode(f"\"{key_name}\": ").tolist()[0]
		self.__fsm.set_parameters_state(ParameterState.IN_VALUE)

	def __masked_logits_by_type(self, logits: list[float], type_mask: str) -> list[float]:
		
		if type_mask == '':
			return logits

		masked_logits = np.full(len(logits), -np.inf)
		if type_mask == 'number' or type_mask == 'float' or type_mask == 'integer':
			for allowed_id in self.__ids_for_numbers:
				masked_logits[allowed_id] = logits[allowed_id]
			return masked_logits
		elif type_mask == 'string':
			for allowed_id in self.__ids_for_strings:
				masked_logits[allowed_id] = logits[allowed_id]

			return masked_logits
		return logits

	def __generate_tokens_in_value_parameters(self, parameters: dict[str, dict[str, str]]) -> None:
		"""generate tokens in state IN_PARAMETERS IN_VALUE"""
		index_item = 1
		for key_param, dict_schema in parameters.items():
			if index_item - 1 == self.__index_key_param:
				break
			index_item += 1
		self.__dynamic_ids = self.encode("\"").tolist()[0]
		self.__dynamic_generated = "\""
		while (self.__dynamic_generated.count("\"") != 2):
			logits = self.get_logits_from_input_ids(self.__ids_current_prompt + self.__dynamic_ids)
			masked_logits = self.__masked_logits_by_type(logits, dict_schema.get('type', ''))
			index_max_logit = np.argmax(masked_logits)
			
			self.__dynamic_generated += self.decode([index_max_logit])
			self.__dynamic_ids.append(int(index_max_logit))
		type_param = dict_schema.get('type', '')
		if type_param == 'number' or type_param == 'integer' or type_param == 'float':
			stripped_number = self.__dynamic_generated.strip().strip('}').strip(",").strip("\"")
			self.__json_result += stripped_number
			self.__ids_current_prompt += self.encode(stripped_number).tolist()[0]

		else:
			stripped_value = self.__dynamic_generated.strip().rstrip(",").rstrip("}")
			self.__json_result += self.__dynamic_generated
			self.__ids_current_prompt += self.__dynamic_ids

	
		if self.__filter_decoder.is_closed_brackets(self.__json_result):
			self.__fsm.set_state(JSONState.IN_END)
		elif index_item - 1 == len(parameters):
			self.__fsm.set_parameters_state(ParameterState.IN_CLOSE)
		elif index_item - 1 < len(parameters):
			self.__json_result += ", "
			self.__ids_current_prompt += self.encode(", ").tolist()[0]
			self.__fsm.set_parameters_state(ParameterState.IN_KEY)
		
		return logits

	def __generate_tokens_in_close_parameters(self) -> None:
		while not self.__filter_decoder.is_closed_brackets(self.__json_result):

			logits = self.get_logits_from_input_ids(self.__ids_current_prompt)
			masked_logits = self.__masked_logits_by_type(logits, 'in_close')
			index_max_logit = np.argmax(masked_logits)
			self.__ids_current_prompt.append(int(index_max_logit))
			self.__json_result += self.decode([int(index_max_logit)])
		self.__fsm.set_state(JSONState.IN_END)

	def __generate_tokens_in_parameters(self) -> None:
		"""generate tokens in state IN_PARAMETERS"""
		function_parameters = self.__get_parameters_function(self.__current_function_name[:-3]) # rm stripping current_function_name
		if self.__fsm.get_parameters_state() == ParameterState.IN_KEY:
			self.__generate_tokens_in_key_parameters(function_parameters)
			self.__index_key_param += 1

		elif self.__fsm.get_parameters_state() == ParameterState.IN_VALUE:
			self.__generate_tokens_in_value_parameters(function_parameters)
		
		elif self.__fsm.get_parameters_state() == ParameterState.IN_CLOSE:
			self.__generate_tokens_in_close_parameters()
		return None

	def __generate(self) -> None:
		"""function generate each json output separate"""
		while not self.__fsm.is_in_end_state():
			print(f">>: {self.__json_result}")
			print(f"parameter state:", self.__fsm.get_parameters_state())
			if self.__fsm.get_state() == JSONState.BEFORE_PROMPT:
				self.__generate_tokens_before_prompt()

			elif self.__fsm.get_state() == JSONState.IN_PROMPT:
				self.__generate_tokens_in_prompt()

			elif self.__fsm.get_state() == JSONState.BEFORE_NAME:
				self.__generate_tokens_before_name()

			elif self.__fsm.get_state() == JSONState.IN_NAME:
				self.__generate_tokens_in_name()

			elif self.__fsm.get_state() == JSONState.BEFORE_PARAMETERS:
				self.__generate_tokens_before_parameters()

			elif self.__fsm.get_state() == JSONState.IN_PARAMETERS:
				self.__generate_tokens_in_parameters()

	def __reinitial_data_for_each_prompt(self, user_prompt) -> None:
		"""" reinitial data for next prompt"""
		self.__fsm.set_state(JSONState.BEFORE_PROMPT)
		self.__fsm.set_static_json(JSONStatic.STR_BEFORE_PROMPT)
		self.__fsm.set_parameters_state(ParameterState.IN_KEY)

		self.__current_prompt = user_prompt
		self.__ids_current_prompt = self.__prefix_ids[:]
		
		self.__index_key_param = 0
		self.__current_function_name = ''
		self.__dynamic_generated = ''
		self.__dynamic_ids = []
		self.__json_result = ''

	def execute(self, data: Any) -> Any:
		""" execute json_generator from pipeline
			for generating json_output
		"""
		self.__prepare_data(data)

		for prompt_schema in self.__prompts:
			self.__reinitial_data_for_each_prompt(
				prompt_schema.prompt['prompt']
			)
			self.__generate()
			print(f"json result @: {self.__json_result}")
			self.__json_results.append(self.__json_result)

		return {
			'json_results': self.__json_results,
			'output_path': data.get('output_path', '')
		}

	def __get_parameters_keys(self, function_name: str) -> dict[str, dict[str, str]]:
		""" get keys of function from functions definition"""
		for function in self.__functions_definition:
			if function.name == function_name:
				keys = [key for key in function.parameters.keys()]
				return keys
		return []

	def __get_parameters_function(self, function_name: str) -> dict[str, dict[str, str]]:
		for function in self.__functions_definition:
			if function.name == function_name:
				return {key: value for key, value in function.parameters.items()}
		return dict()

	def __get_only_available_function(self, dynamic_generated: str) -> str:
		"""get only function start with dynamic generated"""
		for function in self.__function_names:
			if function.startswith(dynamic_generated):
				return function
		return dynamic_generated
