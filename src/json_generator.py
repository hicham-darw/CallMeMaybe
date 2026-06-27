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
	"""JSONGenerator
	"""
	def __init__(self) -> None:
		super().__init__()
		self.__json_results: list[str] = list()

		self.__prompt_builder = PromptBuilder()
		self.__fsm = FiniteStateMachine()
		self.__filter_decoder = FilterDecoder()

	def __prepare_function_names(self) -> None:
		self.__function_names: list[str] = list(
			function.name + "\", " for function in self.__functions_definition
		)
		self.__ids_function_names_set: set[int] = set()
		for name in self.__function_names:
			function_ids = self.encode(name).tolist()[0]
			for token_id in function_ids:
				self.__ids_function_names_set.add(token_id)
		

	def __prepare_data(self, data: Any) -> None:
		
		self.__functions_definition = data['functions_definition']
		self.__prompts = data['prompts']
		# self.load_model_vocabulary()
		self.__prepare_function_names()
		self.__prompt_builder.set_available_functions(self.__functions_definition)

		self.__tokens_before_prompt = self.encode(JSONStatic.STR_BEFORE_PROMPT.value).tolist()[0]
		self.__tokens_before_name = self.encode(JSONStatic.STR_BEFORE_NAME.value).tolist()[0]
		self.__tokens_before_parameters = self.encode(JSONStatic.STR_BEFORE_PARAMETERS.value).tolist()[0]

		self.__filter_decoder.set_tokens_before_prompt(self.__tokens_before_prompt)
		self.__filter_decoder.set_tokens_before_name(self.__tokens_before_name)
		self.__filter_decoder.set_tokens_before_parameters(self.__tokens_before_parameters)
		
		self.__prefix_ids: list[int] = self.encode(self.__prompt_builder()).tolist()[0]
		
	def get_only_function_found(self, dynamic_str: str) -> str:
		for function_name in self.__function_names:
			if function_name.startswith(dynamic_str):
				return function_name
		return dynamic_str

	# generating
	def __generate_tokens_before_prompt(self) -> None:
		self.__ids_current_prompt += self.__tokens_before_prompt
		self.__json_result += self.__fsm.get_static_json()
		self.__fsm.set_static_json(JSONStatic.STR_BEFORE_NAME)
		self.__fsm.set_state(JSONState.IN_PROMPT)

	def __generate_tokens_in_prompt(self) -> None:
		self.__ids_current_prompt += self.encode(self.__current_prompt + "\", ").tolist()[0]
		self.__json_result += self.__current_prompt + "\", "
		self.__fsm.set_state(JSONState.BEFORE_NAME)

	def __generate_tokens_before_name(self) -> None:
		self.__ids_current_prompt += self.__tokens_before_name
		self.__json_result += self.__fsm.get_static_json()
		self.__fsm.set_static_json(JSONStatic.STR_BEFORE_PARAMETERS)
		self.__fsm.set_state(JSONState.IN_NAME)
	
	def __is_at_least_one_function(self, generated: str) -> bool:
		for function_name in self.__function_names:
			if function_name.startswith(generated):
				return True
		return False
		
	def __generate_tokens_in_name(self) -> None:
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
			self.__json_result += self.__dynamic_generated
			self.__current_function_name = self.__dynamic_generated
			self.__dynamic_ids = self.encode(self.__dynamic_generated).tolist()[0]
			self.__ids_current_prompt += self.__dynamic_ids
			self.__dynamic_ids = []
			self.__dynamic_generated = ''
			self.__fsm.set_state(JSONState.BEFORE_PARAMETERS)

	
	def __generate_tokens_before_parameters(self) -> None:
		self.__ids_current_prompt += self.__tokens_before_parameters
		self.__json_result += self.__fsm.get_static_json()
		self.__fsm.set_state(JSONState.IN_PARAMETERS)
		self.__dynamic_generated = ""
		self.__dynamic_ids = []

	def __generate_tokens_in_key_parameters(self, parameter_key: str) -> None:
		self.__json_result += f'"{parameter_key}": \"'
		self.__ids_current_prompt += self.encode(f"\"{parameter_key}\": \"").tolist()[0]
		self.__fsm.set_parameters_state(ParameterState.IN_VALUE)

	def __generate_tokens_in_value_parameters(self, parameter_key: str) -> None:

		if parameter_key == 'source_string' or parameter_key == "string":
			quoted_string_in_prompt: list[Any] = re.findall(r'["\'][^"]+["\']', self.__current_prompt)
			if quoted_string_in_prompt:
				taller_string: str = max(quoted_string_in_prompt, key=len)
				taller_string = taller_string.strip('"').strip("'")
				self.__ids_current_prompt += self.encode("\"" + taller_string + "\"").tolist()[0]
				self.__ids_current_prompt += self.encode(", ").tolist()[0]
				self.__json_result += taller_string + '", '
				self.__fsm.set_parameters_state(ParameterState.IN_KEY)
				return None
			else:
				logits = self.get_logits_from_input_ids(self.__ids_current_prompt)
				masked_logits = np.full(len(logits), -np.inf)
				allowed_ids = self.encode(self.__current_prompt)
				for each_id in allowed_ids:
					masked_logits[each_id] = logits[each_id]
				index_max_logit = np.argmax(masked_logits)
				self.__ids_current_prompt.append(int(index_max_logit))
				self.__json_result += self.decode([int(index_max_logit)])
				return None
		logits = self.get_logits_from_input_ids(self.__ids_current_prompt)
		index_max_logit = np.argmax(logits)
		self.__ids_current_prompt.append(int(index_max_logit))
		self.__json_result += self.decode([int(index_max_logit)])
		return None

	def __generate_tokens_in_close(self) -> bool:
		logits = self.get_logits_from_input_ids(self.__ids_current_prompt)
		index_max_logit = np.argmax(logits)
		self.__ids_current_prompt.append(int(index_max_logit))
		self.__json_result += self.decode([int(index_max_logit)])
		if self.__fsm.is_closed_brackets(self.__json_result):
			self.__json_results.append(self.__json_result)
			self.__fsm.set_static_json(JSONStatic.STR_BEFORE_PROMPT)
			self.__fsm.set_state(JSONState.BEFORE_PROMPT)
			self.__fsm.set_parameters_state(ParameterState.IN_KEY)
			return True
		return False

	def execute(self, data: Any) -> Any:

		self.__prepare_data(data)

		for prompt_schema in self.__prompts:
	
			self.__current_prompt = prompt_schema.prompt['prompt']
			self.__ids_current_prompt = self.__prefix_ids[:]
			
			index_keys = 0
			self.__current_function_name: str = ''
			self.__dynamic_generated: str = ''
			self.__dynamic_ids: list[int] = []
			self.__json_result: str = ''

			while not self.__fsm.is_in_end_state():
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
					keys_param, values_param = self.get_parameters_keys(self.__current_function_name.strip().rstrip(",").strip("\""))
					if not len(keys_param):
						self.__ids_current_prompt += self.encode("null}").tolist()[0]
						self.__json_result += "null}"
						
						self.__fsm.set_state(JSONState.BEFORE_PROMPT)
						self.__fsm.set_static_json(JSONStatic.STR_BEFORE_PROMPT)
						self.__fsm.set_parameters_state(ParameterState.IN_KEY)
						break
					if self.__fsm.get_parameters_state() == ParameterState.IN_KEY:
						self.__generate_tokens_in_key_parameters(keys_param[index_keys])
						index_keys += 1
					elif self.__fsm.get_parameters_state() == ParameterState.IN_VALUE and (index_keys - 1) < len(keys_param):
						self.__generate_tokens_in_value_parameters(keys_param[index_keys - 1])
						if self.__filter_decoder.is_closed_brackets(self.__json_result)\
        						and index_keys == len(keys_param):
							self.__fsm.set_state(JSONState.BEFORE_PROMPT)
							self.__fsm.set_static_json(JSONStatic.STR_BEFORE_PROMPT)
							self.__fsm.set_parameters_state(ParameterState.IN_KEY)
							break

						elif self.__json_result.rstrip()[-1] == ',' and index_keys < len(keys_param):
							self.__fsm.set_parameters_state(ParameterState.IN_KEY)

				elif self.__fsm.get_parameters_state() == ParameterState.IN_CLOSE:
					if self.__generate_tokens_in_close():
						break
			print("output: ", self.__json_result)
		return None

	def get_parameters_keys(self, function_name: str) -> tuple[list[Any]]:
		for function in self.__functions_definition:
			if function.name == function_name:
				keys = [key for key in function.parameters.keys()]
				values = [value for value in function.parameters.values()]
				return (keys, values)
		return ([],[])

	def found_in_function_names(self, part_name: str) -> bool:
		for function_name in self.__function_names:
			if function_name.startswith(part_name):
				return True
		return False

	def __get_only_available_function(self, dynamic_generated: str) -> str:
		for function in self.__function_names:
			if function.startswith(dynamic_generated):
				return function
		return dynamic_generated

