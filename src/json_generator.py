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
from src.state import JSONStatic
import json

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

	def load_model_vocabulary(self) -> None:
		path_to_vocabulary = self.get_path_to_vocab_file()
		with open(path_to_vocabulary) as file:
			self.__vocabulary = json.load(file)

	def get_allowed_logits(self, logits, generated_str: str, target: list[str]) -> list[int]:

		if not target:
			return logits
		masked_logits = np.full_like(logits, -np.inf)
		for item in target:
			for k, v in self.__vocabulary.items():
				if item.startswith((generated_str + self.decode([v]))):
					masked_logits[v] = logits[v]
# maybe line under me
				if generated_str + self.decode([v]) == target:
					break
		return masked_logits

	def __prepare_function_names(self) -> None:
		self.__function_names: list[str] = list(
			function.name for function in self.__functions_definition
		)
		self.__function_names_set = set(self.__function_names)
		self.__function_names.append("\"null\"")
		self.__function_names_set.add("\"null\"")

	def __prepare_data(self, data: Any) -> None:
		
		self.__functions_definition = data['functions_definition']
		self.__prompts = data['prompts']
		self.load_model_vocabulary()
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


	def execute(self, data: Any) -> Any:

		self.__prepare_data(data)

		for prompt_schema in self.__prompts:
	
			self.__current_prompt = prompt_schema.prompt['prompt']
			ids_current_prompt = self.__prefix_ids[:]
			ids_current_prompt += self.__tokens_before_prompt
			ids_current_prompt += self.encode(self.__current_prompt).tolist()[0]
			ids_current_prompt += self.__tokens_before_name
			
			dynamic_generated: str = ''
			dynamic_ids: list[int] = []
			while not self.__fsm.is_in_end_state():
				#print(self.decode(ids_current_prompt[len(self.__prefix_ids):]))
				if self.__fsm.get_state() == JSONState.BEFORE_PARAMETERS:
					ids_current_prompt += self.__tokens_before_parameters
					self.__fsm.goto_next_static_json()
					self.__fsm.goto_next_state()
				else:
					# must let model generate tokns 1 by 1
					masked_logits = self.get_logits_from_input_ids(
						ids_current_prompt + dynamic_ids
					)
		
					index_max_logit = np.argmax(masked_logits)
					next_token_id = int(index_max_logit)
					next_token = str(self.decode([next_token_id]))
					dynamic_generated = f"{dynamic_generated}{next_token}"
					dynamic_ids.append(next_token_id)
					
					if self.__fsm.get_state() == JSONState.IN_NAME\
							and self.__filter_decoder.is_found_only_one_function(dynamic_generated, self.__function_names):
						dynamic_generated = self.__get_only_available_function(dynamic_generated)
						dynamic_ids = self.encode(dynamic_generated).tolist()[0]

					if self.__fsm.get_state() == JSONState.IN_NAME\
							and dynamic_generated.rstrip() in self.__function_names_set:
						ids_current_prompt += dynamic_ids
						self.__fsm.goto_next_state()
						self.__fsm.goto_next_static_json()
						dynamic_generated = ''
						dynamic_ids = []
						continue

					if self.__fsm.get_state() == JSONState.IN_PARAMETERS:
						ids_current_prompt += [int(index_max_logit)]

				if self.__fsm.get_state() == JSONState.IN_PARAMETERS\
						and self.__filter_decoder.is_closed_brackets(dynamic_generated):
					self.__json_results.append(self.decode(ids_current_prompt[len(self.__prefix_ids):]))
					print("json_results:", self.__json_results[-1])
					self.__fsm.set_state(JSONState.IN_NAME)
					break

		return None

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

