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
		self.__function_names.append("\"null\"")

	def __prepare_data(self, data: Any) -> None:
		
		self.__functions_definition = data['functions_definition']
		self.__prompts = data['prompts']
		self.load_model_vocabulary()
		self.__prepare_function_names()
		self.__prompt_builder.set_available_functions(self.__functions_definition)

		self.__filter_decoder.set_list_before_parameters(
			self.encode(JSONStatic.STR_BEFORE_PARAMETERS.value).tolist()[0]
		)
		self.__input_ids_as_list: list[int] = self.encode(self.__prompt_builder()).tolist()[0]
		
	def get_only_function_found(self, dynamic_str: str) -> str:
		for function_name in self.__function_names:
			if function_name.startswith(dynamic_str):
				return function_name
		return dynamic_str

	def __generate_function_call(self) -> str:

		json_result = '{"prompt":"' + self.__current_prompt + '","name": "'
		input_ids_as_list: list[int] = self.__input_ids_as_list + self.encode(json_result).tolist()[0]
		dynamic_generated = ''
		dynamic_ids = []
		while not self.__fsm.is_in_end_state():
			print("decode1:", self.decode(input_ids_as_list[len(self.__input_ids_as_list):]))
			if self.__fsm.is_in_state_static_tokens():
				input_ids_as_list += self.__filter_decoder.get_static_tokens_by_state()
				json_result += self.__fsm.get_static_json()
				self.__fsm.goto_next_state()
				self.__fsm.goto_next_static_json()
			else:
				masked_logits = self.get_logits_from_input_ids(input_ids_as_list + dynamic_ids)
				# if self.__fsm.get_state() == JSONState.IN_NAME:
				# 	masked_logits = np.full(len(logits), -np.inf)
				# 	for token, token_id in self.__vocabulary.items():
				# 		if self.__filter_decoder.is_in_functions(
				# 			dynamic_generated + self.decode([token_id]), self.__function_names
				# 		):
				# 			masked_logits[token_id] = logits[token_id]
				# else:
				# 	masked_logits = logits
				index_max_logit = np.argmax(masked_logits)
				# print("predict :", self.decode([index_max_logit]))
				dynamic_generated += self.decode([int(index_max_logit)])
				print("dynamic_generated:", dynamic_generated)
				if self.__fsm.get_state() == JSONState.IN_NAME:
					if self.__filter_decoder.is_found_only_one_function(dynamic_generated, self.__function_names):
						# print("DG1:", dynamic_generated)
						dynamic_generated = self.get_only_function_found(dynamic_generated)

						# print("DG2:", dynamic_generated)
						json_result += dynamic_generated
						input_ids_as_list += self.encode(dynamic_generated).tolist()[0]
						self.__fsm.goto_next_state()
						self.__fsm.goto_next_static_json()
						dynamic_generated = ''
						dynamic_ids = []
					else:
						dynamic_ids += [int(index_max_logit)]
						# dynamic_generated += self.decode([index_max_logit])
					# json_result += self.decode([index_max_logit])
				else:
					input_ids_as_list += [int(index_max_logit)]
					json_result += self.decode([index_max_logit])
			print(f"json_result: {json_result}")
			if self.__fsm.get_state() == JSONState.IN_PARAMETERS and self.__filter_decoder.is_closed_bracket(json_result):
				self.__fsm.goto_next_state()
				self.__fsm.goto_next_static_json()
				break
		self.__fsm.goto_next_state()
		return json_result


	def execute(self, data: Any) -> Any:

		self.__prepare_data(data)

		for prompt_schema in self.__prompts:
			self.__current_prompt = prompt_schema.prompt['prompt']

			json_result = self.__generate_function_call()
			print(f"json_result: |{json_result}|")
			# self.__input_ids_as_list = self.encode(clean_prompt).tolist()[0]
			# dynamic_generated = 'fn_'
			# while not self.__fsm.is_in_end_state():
			# 	print(generated_str)

			# 	if self.__fsm.is_in_static_state():
			# 		generated_str += self.__fsm.get_static_json()
						
			# 		static_input_ids: Any = self.encode(
			# 			self.__fsm.get_static_json()	
			# 		).tolist()[0]
			# 		self.__input_ids_as_list += static_input_ids
			# 		self.__fsm.goto_next_static_json()
			# 		self.__fsm.goto_next_state()
			# 	else:
			# 		# must let model generate tokns 1 by 1
			# 		logits = self.get_logits_from_input_ids(
			# 			self.__input_ids_as_list
			# 		)
			# 		if self.__fsm.get_state() == JSONState.IN_NAME:
			# 			masked_logits = np.full(len(logits), -np.inf)
			# 			for token, token_id in self.__vocabulary.items():
			# 				if self.found_in_function_names(
	        #   					dynamic_generated + self.decode(token_id)
	        #        			):
			# 					masked_logits[token_id] = logits[token_id]
			# 		else:
			# 			masked_logits = logits
		
			# 		index_max_logit = np.argmax(masked_logits)
			# 		#generated_str += self.decode(int(index_max_logit))
			# 		dynamic_generated += self.decode(int(index_max_logit))
			# 		self.__input_ids_as_list.append(index_max_logit)
					
			# 		if self.__fsm.get_state() == JSONState.IN_NAME\
			# 				and self.__filter_decoder.is_only_one_function(dynamic_generated, self.__function_names):
			# 			dynamic_generated = self.__get_only_available_function(dynamic_generated)

			# 		if self.__fsm.get_state() == JSONState.IN_NAME\
			# 				and dynamic_generated.rstrip() in self.__function_names:
			# 			generated_str += dynamic_generated[3:]
			# 			self.__fsm.goto_next_state()
			# 			self.__fsm.goto_next_static_json()
			# 			dynamic_generated = ''
			# 			continue
			# 		else:
			# 			generated_str += self.decode(int(index_max_logit))
			# 			dynamic_generated += self.decode(int(index_max_logit))
			# 	if self.__fsm.get_state() == JSONState.IN_PARAMETERS\
			# 			and self.__filter_decoder.is_closed_brackets(generated_str):
			# 		self.__json_results.append(generated_str)
			# 		print("generated_str:", generated_str)
			# 		self.__fsm.set_state(JSONState.IN_NAME)
			# 		break

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

	def add_static_json(self) -> Any:
		if self.__fsm.get_state() == JSONState.BEFORE_PROMPT:
			return self.encode('{"prompt":').tolist()[0]
		elif self.__fsm.get_state() == JSONState.BEFORE_NAME:
			return self.encode(',"name":').tolist()[0]
		elif self.__fsm.get_state() == JSONState.BEFORE_PARAMETERS:
			return self.encode(',"parameters":').tolist()[0]
		return []
