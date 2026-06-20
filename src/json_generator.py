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
# maybe line under me decode here take a time more
				if generated_str + self.decode([v]) == target:
					break
		return masked_logits

	def __prepare_function_names(self) -> None:
		self.__function_names: list[str] = list(
			function.name for function in self.__functions_definition
		)
		self.__function_names.append("\"null\"")

	def __prepare_data(self, data: Any) -> None:
		
		self.load_model_vocabulary()
		self.__swapped_vocabulary = {_id: token for token, _id in self.__vocabulary.items()}
		self.__functions_definition = data['functions_definition']
		self.__prompts = data['prompts']
		self.__prepare_function_names()
		self.__prompt_builder.set_available_functions(self.__functions_definition)

	def execute(self, data: Any) -> Any:

		self.__prepare_data(data)

		for prompt_schema in self.__prompts:
			json_result = ''
			
			self.__current_prompt = prompt_schema.prompt['prompt']
			generated_str = '{"prompt":"' + self.__current_prompt + '","name":"fn'
			clean_prompt = self.__prompt_builder(self.__current_prompt)
			clean_prompt += generated_str
			self.__input_ids_as_list = self.encode(clean_prompt).tolist()[0]
			dynamic_generated = 'fn'
			while not self.__fsm.is_in_end_state():
				print(f"generated: |{generated_str}|")
				logits = self.get_logits_from_input_ids(
					self.__input_ids_as_list
				)
				if self.__fsm.is_in_static_state():
					generated_str += self.__fsm.get_static_json()
						
					static_input_ids: Any = self.encode(
						self.__fsm.get_static_json()	
					).tolist()[0]
					self.__input_ids_as_list += static_input_ids
					self.__fsm.goto_next_static_json()
					self.__fsm.goto_next_state()
				else:
					# must let model generate tokns 1 by 1
					index_max_logit = np.argmax(logits)
					token = self.decode(int(index_max_logit))
					generated_str += token
					dynamic_generated += token
					self.__input_ids_as_list.append(index_max_logit)
					
					if self.__fsm.get_state() == JSONState.IN_NAME\
							and self.__filter_decoder.is_only_one_function(dynamic_generated, self.__function_names):
						dynamic_generated = self.__get_only_available_function(dynamic_generated)

					if self.__fsm.get_state() == JSONState.IN_NAME\
							and dynamic_generated.rstrip() in self.__function_names:
						self.__fsm.goto_next_state()
						self.__fsm.goto_next_static_json()
						dynamic_generated = ''

				if self.__fsm.get_state() == JSONState.IN_PARAMETERS\
						and self.__filter_decoder.is_closed_brackets(generated_str):
					self.__json_results.append(generated_str)
					print("generated_str:", generated_str)
					self.__fsm.set_state(JSONState.IN_NAME)
					break

		return None

	def __get_only_available_function(self, dynamic_generated: str) -> str:
		for function in self.__function_names:
			if dynamic_generated in function:
				print("this is a function not completed but only one found ")
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
