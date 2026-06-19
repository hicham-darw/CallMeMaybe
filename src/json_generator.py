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

	def get_allowed_tokens(self) -> list[str]:
		if self.__fsm.get_state() == JSONState.IN_START:
			return ['{']
		if self.__fsm.get_state() == JSONState.IN_PROMPT_KEY:
			return ["\"prompt\""]
		if self.__fsm.get_state() == JSONState.IN_PROMPT_COLON:
			return [':']
		if self.__fsm.get_state() == JSONState.IN_PROMPT_VALUE:
			return [self.current_prompt]
		if self.__fsm.get_state()== JSONState.IN_COMMA_AFTER_PROMPT:
			return [","]
		if self.__fsm.get_state() == JSONState.IN_NAME_KEY:
			return ["\"name\""]
		if self.__fsm.get_state() == JSONState.IN_NAME_COLON:
			return [":"]
		if self.__fsm.get_state() == JSONState.IN_NAME_VALUE:
			return self.__function_names
		if self.__fsm.get_state() == JSONState.IN_COMMA_AFTER_NAME:
			return [","]
		if self.__fsm.get_state() == JSONState.IN_PARAMETERS_KEY:
			return["\"parameters\""]
		if self.__fsm.get_state() == JSONState.IN_PARAMETERS_COLON:
			return [":"]
		if self.__fsm.get_state() == JSONState.IN_PARAMETERS_VALUE:
			return []
		else:
			return ['}']
		# if self.__fsm.get_state() == JSONState.IN_CLOSE_BRACE:
		# 	return ['}']
		# else:
		

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
		
		self.load_model_vocabulary()
		self.__swapped_vocabulary = {_id: token for token, _id in self.__vocabulary.items()}
		self.__functions_definition = data['functions_definition']
		self.__prompts = data['prompts']
		self.__prepare_function_names()
		self.__prompt_builder.set_available_functions(self.__functions_definition)

	def execute(self, data: Any) -> Any:

		self.__prepare_data(data)

		# for PromptBuilder object.
		print("pipeline generator:")
		
		for prompt_schema in self.__prompts:
			json_result = ''
			
			self.__current_prompt = prompt_schema.prompt['prompt']
			generated_str = '{"prompt":"' + self.__current_prompt + '","name": "'
			clean_prompt = self.__prompt_builder(self.__current_prompt)
			clean_prompt += generated_str
			self.__input_ids_as_list = self.encode(clean_prompt).tolist()[0]
			dynamic_generated = ''
			while not self.__fsm.is_in_end_state():
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
					generated_str += self.__swapped_vocabulary[int(index_max_logit)]\
						.replace('Ġ', ' ')\
						.replace('Ċ', '\n')
					dynamic_generated += self.__swapped_vocabulary[int(index_max_logit)].replace("Ġ", ' ').replace('Ċ', '\n')
					self.__input_ids_as_list.append(index_max_logit)
					if self.__fsm.get_state() == JSONState.IN_NAME\
							and dynamic_generated.rstrip() in self.__function_names:
						self.__fsm.goto_next_state()
						self.__fsm.goto_next_static_json()
						dynamic_generated = ''

				if self.__fsm.get_state() == JSONState.IN_PARAMETERS\
					and self.__filter_decoder.is_closed_brackets(generated_str):
                                    self.__json_results.append(generated_str)
                                    self.__fsm.set_state(JSONState.IN_NAME)
                                    break

				for json in self.__json_results:
				    print(json)
				print("*" * 60)
		return None

	def add_static_json(self) -> Any:
		if self.__fsm.get_state() == JSONState.BEFORE_PROMPT:
			return self.encode('{"prompt":').tolist()[0]
		elif self.__fsm.get_state() == JSONState.BEFORE_NAME:
			return self.encode(',"name":').tolist()[0]
		elif self.__fsm.get_state() == JSONState.BEFORE_PARAMETERS:
			return self.encode(',"parameters":').tolist()[0]
		return []
