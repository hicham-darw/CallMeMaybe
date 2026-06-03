from typing import Any
from src.processing_stage import ProcessingStage
from src.validator import PromptSchema, FunctionDefinitionSchema

class JSONParser(ProcessingStage):
	"""
	"""
	def __init__(self) -> None:
		self.__prompts: list[PromptSchema] = list()
		self.__functions_definition: list[FunctionDefinitionSchema] = list()

	# for pipeline execution
	def execute(self, data: Any) -> Any:
		print("PARSER:")
		for k, v in data.items():
			print(f"key: {k}")
			print(f"value: {v}")
		print("#" * 30)
		self.__is_valid_type(data)
		self.__is_structure_prompts_valid(data['prompts'])
		self.__is_structure_functions_definition_valid(data['functions_definition'])
		return {
			'prompts': self.__prompts,
			'functions_definition': self.__functions_definition
		}

	def __is_structure_prompts_valid(self, data: Any) -> None:
		if isinstance(data, dict):
			prompt_model = PromptSchema(prompt=data)
			self.__prompts.append(prompt_model)
			return None
		for prompt in data:
			prompt_model = PromptSchema(prompt=prompt)
			self.__prompts.append(prompt_model)

	def __is_structure_functions_definition_valid(self, data: Any) -> None:
		if isinstance(data, dict):
			function_definition_model = FunctionDefinitionSchema(
				name=data.get('name', None),
				description=data.get('description', None),
				parameters=data.get('parameters', None),
				returns=data.get('returns', None)
			)
			self.__functions_definition.append(function_definition_model)
			return None
		for function_def in data:
			function_definition_model = FunctionDefinitionSchema(
				name=function_def.get('name', None),
				description=function_def.get('description', None),
				parameters=function_def.get('parameters', None),
				returns=function_def.get('returns', None)
			)
			self.__functions_definition.append(function_definition_model)

   
  
	def __is_valid_type(self, data: Any) -> None:
		if not isinstance(data['functions_definition'], dict)\
  				and not isinstance(data['functions_definition'], list):
			raise Exception("Functions_definition content must be dictionary or list.")
		if not isinstance(data['prompts'], dict)\
				and not isinstance(data['prompts'], list):
			raise Exception("prompts_content must be dictionary or list")
