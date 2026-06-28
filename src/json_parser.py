from typing import Any
from src.processing_stage import ProcessingStage
from src.validator import PromptSchema, FunctionDefinitionSchema
from pathlib import Path


class JSONParser(ProcessingStage):
	"""JSONParser parse input user and what read from arguments
	"""
	def __init__(self, output_file: str) -> None:
		self.__prompts: list[PromptSchema] = list()
		self.__functions_definition: list[FunctionDefinitionSchema] = list()
		self.__output_path: str = output_file

	# for pipeline execution
	def execute(self, data: Any) -> Any:
		"""execute pipeline to parse structure and pass to execute generator"""
		self.__create_folders_path()     
		self.__is_valid_type(data)
		self.__is_structure_prompts_valid(data['prompts'])
		self.__is_structure_functions_definition_valid(data['functions_definition'])
		return {
			'prompts': self.__prompts,
			'functions_definition': self.__functions_definition,
			'output_path': self.__path
		}

	def __create_folders_path(self) -> None:
		""" create folders of path if exist go to subdirs"""
		self.__path = Path(self.__output_path)
		dirs = self.__path.parent
		dirs.mkdir(parents=True, exist_ok=True)
		self.__path.touch()
    
	def __create_json_file(self) -> None:
	    """create json file for storing output here"""
	    filename = self.__path.name
	    filename.touch()

	def __is_structure_prompts_valid(self, data: Any) -> None:
		""" is structure prompts is valid before execute next pipeline"""
		if isinstance(data, dict):
			prompt_model = PromptSchema(prompt=data)
			self.__prompts.append(prompt_model)
			return None
		for prompt in data:
			prompt_model = PromptSchema(prompt=prompt)
			self.__prompts.append(prompt_model)

	def __is_structure_functions_definition_valid(self, data: Any) -> None:
		""" is structure functions definition is valid before execute next pipeline"""
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
		""" is valid type of functions definition and in prompts"""
		if not isinstance(data['functions_definition'], dict)\
  				and not isinstance(data['functions_definition'], list):
			raise Exception("Functions_definition content must be dictionary or list.")
		if not isinstance(data['prompts'], dict)\
				and not isinstance(data['prompts'], list):
			raise Exception("prompts_content must be dictionary or list")
