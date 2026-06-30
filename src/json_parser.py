from typing import Any
from src.processing_stage import ProcessingStage
from src.validator import PromptSchema, FunctionDefinitionSchema
from src.Exceptions import ParsingError
from pathlib import Path
from pydantic import ValidationError
import sys


class JSONParser(ProcessingStage):
	"""JSONParser parse input user and what read from arguments
	"""
	def __init__(self, output_file: str) -> None:
		self.__prompts: list[PromptSchema] = list()
		self.__functions_definition: list[FunctionDefinitionSchema] = list()
		self.__output_path: str = output_file

	def __create_prompts_models(self, prompts: list[dict[str, Any]]) -> None:
		""" create prompts model and validate it by pydantic for specific schema"""
		if isinstance(prompts, list):
			self.__prompts = [PromptSchema(prompt=prompt) for prompt in prompts]
		elif isinstance(prompts, dict):
			self.__prompts.append(PromptSchema(prompt=prompts))
		else:
			raise ValueError("Error: prompts data must be list o dict.")
  
	def __create_functions_definition_models(self, functions_definition: list[dict[str, Any]]) -> None:
		"""create functions definition models  and validate data for specific schema"""
		if isinstance(functions_definition, list):
			for func_def in functions_definition:
				self.__functions_definition.append(
					FunctionDefinitionSchema(
						**func_def
            		)
				)
		elif isinstance(functions_definition, dict):
			self.__functions_definition.append(
				FunctionDefinitionSchema(
					**functions_definition
       	     			)
			)

		else:
			raise ValueError("Error: FunctionsDefinition schema must list or dictionary")

	def execute(self, data: Any) -> Any:
		"""execute pipeline to parse structure and pass to execute generator"""
		try:
			self.__create_prompts_models(data['prompts'])
			self.__create_functions_definition_models(data['functions_definition'])
			self.__create_folders_path() 
		except (ValidationError) as e:
			raise ParsingError(e.errors()[0].get("msg", ''))
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



if __name__ == "__main__":
    parser = JSONParser("data/output/function_calling_results.json")
    parser.execute({
		'functions_definition': {
    		"name": "fn_get_square_root",
	    	"description": "Calculate the square root of a number.",
    		"parameters": {
    			"a": {
    	    		"type": "number"
    			}
    		},
    		"returns": {
    	  		"type": "number"
    		}
  		},
		'prompts': {
			'prompt': "hello"
		}
	})
