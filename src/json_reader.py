from typing import Any
from sys import exit
import json
from src.validator import PromptSchema, FunctionDefinitionSchema
from src.processing_stage import ProcessingStage

class JSONReader(ProcessingStage):
    """ Class JSONReader reads and stores json files 
    """
    def __init__(self) -> None:
        """constructor of JSONReader load every file json and store the in hash-map"""
        self.__prompts: Any = list()
        self.__functions_definition: Any = list()

    #getters
    def get_prompts(self) -> list[dict[str, str]]:
        """ get all prompts """
        return self.__prompts
    
    def get_functions_definition(self) -> list[dict[str, Any]]:
        """get functions definition schemas"""
        return self.__functions_definition

    # for pipeline execution
    def execute(self, data: Any) -> Any:
        """ execute pipeline: read json files to parse them in next pipeline"""
        self.__read_functions_definition(data['functions_definition_path'])
        self.__read_prompts(data['prompts_path'])
        return {
            'functions_definition': self.__functions_definition,
            'prompts': self.__prompts
        }

    # read input files    
    def __read_functions_definition(self, path: str) -> None:
        """read function_definition  from json file"""
        self.__read_file(path, 'functions_definition')

    def __read_prompts(self, path: str) -> None:
        """ read prompts from json file """
        self.__read_file(path, 'prompts')

    def __read_file(self, filename: str, key: str) -> None:
        """ read file with specific parameter path"""
        try:
            with open(filename, "r") as file:
                data = json.load(file)
            if key == 'prompts':
                self.__prompts = data
            else:
                self.__functions_definition = data
        except PermissionError as e: #files error handling
            print(e)
        except ValueError as e:
            print(e)
    # only reader read files
    def validate_prompts_json_file(self) -> None:
        """ validate prompts with pydantic """
        for prompt in self.__prompts:
            try:
                new_model = PromptSchema(prompt=prompt)
            except Exception as e:
                print(e)
                exit(0)

    def validate_functions_definition_json_file(self) -> None:
        """ validate  functions definition with pydnatic """
        for function_definition in self.__functions_definition:
            if not isinstance(function_definition, dict):
                raise Exception("Error function definition must be dictionary.")
            try:
                new_model = FunctionDefinitionSchema(
                    name=function_definition.get('name', None),
                    description=function_definition.get('description', None),
                    parameters=function_definition.get('parameters', None),
                    returns=function_definition.get('returns', None)
                )
            except Exception as e: # error handling pydantic
                print(e)
                exit(0)
