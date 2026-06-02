from typing import Any
from sys import exit
import json
from src.validator import PromptSchema, FunctionDefinitionSchema
from src.processing_stage import ProcessingStage

class JSONReader(ProcessingStage):
    """ Class JSONLoader load and stored json files 
    """
    def __init__(self) -> None:
        """constructor of loader load every file json and store the in hash-map

        Args:
            None
        Returns;
            None
        """
        self.__prompts: Any = list()
        self.__functions_definition: Any = list()

    #getters
    def get_prompts(self) -> list[dict[str, str]]:
        return self.__prompts
    
    def get_functions_definition(self) -> list[dict[str, Any]]:
        return self.__functions_definition

    # for pipeline execution
    def execute(self, data: Any) -> Any:
        self.__read_functions_definition(data['functions_definition_path'])
        self.__read_prompts(data['prompts_path'])
        return {
            'functions_definition': self.__functions_definition,
            'prompts': self.__prompts
        }

    # read input files    
    def __read_functions_definition(self, path: str) -> None:
        self.__read_file(path, 'functions_definition')

    def __read_prompts(self, path: str) -> None:
        self.__read_file(path, 'prompts')

    def __read_file(self, filename: str, key: str) -> None:
        with open(filename, "r") as file:
            data = json.load(file)
        if key == 'prompts':
            self.__prompts = data
        else:
            self.__functions_definition = data
        print("data:\n", data)
        print("type data:\n", type(data))

    def validate_prompts_json_file(self) -> None:
        for prompt in self.__prompts:
            try:
                new_model = PromptSchema(prompt=prompt)
            except Exception as e:
                print(e)
                exit(0)
        print('validata prompt successfully')

    def validat_functions_definition_json_file(self) -> None:
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
                # print("func_def model:", new_model)
            except Exception as e:
                print(e)
                exit(0)
        print("validate functions definition structure.")
