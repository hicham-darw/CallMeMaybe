from typing import Any
from sys import exit
import json
from src.validator import PromptSchema, FunctionDefinitionSchema


class JSONReader:
    """ Class JSONLoader load and stored json files 
    """
    def __init__(self, input_file: str, functions_def: str) -> None:
        """constructor of loader load every file json and store the in hash-map

        Args:
            None
        Returns;
            None
        """
        self.__functions_definition_path = functions_def
        self.__input_file_path = input_file
        self.__prompts: list[dict[str, str]] = list()
        self.__functions_definition: list[dict[str, Any]] = list()
    
    #getters
    def get_prompts(self) -> list[dict[str, str]]:
        return self.__prompts
    
    def get_functions_definition(self) -> list[dict[str, Any]]:
        return self.__functions_definition
    
    # read input files
    def read_prompts(self) -> None:
        self.__read_file(self.__input_file_path, 'prompts')
    
    def read_functions_definition(self) -> None:
        self.__read_file(self.__functions_definition_path, 'functions_definition')

    def __read_file(self, filename: str, key: str) -> None:
        try:
            with open(filename, "r") as f:
                data = json.load(f)
            if key == 'prompts':
                self.__prompts = data
            elif key == 'functions_definition':
                self.__functions_definition = data
            else:
                raise Exception("invalid keyin read method JSONReader()")
        except Exception as e:
            print(e)
    
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
            try:
                new_model = FunctionDefinitionSchema(
                    name=function_definition.get('name', None),
                    description=function_definition.get('description', None),
                    parameters=function_definition.get('parameters', None),
                    returns=function_definition.get('returns', None)
                )
                print("func_def model:", new_model)
            except Exception as e:
                print(e)
                exit(0)
        print("validate functions definition structure.")
