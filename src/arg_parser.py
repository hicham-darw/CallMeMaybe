import argparse
from sys import exit, stderr

class ArgParser:
    """ ArgParser Class parse for user input argument"""
    
    # getters
    def get_prompts_path(self) -> str:
        """ get prompts path json file"""
        return self.__prompts_path
    
    def get_output_path(self) -> str:
        """ get output path from user """
        return self.__output_path
    

    def get_model(self) -> str:
        """ get model from user  """
        return self.__model
    
    def get_functions_definition_path(self) -> str:
        """ functions schema path for functions definition"""
        return self.__functions_definition_path

    def initial_arguments(self) -> None:
        """ initial full argument with default values"""
        parser = argparse.ArgumentParser()
        parser.add_argument("--functions_definition", default="data/input/functions_definition.json")
        parser.add_argument("--input", default="data/input/function_calling_tests.json")
        parser.add_argument("--output", default="data/output/function_calling_results.json")
        parser.add_argument("--model", default="Qwen/Qwen3-0.6B")
        
        args = parser.parse_args()
	
        self.set_functions_definition_path(args.functions_definition)
        self.set_prompts_path(args.input)
        self.set_output_path(args.output)
        self.set_model(args.model)

    def set_functions_definition_path(self, functions_definition_path: str) -> None:
        self.__functions_definition_path = functions_definition_path

    def set_prompts_path(self, prompts_path: str) -> None:
        self.__prompts_path = prompts_path

    def set_output_path(self, output_path: str) -> None:
        self.__output_path = output_path

    def set_model(self, model: str) -> None:
        self.__model = model
