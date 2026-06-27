import argparse


class ArgParser:

    
    # getters
    def get_prompts_path(self) -> str:
        return self.__prompts_path
    
    def get_output_path(self) -> str:
        return self.__output_path
    

    def get_model(self) -> str:
        return self.__model
    
    def get_functions_definition_path(self) -> str:
        return self.__functions_definition_path

    def initial_arguments(self) -> None:
        parser = argparse.ArgumentParser()
        parser.add_argument("--functions_definition", default="data/input/functions_definition.json")
        parser.add_argument("--input", default="data/input/function_calling_tests.json")
        parser.add_argument("--output", default="data/output/function_calling_results.json")
        parser.add_argument("--model", default="Qwen/Qwen3-0.6B")

        args = parser.parse_args()
        self.__functions_definition_path = args.functions_definition
        self.__prompts_path = args.input
        self.__output_path = args.output
        self.__model = args.model
