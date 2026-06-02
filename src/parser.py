import argparse


class ArgParser:
    def __init__(self) -> None:
        self.initial_arguments()
    
    # getters
    def get_input_argument(self) -> str:
        return self.__input_file
    
    def get_output_argument(self) -> str:
        return self.__output_file
    
    def get_model_argument(self) -> str:
        return self.__model
    
    def get_functions_definition(self) -> str:
        return self.__functions_definition

    def initial_arguments(self) -> None:
        parser = argparse.ArgumentParser()
        parser.add_argument("--functions_definition", default="data/input/functions_definition.json")
        parser.add_argument("--input", default="data/input/function_calling_tests.json")
        parser.add_argument("--output", default="data/output/output_file.json")
        parser.add_argument("--model", default="Qwen/Qwen3-0.6B")

        args = parser.parse_args()
        self.__functions_definition = args.functions_definition
        self.__input_file = args.input
        self.__output_file = args.output
        self.__model = args.model
        print(args.input)
        print(args.output)
        print(args.model)
        print("-" * 30)