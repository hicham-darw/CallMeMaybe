import argparse


class Parser:
    def __init__(self) -> None:
        pass
    def initial_arguments(self) -> None:
        parser = argparse.ArgumentParser()
        parser.add_argument("--input", default="../../data/input/function_calling_tests.json")
        parser.add_argument("--output", default="../../data/output/function_calling_tests.json")
        parser.add_argument("--model", default="Qwen/Qwen3-0.6B")
        
        args = parser.parse_args()
        self.__input_file = args.input
        self.__output_file = args.output
        self.__model = args.model
        print(args.input)
        print(args.output)
        print(args.model)