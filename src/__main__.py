from typing import Any
from src.JSONReader import JSONReader
from src.JSONWriter import JSONWriter
from src.ArgParser import ArgParser
from src.Validator import PromptSchema
from src.JSONManager import JSONManager
from src.Exceptions import ParsingError, ReadingError


if __name__ == '__main__':
    
    start = time.time()

    arg_parser = ArgParser()
    arg_parser.initial_arguments()
    json_manager = JSONManager()
    data = {
        'functions_definition_path': arg_parser.get_functions_definition_path(),
        'prompts_path': arg_parser.get_prompts_path(),
        'output_path': arg_parser.get_output_path(),
        'model': arg_parser.get_output_path(),
    }
    json_manager.set_data_input(data)
    try:
        json_manager.generate_json_file()
    except ReadingError as e:
        print(e)
    except ParsingError as e:
        print(e)
    print('FIN:', '#' * 40)
    end = time.time()
    print("time:", (end - start) / 60)
