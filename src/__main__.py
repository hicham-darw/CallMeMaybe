from typing import Any
from src.json_reader import JSONReader
from src.json_writer import JSONWriter
from src.arg_parser import ArgParser
from src.validator import PromptSchema
from src.json_manager import JSONManager
from src.Exceptions import ParsingError, ReadingError
import time


if __name__ == '__main__':
    
    # parser her
    start = time.time()
    print(start)

    arg_parser = ArgParser()
    arg_parser.initial_arguments()
    json_manager = JSONManager(
        arg_parser.get_functions_definition_path(),
        arg_parser.get_prompts_path(),
        arg_parser.get_model(),
        arg_parser.get_output_path()
    )
    data = {
        'functions_definition_path': arg_parser.get_functions_definition_path(),
        'prompts_path': arg_parser.get_prompts_path()
    }
    try:
    	for stage in json_manager.get_stages():
        	data = stage.execute(data)
    except ReadingError as e:
        print(e)
    except ParsingError as e:
    	print(e)
#    except Exception as e:
#        print("OKOKOKOKO")
#        print(e)
    print('FIN:', '#' * 40)
    end = time.time()
    print("time:", end - start)
