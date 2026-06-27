from typing import Any
import json

class JSONWriter:
    
    def __init__(self, output_file: str) -> None:
        self.__output_file: str = output_file
        # self.__content: str = ''

    # for pipeline execution
    def execute(self, data: Any) -> Any:
    	# data must be list of dictionaries or list of strings
        list_of_json = data
        print(f"list of json: {list_of_json}")
        with open(self.__output_file) as file:
            if len(list_of_json) > 1:
                file.write('[\n')
    
            for prompt_result in list_of_json:
                json.dump(prompt_result, file)
    
            if len(list_of_json) > 1:
                file.write(']')

    # getters
    def get_output_file(self) -> str:
        return self.__output_file
