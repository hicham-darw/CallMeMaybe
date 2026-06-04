from typing import Any
import json

class JSONWriter:
    
    def __init__(self, output_file: str) -> None:
        self.__output_file: str = output_file
        # self.__content: str = ''

    # for pipeline execution
    def execute(self, data: Any) -> Any:
    	# data must be list of dictionaries or list of strings
        
        with open(self.__output_file) as file:
            file.write('[\n')
            for prompt_result in data:
                json.dump(prompt_result, file)
            file.write(']')

    # getters
    def get_output_file(self) -> str:
        return self.__output_file
    
    def write_output(self) -> None:
        with open(self.__output_file, "w") as f:
            f.write(self.__content)
