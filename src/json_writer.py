from typing import Any
from pathlib import Path
import json


class JSONWriter:
    
    # for pipeline execution
    def execute(self, data: Any) -> Any:
    	# data must be list of dictionaries or list of strings
        
        self.create_path_to_json()

        full_path = Path(self.__output_file)
        print(full_path)
        directories = full_path.parent
        file_name = full_path.name
        directories.mkdir(parents=True)
        full_path.touch()
        list_of_json = data
        print(f"list of json: {list_of_json}")
        #with open(self.__output_file) as file:
        #   if len(list_of_json) > 1:
        #        file.write('[\n')
    
      #      for prompt_result in list_of_json:
      #          json.dump(prompt_result, file)
    #
     #       if len(list_of_json) > 1:
      #          file.write(']')

    # getters
    def get_output_file(self) -> str:
        return self.__output_file


if __name__ == '__main__':
    writer = JSONWriter("dasa/output/file.txt")
    writer.execute([])
