from typing import Any
from pathlib import Path
import json


class JSONWriter:
    """ JSONWriter class write a json to a specific file from user"""
    # for pipeline execution
    def execute(self, data: Any) -> Any:
      """execute pipeline serialize data to specific output path"""
    	# data must be list of dictionaries or list of strings
      list_of_json = data.get('json_results', [])
      with open(data['output_path'], "w+") as file:
        if len(list_of_json) > 1:
          file.write("[\n")
        for index, output_json in enumerate(list_of_json):
          try:
            data_object = json.loads(output_json)
            json.dump(data_object, file, indent=4)
            if index < len(list_of_json) - 1:
              file.write(",\n")
            else:
              file.write("\n")
          except Exception:
            print("data Error\n")
        if len(list_of_json) > 1:
          file.write("]")

# if __name__ == '__main__':
#     writer = JSONWriter("dasa/output/file.txt")
#     writer.execute([])
