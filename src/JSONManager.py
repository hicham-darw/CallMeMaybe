from src.ExecutingStage import ExecutingStage
from src.JSONReader import JSONReader
from src.JSONParser import JSONParser
from src.JSONGenerator import JSONGenerator
from src.JSONWriter import JSONWriter
from typing import Any


class JSONManager:
	"""JSONManager class manage pipeline to generate structured json
	"""
	def __init__(self) -> None:
		self.__data: dict[str, Any] = dict()
		self.__stages: list[ExecutingStage]= [
		JSONReader(),
        	JSONParser(),
         	JSONGenerator(),
          	JSONWriter()
        ]

	def set_data_input(self, data: dict[str, Any]) -> None:
		self.__data = data
  
	def get_stages(self):
		"""get all stages pipeline"""
		return self.__stages

	def add_stage(self, stage: ExecutingStage) -> None:
		"""additional function for adding stage just for automation"""
		self.__stages.append(stage)

	def generate_json_file(self):
		"""generate json file run full pipeline"""
		for stage in self.__stages:
			self.__data = stage.execute(self.__data)


# if __name__ == '__main__':
# 	manager = JSONManager()
# 	print(manager.create_folder_output())
