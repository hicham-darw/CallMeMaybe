from src.processing_stage import ProcessingStage
from src.json_reader import JSONReader
from src.json_parser import JSONParser
from src.json_generator import JSONGenerator
from src.json_writer import JSONWriter


class JSONManager:
	"""JSONManager class manage pipeline to generate structured json
	"""
	def __init__(self, functions_definition_path: str, prompts_path: str, model: str, output_path: str) -> None:
		self.__functions_definition_path = functions_definition_path
		self.__prompts_path = prompts_path
		self.__model = model
		self.__output_path = output_path
		self.__stages: list[ProcessingStage]= [
      		JSONReader(),
        	JSONParser(),
         	JSONGenerator(),
          	# JSONWriter()
        ]
	
	def get_stages(self):
		return self.__stages

	def add_stage(self, stage: ProcessingStage) -> None:
		self.__stages.append(stage)

	def generate_json_file(self):
		for stage in self.__stages:
			data = stage.execute()
