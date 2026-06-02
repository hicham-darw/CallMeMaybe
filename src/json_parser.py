from typing import Any
from src.processing_stage import ProcessingStage


class JSONParser(ProcessingStage):
	"""
	"""
	
	# for pipeline execution
	def execute(self, data: Any) -> Any:
		# must be parse content data
		self.__is_valid_type(data)
		self.__is_structure_prompts_valid()
		self.__is_structure_functions_definition_valid()
	
	def __is_structure_prompts_valid(self):
		pass
	
	def __is_structure_functions_definition_valid(self):
		pass
  
	def __is_valid_type(self, data: Any) -> None:
		print("SSSSSSS:", type(data['functions_definition']))
		if not isinstance(data['functions_definition'], dict)\
  				and not isinstance(data['functions_definition'], list):
			raise Exception("Functions_definition content must be dictionary or list.")
		if not isinstance(data['prompts'], dict)\
				and not isinstance(data['prompts'], list):
			raise Exception("prompts_content must be dictionary or list")
