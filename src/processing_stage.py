from typing import Any
from abc import ABC, abstractmethod


class ProcessingStage(ABC):
	"""interface of json stages
	"""
	@abstractmethod
	def execute(self, data: Any) -> Any:
		"""must implement this function on all derived class
		"""
		pass
	

