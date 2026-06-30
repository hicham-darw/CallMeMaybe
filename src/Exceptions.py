class ParsingError(Exception):
	"""ParsingError class throw exception in specific pipeline"""
	def __init__(self, msg: str) -> None:
		""" Raising parsingError when error threw in parser"""
		super().__init__(msg)

class ReadingError(Exception):
	""" ReadingError class throw Exception in specific pipeline"""
	def __init__(self, msg: str) -> None:
		""" Raising ReadingError when error threw in Reader """
		super().__init__(msg)

