from typing import Any
from src.processing_stage import ProcessingStage
from llm_sdk.llm_sdk import Small_LLM_Model


class JSONGenerator(ProcessingStage):
	"""JSONGenerator
	"""
	def __init__(self) -> None:
		self.__json_results = list()

	# for pipeline execution
	def execute(self, data: Any) -> Any:
		print("GENERATOR:")
		for key, value in data.items():
			print(f"key: {key}")
			print(f"value: {value}")
		sys_prompt = self.build_clean_prompt(data)

	def build_clean_prompt(self, data: Any) -> Any:
		available_functions = ''
		for function in data['functions_definition']:
			available_functions += function.model_dump_json() + '\n'
		# print("available_functions:")
		# print(available_functions)
		print("START PROMPT:::::")
		for prompt in data['prompts']:
			clean_prompt = f"""
			You are a strict function_calling AI assistant.
			your only job is to analyze the user prompt and decide if can use any
			function from THE AVAILABLE FUNCTIONS

			AVAILABLE FUNCTIONS:
				{available_functions}

			USER PROMPT:
				{prompt}
    		
			INSTRUCTIONS:
				choose one matching function
			RULES:
				- output ONLY valid JSON. NO EXTRA TEXT, NO EXPLANATION, NO MARKDOWN.
				- never invent functions that are not listed above
				- extract values from the user prompt accurately
				- ONLY JSON STRUCTURE	
			EXAMPLE OUTPUT:
			   if match one in AVAILABLE FUNCTIONS:
				{{
					"prompt": "<USER PROMPT HERE>",
					"name": "<function_name>",
					"parameters": {{"a": 2.0, "b": 3.0}}
				}}
			   else:
				{{
					"prompt": "<USER PROMPT HERE>",
					"name": null,
					"parameters": null,
				}}
   			"""
			break
		# print(clean_prompt)
		qwen_model = Small_LLM_Model()
		result = qwen_model.generate(clean_prompt, max_new_tokens=64)
		print(result[len(clean_prompt):])
