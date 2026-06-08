from typing import Any
from src.processing_stage import ProcessingStage
from transformers import AutoModel
from src.validator import FunctionDefinitionSchema
from src.state import JSONState
from llm_sdk.llm_sdk import Small_LLM_Model
import torch


class JSONGenerator(Small_LLM_Model, ProcessingStage):
	"""JSONGenerator
	"""
	def __init__(self) -> None:
		super().__init__()
		self.__json_results = list()
		self.current_state = JSONState.IN_OPEN_BRACE

	def execute(self, data: Any) -> Any:
		# must be load model first for tokenization
		print("pipeline generator:")
		prompt = data['prompts'][0]		
		clean_prompt = self.build_clean_prompt(data['functions_definition'], prompt)
		input_ids = self.encode(clean_prompt)
		outputs = self._model(input_ids=input_ids)
		logits = outputs.logits[:, -1,:]
		probs = torch.softmax(logits, dim=-1)
		# print("input_ids:\n")
		# print(input_ids)
		# print("outputs:\n")
		# print(outputs)
		print("logits:")
		print(logits)
		print("Probs:")
		print(probs)
		self.get_allowed_tokens(probs) # need contrained decoding with state and field
   
	def build_clean_prompt(
    	self, functions_definition: list[FunctionDefinitionSchema], prompt: str
    ) -> Any:
		available_functions = ''
		for function in functions_definition:
			available_functions += function.model_dump_json() + '\n'

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
		return clean_prompt

	def set_token_hugging_face(self) -> None:
		token_hf = "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"
		model = AutoModel.from_pretrained("private/model", token=token_hf)