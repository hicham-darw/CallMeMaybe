# from llm_sdk.llm_sdk import Small_LLM_Model
from typing import Any
from src.json_reader import JSONReader
from src.json_writer import JSONWriter
from src.arg_parser import ArgParser
from src.validator import PromptSchema
from src.json_manager import JSONManager

from sys import exit

import torch


if __name__ == '__main__':
    
    # parser her 
	torch.set_num_threads(4)    
	arg_parser = ArgParser()
	arg_parser.initial_arguments()
	json_manager = JSONManager(
		arg_parser.get_functions_definition_path(),
		arg_parser.get_prompts_path(),
		arg_parser.get_model(),
		arg_parser.get_output_path()
	)
	data = {
		'functions_definition_path': arg_parser.get_functions_definition_path(),
		'prompts_path': arg_parser.get_prompts_path()		
	}
	for stage in json_manager.get_stages():
		data = stage.execute(data)
	print('FIN:', '#' * 40)
 
	exit(0)
	# reader = JSONReader(
    #  			parser.get_input_argument(),
	# 			parser.get_functions_definition(),
    #         )
	# writer = JSONWriter(
	# 	parser.get_output_argument()
	# )
	reader.read_prompts()
	reader.read_functions_definition()

	reader.validate_prompts_json_file()
	reader.validat_functions_definition_json_file()
	# parser must has attributes JSONReader
	
	exit(0)
	#loading files
# 	json_loader = JSONLoader()
    
# 	json_loader.read_file('data/input/functions_definition.json', "functions_definition")
# 	json_loader.read_file('data/input/function_calling_tests.json', "prompts")
# 	json_data = json_loader.get_json_data()

# 	for key, val in json_data.items():
# 		print(f"key : {key}")
# 		print(f"value:")
# 		for func in val:
# 			print(f"- {func}")
# 		print("-" * 60)

# 	small_model = Small_LLM_Model()
# 	print("small_model:", small_model)

# 	print("#####" * 100)
# 	# while True:
# 	inp = input("enter prompt: ")

# 	SYS_PROMPT = f"""
# 	You are a function-calling AI assistant.
# 	YOUR ONLY JJOB IS TO ANALYZE THE USER PROMPT AND DECIDE IF IT MATCHES ONE OF THE AVAILABLE FUNCTIONS:.
	
#  	AVAILABLE FUNCTIONS:
# 		{json_data.get('functions_defiition', None)}

# 	USER PROMPT:
#  		{inp}
	
#  	INSTRUCTIONS:
# 		- If the prompt can call function from available functions:
# 			REPLY WITH JSON STRUCTURE FROM AVAILABLE FUNCTIONS ONLY
# 		- If the prompt does NOT match any available function, respond ONLY with:
# 			{{
# 				"function_call": null,
# 				"error": "not match json file"
#  			}}

# 	RULES:
# 		- Output ONLY valid JSON. No extra text, no explanation, no markdown.
# 		- Never invent functions that are not listed above.
# 		- Extract numeric values from the user prompt accurately.
# 	ANSWER:
# 		FUNCTION FROM AVAILABLE FUNCTIONS OTHERWISE ERROR JSON
# """

# 	generated_text = small_model.generate(
# 		prompt=SYS_PROMPT,
# 		)
# 	print("generated_text:", generated_text)
