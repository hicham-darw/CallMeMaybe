from llm_sdk.llm_sdk import Small_LLM_Model
from src.Loader.Loader import JSONLoader
from src.Parser.Parser import Parser

from sys import exit

if __name__ == '__main__':
    
    # parser her 
    
	parser = Parser()
	parser.initial_arguments()
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
