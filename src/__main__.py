from llm_sdk.llm_sdk import Small_LLM_Model


if __name__ == '__main__':
	small_model = Small_LLM_Model()
	print("small_model:", small_model)
	print("OK!, start CallMeMaybe!\n")

	print("-" * 100)
	while True:
		inp = input("enter prompt: ")

		SYS_PROMPT = f"""
		You are a function-calling AI assistant.
		Your ONLY job is to analyze the user prompt and decide if it matches one of the available functions.

		AVAILABLE FUNCTIONS:
		[
		{{
			"name": "fn_add_numbers",
			"description": "Add two numbers together and return their sum.",
			"parameters": {{
				"a": {{
					"type": "number",
					"description": "The first number"
					}},
				"b": {{
					"type": "number",
					"description": "The second number"
				}}
			}},
			"returns": {{
				"type": "number"
				}}
 		}}
		]
		USER PROMPT: {inp}

		INSTRUCTIONS:
			- If the prompt can call function from available functions:
				REPLY WITH JSON STRUCTURE FROM AVAILABLE FUNCTIONS ONLY
			- If the prompt does NOT match any available function, respond ONLY with:
				{{
					"function_call": null,
					"error": "not match json file"
 				}}

		RULES:
			- Output ONLY valid JSON. No extra text, no explanation, no markdown.
			- Never invent functions that are not listed above.
			- Extract numeric values from the user prompt accurately.
		"""

		generated_text = small_model.generate(
			prompt=SYS_PROMPT,
			)
		print("generated_text:", generated_text)
