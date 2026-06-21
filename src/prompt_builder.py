class PromptBuilder:

    def set_available_functions(self, functions_definition) -> None:
        self.__available_functions: str = ''
        for function in functions_definition:
            self.__available_functions += function.model_dump_json() + '\n'

    def __call__(self, current_prompt: str) -> str:

        clean_prompt = f"""
        You are a strict function_calling AI assistant.

        AVAILABLE FUNCTIONS:
        	{self.__available_functions}

        USER PROMPT:
        	{current_prompt}
    		
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
        		"parameters": {{"a": 2, "b": 3}}
        	}}
<<<<<<< HEAD
=======
           else:
        	{{
        		"prompt": "<USER PROMPT HERE>",
        		"name": null,
        		"parameters": null,
        	}}
        {{"prompt": {current_prompt},"name": "fn_s
>>>>>>> 0a09fd5 (apply constrained decoding on function_names)
        """
        return clean_prompt
