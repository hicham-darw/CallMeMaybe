from typing import Any
from pydantic import BaseModel, model_validator
from typing_extensions import Self


class PromptSchema(BaseModel):
    prompt: dict[str, str]

    @model_validator(mode='after')
    def validate_prompt_schema(self) -> Self:
        if len(self.prompt.keys()) != 1:
            raise Exception("Error dictionary must have one key and value")

        prompt_value = self.prompt.get('prompt', None)
        if not isinstance(prompt_value, str):
            raise Exception("Error: value in data['prompt'] must be string\n")
        return self

class FunctionDefinitionSchema(BaseModel):
    name: str
    description: str
    parameters: dict[str, dict[str, Any]]
    returns: dict[str, str]

    @model_validator(mode='after')
    def validate_function_definition_schema(self) -> Self:
        for key_param, val_param in self.parameters.items():
            if len(val_param.keys()) != 1:
                raise Exception("key of each value in parameters accept only 'type' key.")
            for typ in val_param.keys():
                if typ != 'type':
                    raise Exception("key of each value in parameters accept only 'type' key.")
        
        if len(self.returns.keys()) != 1:
            raise Exception("return dictionary contain only 1 key and value")
        for key in self.returns.keys():
            if key != "type":
                raise Exception('key type in returns not found!')
        return self