from typing import Any
from pydantic import BaseModel, model_validator, Field
from typing_extensions import Self
from src.Enums import FunctionDefinitionKeys


class PromptSchema(BaseModel):
    """PromptSchema model for each prompt."""
    prompt: dict[str, str] = Field(max_length=1, alias="prompt")

    @model_validator(mode='after')
    def validate_prompt_schema(self) -> Self:
        keys = self.prompt.keys()
        if len(keys) != 1:
            raise ValueError("Error dictionary must contain one pair.")
        if list(self.prompt)[0] != 'prompt':
            raise ValueError("Error: key prompt must always prompt.")
        return self


class FunctionDefinitionSchema(BaseModel):
    """Model function definition schema for every function schema."""
    name: str = Field(
        max_length=100,
        min_length=3,
        pattern=r"^[A-Za-z_.]+$",
        alias="name",
    )
    description: str = Field(
        max_length=300,
        min_length=10,
        alias="description",
    )
    parameters: dict[str, dict[str, str]] = Field(
        max_length=10,
        min_length=0,
        alias="parameters",
    )
    returns: dict[str, str] = Field(max_length=1, alias="returns")

    @model_validator(mode='before')
    def validate_raw_data(cls, data: Any) -> Any:
        """Validate raw data before creating the model instance."""
        if not isinstance(data, dict):
            raise ValueError(
                "Error: function_definition schema must be dictionary."
            )

        all_keys = [key for key in data.keys()]
        for key in all_keys:
            if key not in FunctionDefinitionKeys:
                raise ValueError(
                    f'Error: Invalid key {key} must be "type"'
                )
        return data

    @model_validator(mode='after')
    def validate_function_definition_schema(self) -> Self:
        """Validate the function definition after model creation."""
        self.__validate_parameters()
        self.__validate_returns()
        return self

    def __validate_returns(self) -> None:
        "Validate returns has only one item and all keys are 'type'."
        if len(self.returns) > 1:
            raise ValueError(
                "Error: returns dictionary must contain 1 item {key: value}"
            )

        for key in self.returns.keys():
            if key != "type":
                raise KeyError(
                    f'Error: in returns {key} must be "type".'
                )

    def __validate_parameters(self) -> None:
        """Validate parameter keys."""
        for value_dict in self.parameters.values():
            for key in value_dict.keys():
                if key != "type":
                    raise KeyError(
                        'Error: parameter keys must be "type".'
                    )
