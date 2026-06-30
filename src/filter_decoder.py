from typing import Any
import numpy as np
from src.state import JSONState


class FilterDecoder:

    def __init__(self) -> None:
        self.__tokens_before_prompt: list[int] = []
        self.__tokens_before_name: list[int] = []
        self.__tokens_before_params: list[int] = []

    # setters
    def set_tokens_before_prompt(self, tokens_before_prompt: list[int]) -> None:
        self.__tokens_before_prompt = tokens_before_prompt

    def set_tokens_before_name(self, tokens_before_name: list[int]) -> None:
        self.__tokens_before_name = tokens_before_name

    def set_tokens_before_parameters(
        self, list_before_param: list[int]
    ) -> None:
        self.__tokens_before_params = list_before_param

    # getters
    def get_tokens_before_prompt(self) -> list[int]:
        return self.__tokens_before_prompt

    def get_tokens_before_name(self) -> list[int]:
        return self.__tokens_before_name

    def get_tokens_before_parameters(self) -> list[int]:
        return self.__tokens_before_params
        
    def is_in_functions(self, dynamic_str: str, function_names: list[str]) -> bool:
        for function_name in function_names:
            if function_name.startswith(dynamic_str):
                return True
        return False
    
    def is_found_only_one_function(self, dynamic_generated: str, function_names: list[str]) -> bool:
        counter = 0
        for function_name in function_names:
            if function_name.startswith(dynamic_generated):
                counter += 1
        if counter == 1:
            return True
        return False

    def is_closed_brackets(self, json_str: str) -> bool:
        stack = []
        for char in json_str:
            if char == '{':
                stack.append('{')
            elif char == '}' and stack:
                stack.pop()
            elif char == '}' and not stack:
                return False
        if stack:
            return False
        return True
                