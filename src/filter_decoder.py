import numpy as np


class FilterDecoder:

    def __init__(self) -> None:
        self.__tokens_before_params: list[int] = []

    def set_list_before_parameters(
        self, list_before_param: str
    ) -> None:
        self.__tokens_before_params: list[int] = list_before_param

    def get_static_tokens_by_state(self) -> list[int]:
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

    def is_closed_bracket(self, json_str: str) -> None:
        json_str = json_str.strip()
        if json_str[0] == '{' and json_str[-1] == '}':
            return True
        return False
