class FilterDecoder:

    def __init__(self) -> None:
        """constructor Filter Decoder for constrained decoding"""
        self.__tokens_before_prompt: list[int] = []
        self.__tokens_before_name: list[int] = []
        self.__tokens_before_params: list[int] = []

    # setters
    def set_tokens_before_prompt(
        self, tokens_before_prompt: list[int]
    ) -> None:
        """set ids stringbefore_prompt"""
        self.__tokens_before_prompt = tokens_before_prompt

    def set_tokens_before_name(
        self, tokens_before_name: list[int]
    ) -> None:
        """set ids string before name"""
        self.__tokens_before_name = tokens_before_name

    def set_tokens_before_parameters(
        self, list_before_param: list[int]
    ) -> None:
        """set ids of string before parameters"""
        self.__tokens_before_params = list_before_param

    # getters
    def get_tokens_before_prompt(self) -> list[int]:
        """get ids of string before prompt"""
        return self.__tokens_before_prompt

    def get_tokens_before_name(self) -> list[int]:
        """get ids of string before name"""
        return self.__tokens_before_name

    def get_tokens_before_parameters(self) -> list[int]:
        """get ids of string before parameters"""
        return self.__tokens_before_params

    def is_in_functions(
        self, dynamic_str: str, function_names: list[str]
    ) -> bool:
        """check if dynamic string in a function_names"""
        for function_name in function_names:
            if function_name.startswith(dynamic_str):
                return True
        return False

    def is_found_only_one_function(
        self, dynamic_generated: str, function_names: list[str]
    ) -> bool:
        """check if dynamic string starts in one function
              not more in function names
        """
        counter = 0
        for function_name in function_names:
            if function_name.startswith(dynamic_generated):
                counter += 1
        if counter == 1:
            return True
        return False

    def is_closed_brackets(self, json_str: str) -> bool:
        """check if is closed brackets in generated json"""
        counter = 0
        stack = []
        index_in_param = json_str.rfind('"parameters": ')
        if index_in_param < 0:
            return False
        json_str = json_str[index_in_param:]
        for char in json_str:
            if char == '{':
                counter += 1
            elif char == '}':
                counter -= 1
        if counter == -1:
            return True
        return False
