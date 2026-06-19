class FilterDecoder:
    
    def __init__(self) -> None:
        self.__lowercase_alpha = 'abcdefghijklmnopqrstuvwxyz'
        self.__uppercase_alpha = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ'
        self.__special_characters = '{",Ġ:"}'

    def is_closed_brackets(self, generated_json: str) -> True:
        stack = list()
        for char in generated_json:
            if char == '{':
                stack.append(char)
            elif char == '}' and not stack:
                return False
            elif char == '}' and stack[-1] == '{':
                stack.pop()
        if not stack:
            return True
        return False
    
    def is_allowed_token(self, token: str) -> bool:
        for char in token[0]:
            if char not in self.__lowercase_alpha\
                    and char not in self.__uppercase_alpha\
                    and char not in self.__special_characters:
                return False
        return True

    def filter_model_vocabulary(self, vocab: tuple[list, int]) -> None:
        return dict(filter(self.is_allowed_token, vocab))


if __name__ == '__main__':
    
    filter_decoder = FilterDecoder()
    dic = filter_decoder.filter_model_vocabulary(
        {
            "name": "hello",
            "age": 19,
            '\tOK': "nn"
        }.items()
    )
    print(dic)
    print(type(dic))