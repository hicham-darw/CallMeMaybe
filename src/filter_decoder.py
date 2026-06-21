import numpy as np


class FilterDecoder:
    
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
    
    def is_only_one_function(self, part_name: str, function_names: list[str]) -> bool:
        
        counter = 0
        for function in function_names:
            if function.startswith(part_name):
                counter += 1

        if counter == 1:
            return True
        return False


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