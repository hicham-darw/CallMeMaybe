class FilterDecoder:
    
    def __init__(self) -> None:
        self.__index_of_static_json = 0

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
    
    
if __name__ == '__main__':
    
    print(FilterDecoder.is_closed_brackets("{dhidhodhhdd}}"))
    print(FilterDecoder.is_closed_brackets("}"))
    print(FilterDecoder.is_closed_brackets("{{{}}}"))
    print(FilterDecoder.is_closed_brackets("{{{{{"))
    print(FilterDecoder.is_closed_brackets("}}}}}}}}}}"))