

class JSONWriter:
    
    def __init__(self, output_file: str) -> None:
        self.__output_file: str = output_file
        self.__content: str = ''

    # getters
    def get_output_file(self) -> str:
        return self.__output_file
    
    def write_output(self) -> None:
        with open(self.__output_file, "w") as f:
            f.write(self.__content)