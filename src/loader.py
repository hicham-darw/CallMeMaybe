import json


class JSONReader:
    """ Class JSONLoader load and stored json files 
    """
    def __init__(self, input_file: str, functions_def: str) -> None:
        """constructor of loader load every file json and store the in hash-map

        Args:
            None
        Returns;
            None
        """
        self.__functions_definition = functions_def
        self.__input_file = input_file
        self.__json_data: dict[str, str] = dict()

    #getters
    def get_json_data(self) -> dict[str, str]:
        return self.__json_data

    # read input files
    def read_prompts(self) -> None:
        self.__read_file(self.__input_file, 'prompts')
    
    def read_functions_definition(self) -> None:
        self.__read_file(self.__functions_definition, 'functions_definition')

    def __read_file(self, filename: str, key: str) -> None:
        with open(filename, "r") as f:
            data = json.load(f)
        self.__json_data[key] = data


    