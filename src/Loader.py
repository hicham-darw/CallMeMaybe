import json


class JSONLoader:
    """ Class JSONLoader load and stored json files 
    """
    def __init__(self) -> None:
        """constructor of loader load every file json and store the in hash-map

        Args:
            None
        Returns;
            None
        """
        self.__json_data: dict[str, str] = dict()

    #getters
    def get_json_data(self) -> dict[str, str]:
        return self.__json_data

    def read_file(self, filename: str, key: str) -> None:
        with open(filename, "r") as f:
            data = json.load(f)
        self.__json_data[key] = data
    

    