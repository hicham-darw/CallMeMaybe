from typing import Any
import json
from src.ExecutingStage import ExecutingStage
from src.Exceptions import ReadingError


class JSONReader(ExecutingStage):
    """Class JSONReader reads and stores json files."""

    def __init__(self) -> None:
        """Load every JSON file and store the result in memory."""
        self.__prompts: list[dict[str, str]] = list()
        self.__functions_definition: list[dict[str, Any]] = list()

    # getters
    def get_prompts(self) -> list[dict[str, str]]:
        """get all prompts"""
        return self.__prompts

    def get_functions_definition(self) -> list[dict[str, Any]]:
        """get functions definition schemas"""
        return self.__functions_definition

    # for pipeline execution
    def execute(self, data: Any) -> Any:
        """Read JSON files and prepare the next pipeline stage."""
        self.__read_functions_definition(
            data.get('functions_definition_path', '')
        )
        self.__read_prompts(data.get('prompts_path', ''))

        data.update(
            {
                'functions_definition': self.__functions_definition,
                'prompts': self.__prompts,
            }
        )
        return data

    # read input files
    def __read_functions_definition(self, path: str) -> None:
        """Read function_definition from json file."""
        self.__read_file(path, 'functions_definition')

    def __read_prompts(self, path: str) -> None:
        """ read prompts from json file """
        self.__read_file(path, 'prompts')

    def __read_file(self, filename: str, key: str) -> None:
        """ read file with specific parameter path"""
        try:
            with open(filename, "r") as file:
                data = json.load(file)
            if key == 'prompts':
                self.__prompts = data
            else:
                self.__functions_definition = data
        except IsADirectoryError:
            raise ReadingError(f"Error: Cannot read {key} is a directory!")
        except PermissionError:
            raise ReadingError(f"Error: Cannot read {key} not permitted!")
        except FileNotFoundError:
            raise ReadingError(f"Error: Cannot read {key} file not found!")
        except json.JSONDecodeError:
            raise ReadingError(f"Error: Failed to parse json from '{key}'!")
