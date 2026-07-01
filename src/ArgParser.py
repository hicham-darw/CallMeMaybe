import argparse


class ArgParser:
    """ ArgParser Class parse user input"""

    # getters
    def get_prompts_path(self) -> str:
        """ get prompts path json file"""
        return self.__prompts_path

    def get_output_path(self) -> str:
        """ get output path from user """
        return self.__output_path

    def get_functions_definition_path(self) -> str:
        """ functions schema path for functions definition"""
        return self.__functions_definition_path

    def initial_arguments(self) -> None:
        """ initial full argument with default values"""
        parser = argparse.ArgumentParser()
        parser.add_argument(
            "--functions_definition",
            default="data/input/functions_definition.json"
        )
        parser.add_argument(
            "--input", default="data/input/function_calling_tests.json"
        )
        parser.add_argument(
            "--output", default="data/output/function_calling_results.json"
        )

        args = parser.parse_args()

        self.set_functions_definition_path(args.functions_definition)
        self.set_prompts_path(args.input)
        self.set_output_path(args.output)

    def set_functions_definition_path(
        self, functions_definition_path: str
    ) -> None:
        """set functions definition own  path"""
        self.__functions_definition_path = functions_definition_path

    def set_prompts_path(self, prompts_path: str) -> None:
        """set prompts own path"""
        self.__prompts_path = prompts_path

    def set_output_path(self, output_path: str) -> None:
        """set own output path"""
        self.__output_path = output_path
