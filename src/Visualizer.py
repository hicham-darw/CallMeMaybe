from time import sleep


class Visualizer:
    """Visualizer print generated text by SDK"""
    @classmethod
    def print_next(cls, generated: str) -> None:
        for char in generated:
            print(char, end='', flush=True)
            sleep(0.02)
        print()
