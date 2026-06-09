from enum import Enum


class JSONState(Enum):
    IN_START = 0
    IN_OPEN_BRACE = 1 # somthing for  first token
    IN_KEY = 2
    IN_DOUBLE_POINTS = 3
    IN_VALUE = 4
    IN_COMMA = 5
    IN_CLOSE_BRACE = 6
    IN_END = 7

class JSONField(Enum):
    START = 0
    PROMPT = 1
    NAME = 2
    PARAMETERS = 3
    END = 4


ALLOWED = {
    'IN_OPEN_BRACE': ['{'],
    'IN_KEY': ['prompt', 'name', 'parameters'],
    'IN_DOUBLE_POINTS': [':'],
    'IN_VALUE': [],
    'IN_COMMA': [','],
    'IN_CLOSE_BRACE': ['}'],
    'IN_END': [],
}

class FiniteStateMachine:

    def __init__(self) -> None:
        # initial_finite_state_machine can run with constrained decoding 
        self.state = JSONState.IDLE

    def change_state(self, event: str) -> None:
        # get state now for constrained decoding     
        if self.state == JSONState.IN_START:
            pass
        elif self.state == JSONState.IN_OPEN_BRACE:
            pass
        elif self.state == JSONState.IN_KEY:
            pass
        elif self.state == JSONState.IN_DOUBLE_POINTS:
            pass
        elif self.state == JSONState.IN_VALUE:
            pass
        elif self.state == JSONState.IN_COMMA:
            pass
        elif self.state == JSONState.IN_CLOSE_BRACE:
            pass
        else:
            pass


# {
#     'prompt': <prompt user>",
#     'name': <function name>,
#     'parameters': <dictionary of parametser>
# } 

