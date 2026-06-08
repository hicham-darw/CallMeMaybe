from enums import Enum


class JSONState(Enum):
    IDLE = 0
    IN_OPEN_BRACE = 1 # somthing for  first token
    IN_PROMPT_KEY = 2
    IN_DOUBLE_POINTS = 3
    IN_PROMPT_VALUE = 4
    IN_COMMA = 5
    IN_FUNCTION_NAME_KEY = 6
    IN_FUNCTION_NAME_VALUE = 8
    IN_KEY_PARAMETERS = 10
    IN_DOUBLE_POINTS = 11
    IN_PARAMETERS_VALUE = 12
    IN_CLOSE_BRACE = 13

class FiniteStateMachine:

    def __init__(self) -> None:
        # initial_finite_state_machine can run with constrained decoding 
        self.state = JSONState.IDLE

    def change_state(self, event: str) -> None:
        # get state now for constrained decoding     
        if self.state == JSONState.IDLE:
            pass
        elif self.state == JSONState.IN_OPEN_BRACE:
            pass
        elif self.state == JSONState.IN_KEY_PROMPT:
            pass
        elif self.state == JSONState.IN_DOUBLE_POINTS:
            pass
        elif self.state == JSONState.IN_PROMPT_VALUE:
            pass
        elif self.state == JSONState.IN_COMMA:
            pass
        elif self.state == JSONState.IN_KEY_NAME:
            pass
        elif self.state == JSONState.IN_FUNCTION_NAME:
            pass
        elif self.state == JSONState.IN_KEY_PARAMETERS:
            pass
        elif self.state == JSONState.IN_PARAM


# {
#     'prompt': <prompt user>",
#     'name': <function name>,
#     'parameters': <dictionary of parametser>
# } 


state = JSONState.IDLE

print(state)
