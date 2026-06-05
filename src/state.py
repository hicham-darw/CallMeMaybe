from enums import Enum


class JSONState:
    OPNE = "{" # somthing for  first token
    IN_QUOTES = "\""
    IN_KEY = "alphabetical"
    IN_DOUBLE_POINT = ":"
    IN_VALUE = 'alphabetical or dictionary' # maybe can constrined also
    IN_VERGULE = ','
    IN_SPACE = ' '


class FiniteStateMachine:
    
    def __init__(self) -> None:
        # initial_finite_state_machine can run with constrained decoding 

    def what_is_state(self) -> None:
        # get state now for constrained decoding     
        pass

# {
#     'prompt': <prompt user>",
#     'name': <function name>,
#     'parameters': <dictionary of parametser>
# } 
