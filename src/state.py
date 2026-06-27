from enum import Enum, auto


class JSONStatic(str, Enum):
    """Small set of structural JSON tokens controlled by the FSM."""
    STR_BEFORE_PROMPT = '{"prompt": "'
    STR_BEFORE_NAME = '"name": "'
    STR_BEFORE_PARAMETERS = '"parameters": {'


class JSONState(Enum):
    """States for this exact JSON shape.

    Target object:

    {
        "prompt": "<prompt user>",
        "name": "<function name>",
        "parameters": { ... }
    }

    This FSM is useful for constrained decoding because each state tells the
    generator which token types are legal next.
    """
    BEFORE_PROMPT = auto()
    IN_PROMPT = auto()
    
    BEFORE_NAME = auto()
    IN_NAME = auto()

    BEFORE_PARAMETERS = auto()
    IN_PARAMETERS = auto()
    IN_END = auto()


class ParameterState(Enum):

    IN_KEY = auto()
    IN_VALUE = auto()

    IN_COMMA = auto()
    IN_CLOSE = auto()
    
# if __name__ == '__main__':
#     var = JSONStatic.STR_BEFORE_PROMPT
    # print(var.value)
# class JSONField(Enum):
#     """Current semantic field being generated."""

#     NONE = auto()
#     PROMPT = auto()
#     NAME = auto()
#     PARAMETERS = auto()
