from enum import Enum, auto


class JSONToken(str, Enum):
    """Small set of structural JSON tokens controlled by the FSM."""

    OPEN_BRACE = "{"
    CLOSE_BRACE = "}"
    COLON = ":"
    COMMA = ","
    PROMPT_KEY = '"prompt"'
    NAME_KEY = '"name"'
    PARAMETERS_KEY = '"parameters"'


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

    IN_START = auto()

    EXPECT_PROMPT_KEY = auto()
    EXPECT_PROMPT_COLON = auto()
    EXPECT_PROMPT_VALUE = auto()

    EXPECT_COMMA_AFTER_PROMPT = auto()

    EXPECT_NAME_KEY = auto()
    EXPECT_NAME_COLON = auto()
    EXPECT_NAME_VALUE = auto()

    EXPECT_COMMA_AFTER_NAME = auto()

    EXPECT_PARAMETERS_KEY = auto()
    EXPECT_PARAMETERS_COLON = auto()
    EXPECT_PARAMETERS_OBJECT = auto()

    EXPECT_ROOT_CLOSE = auto()
    IN_END = auto()


class JSONField(Enum):
    """Current semantic field being generated."""

    NONE = auto()
    PROMPT = auto()
    NAME = auto()
    PARAMETERS = auto()


class FiniteStateMachine:
    """Finite state machine for constrained JSON function-call generation.

    This class does not generate text by itself. Instead, it answers:

    1. What tokens are allowed right now?
    2. If the model emits one token, what is the next state?

    In a real constrained decoder, you would:

    - call allowed_tokens()
    - convert those strings to tokenizer IDs
    - mask all other logits
    - sample or choose the next token
    - call advance(next_token)
    """

    def __init__(self) -> None:
        self.state = JSONState.START
        self.field = JSONField.NONE

    def is_finished(self) -> bool:
        """Return True when the full JSON object is complete."""

        return self.state == JSONState.END

    def allowed_tokens(self) -> list[str]:
        """Return legal next tokens for the current state.

        Notes:
        - Fixed JSON syntax returns exact strings like "{" or ",".
        - Dynamic values return placeholders such as "<prompt_string>".
        - Your real decoder must replace placeholders with constrained logic
          for strings, null, numbers, and nested parameter objects.
        """

        if self.state == JSONState.START:
            return [JSONToken.OPEN_BRACE.value]

        if self.state == JSONState.EXPECT_PROMPT_KEY:
            return [JSONToken.PROMPT_KEY.value]

        if self.state == JSONState.EXPECT_PROMPT_COLON:
            return [JSONToken.COLON.value]

        if self.state == JSONState.EXPECT_PROMPT_VALUE:
            return ["<prompt_string>"]

        if self.state == JSONState.EXPECT_COMMA_AFTER_PROMPT:
            return [JSONToken.COMMA.value]

        if self.state == JSONState.EXPECT_NAME_KEY:
            return [JSONToken.NAME_KEY.value]

        if self.state == JSONState.EXPECT_NAME_COLON:
            return [JSONToken.COLON.value]

        if self.state == JSONState.EXPECT_NAME_VALUE:
            return ["<function_name_string>", "null"]

        if self.state == JSONState.EXPECT_COMMA_AFTER_NAME:
            return [JSONToken.COMMA.value]

        if self.state == JSONState.EXPECT_PARAMETERS_KEY:
            return [JSONToken.PARAMETERS_KEY.value]

        if self.state == JSONState.EXPECT_PARAMETERS_COLON:
            return [JSONToken.COLON.value]

        if self.state == JSONState.EXPECT_PARAMETERS_OBJECT:
            return ["<parameters_object>", "null"]

        if self.state == JSONState.EXPECT_ROOT_CLOSE:
            return [JSONToken.CLOSE_BRACE.value]

        return []

    def advance(self, token: str) -> None:
        """Move the FSM forward after accepting one generated token.

        Raises:
            ValueError: when the token is not valid for the current state.
        """

        if token not in self.allowed_tokens():
            raise ValueError(
                f"Invalid token {token!r} for state {self.state.name}. "
                f"Allowed: {self.allowed_tokens()}"
            )

        if self.state == JSONState.START:
            self.state = JSONState.EXPECT_PROMPT_KEY
            return

        if self.state == JSONState.EXPECT_PROMPT_KEY:
            self.field = JSONField.PROMPT
            self.state = JSONState.EXPECT_PROMPT_COLON
            return

        if self.state == JSONState.EXPECT_PROMPT_COLON:
            self.state = JSONState.EXPECT_PROMPT_VALUE
            return

        if self.state == JSONState.EXPECT_PROMPT_VALUE:
            self.field = JSONField.NONE
            self.state = JSONState.EXPECT_COMMA_AFTER_PROMPT
            return

        if self.state == JSONState.EXPECT_COMMA_AFTER_PROMPT:
            self.state = JSONState.EXPECT_NAME_KEY
            return

        if self.state == JSONState.EXPECT_NAME_KEY:
            self.field = JSONField.NAME
            self.state = JSONState.EXPECT_NAME_COLON
            return

        if self.state == JSONState.EXPECT_NAME_COLON:
            self.state = JSONState.EXPECT_NAME_VALUE
            return

        if self.state == JSONState.EXPECT_NAME_VALUE:
            self.field = JSONField.NONE
            self.state = JSONState.EXPECT_COMMA_AFTER_NAME
            return

        if self.state == JSONState.EXPECT_COMMA_AFTER_NAME:
            self.state = JSONState.EXPECT_PARAMETERS_KEY
            return

        if self.state == JSONState.EXPECT_PARAMETERS_KEY:
            self.field = JSONField.PARAMETERS
            self.state = JSONState.EXPECT_PARAMETERS_COLON
            return

        if self.state == JSONState.EXPECT_PARAMETERS_COLON:
            self.state = JSONState.EXPECT_PARAMETERS_OBJECT
            return

        if self.state == JSONState.EXPECT_PARAMETERS_OBJECT:
            self.field = JSONField.NONE
            self.state = JSONState.EXPECT_ROOT_CLOSE
            return

        if self.state == JSONState.EXPECT_ROOT_CLOSE:
            self.state = JSONState.END
            return

    def reset(self) -> None:
        """Start the machine again from the beginning."""

        self.state = JSONState.START
        self.field = JSONField.NONE


if __name__ == "__main__":
    # Tiny demo showing the intended order.
    machine = FiniteStateMachine()

    demo_tokens = [
        "{",
        '"prompt"',
        ":",
        "<prompt_string>",
        ",",
        '"name"',
        ":",
        "<function_name_string>",
        ",",
        '"parameters"',
        ":",
        "<parameters_object>",
        "}",
    ]

    for demo_token in demo_tokens:
        print(machine.state.name, "allows", machine.allowed_tokens())
        machine.advance(demo_token)

    print(machine.state.name)
