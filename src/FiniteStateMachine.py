from src.Enums import JSONState, JSONStatic, ParameterState


class FiniteStateMachine:
    
    def __init__(self) -> None:
        """constructor of finite state machine  initial stats"""
        self.__state = JSONState.BEFORE_PROMPT
        self.__static_json = JSONStatic.STR_BEFORE_PROMPT
        self.__parameters_state = ParameterState.IN_KEY

    def get_static_json(self) -> str:
        """ get static json static data json as string"""
        return self.__static_json.value

    def get_state(self) -> JSONState:
        """ get state of finite state machine"""
        return self.__state
    
    def get_parameters_state(self) -> ParameterState:
        """ get parameters state"""
        return self.__parameters_state

    def set_parameters_state(self, state: ParameterState) -> None:
        """ set parameters state """
        self.__parameters_state = state

    def set_state(self, new_state: JSONState) -> None:
        """ set state finite state machine """
        self.__state = new_state

    def set_static_json(self, state: JSONStatic) -> None:
        """ set static json state """
        self.__static_json = state

    def is_in_end_state(self) -> bool:
        """ checking if is in end state or not"""
        if self.__state == JSONState.IN_END:
            return True
        return False

    def reinitial_stats(self) -> None:
        """ reinitial state for each jenerating json"""
        self.set_state(JSONState.BEFORE_PROMPT)
        self.set_static_json(JSONStatic.STR_BEFORE_PROMPT)
        self.set_parameters_state(ParameterState.IN_KEY)
