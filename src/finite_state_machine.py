from src.state import JSONState, JSONStatic, ParameterState


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
        """ checking if is end state or not"""
        if self.__state == JSONState.IN_END:
            return True
        return False

    def reinitial_stats(self) -> None:
        """ reinitial state for each jenerating json"""
        self.set_state(JSONState.BEFORE_PROMPT)
        self.set_static_json(JSONStatic.STR_BEFORE_PROMPT)
        self.set_parameters_state(ParameterState.IN_KEY)

    def is_closed_json(self, json_params: str) -> bool:
        """is closed bracket to know generating is well"""
        json_params = json_params.strip()
        if json_params[0] == '{' and json_params[-1] == '}':
            return True
        return False

    def is_finished_parameters_value(self, generated_str) -> bool:
        """check if value of parameters is finihsed or not"""
        if self.__state == JSONState.IN_PARAMETERS_VALUE\
                and self.is_closed_json(generated_str):
            return True
        return False        
            
