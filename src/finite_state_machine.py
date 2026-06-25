from src.state import JSONState, JSONStatic, ParameterState


class FiniteStateMachine:
    
    def __init__(self) -> None:
        self.__state = JSONState.BEFORE_PROMPT
        self.__static_json = JSONStatic.STR_BEFORE_PROMPT
        self.__parameters_state = ParameterState.IN_KEY

    def get_static_json(self) -> str:
        return self.__static_json.value

    def get_state(self) -> JSONState:
        return self.__state
    
    def get_parameters_state(self) -> ParameterState:
        return self.__parameters_state

    def set_parameters_state(self, state: ParameterState) -> None:
        self.__parameters_state = state

    def set_state(self, new_state: JSONState) -> None:
        self.__state = new_state

    def set_static_json(self, state: JSONStatic) -> None:
    	self.__static_json = state
    
    def is_in_end_state(self) -> bool:
        if self.__state == JSONState.IN_END:
            return True
        return False

    def is_closed_json(self, json_params: str) -> bool:
        json_params = json_params.strip()
        if json_params[0] == '{' and json_params[-1] == '}':
            return True
        return False

    def is_finished_parameters_value(self, generated_str) -> bool:
        if self.__state == JSONState.IN_PARAMETERS_VALUE\
                and self.is_closed_json(generated_str):
            return True
        return False

    def is_in_state_static_tokens(self) -> bool:
        if self.__state == JSONState.BEFORE_PARAMETERS:
            return True
        return False
        

    def goto_next_state(self) -> None:
        if self.get_state() == JSONState.IN_NAME:
            self.set_state(JSONState.BEFORE_PARAMETERS)
        elif self.get_state() == JSONState.BEFORE_PARAMETERS:
            self.set_state(JSONState.IN_PARAMETERS)
        elif self.get_state() == JSONState.IN_PARAMETERS:
            self.set_state(JSONState.IN_END)
        elif self.get_state() == JSONState.IN_END:
            self.set_state(JSONState.IN_NAME)

    def goto_next_static_json(self) -> None:
        if self.__static_json == JSONStatic.STR_BEFORE_NAME:
            self.__static_json = JSONStatic.STR_BEFORE_PARAMETERS
        else:
            self.__static_json = JSONStatic.STR_BEFORE_NAME

            
