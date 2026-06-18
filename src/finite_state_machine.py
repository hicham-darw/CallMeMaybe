from src.state import JSONState


class FiniteStateMachine:
    
    def __init__(self) -> None:
        self.__state = JSONState.IN_START

    def get_state(self) -> None:
        return self.__state
    
    def set_state(self, new_state: JSONState) -> None:
        self.__state = new_state

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

    def goto_next_state(self) -> None:
        if self.get_state() == JSONState.IN_START:
            self.set_state(JSONState.IN_PROMPT_KEY)
        
        elif self.get_state() == JSONState.IN_PROMPT_KEY:
        	self.set_state(JSONState.IN_PROMPT_COLON)
        
        elif self.get_state() == JSONState.IN_PROMPT_COLON:
        	self.set_state(JSONState.IN_PROMPT_VALUE)
        
        elif self.get_state() == JSONState.IN_PROMPT_VALUE:
        	self.set_state(JSONState.IN_COMMA_AFTER_PROMPT)
        
        elif self.get_state() == JSONState.IN_COMMA_AFTER_PROMPT:
        	self.set_state(JSONState.IN_NAME_KEY)
        
        elif self.get_state() == JSONState.IN_NAME_KEY:
        	self.set_state(JSONState.IN_NAME_COLON)
        
        elif self.get_state() == JSONState.IN_NAME_COLON:
        	self.set_state(JSONState.IN_NAME_VALUE)
        
        elif self.get_state() == JSONState.IN_NAME_VALUE:
        	self.set_state(JSONState.IN_COMMA_AFTER_NAME)
        
        elif self.get_state() == JSONState.IN_COMMA_AFTER_NAME:
        	self.set_state(JSONState.IN_PARAMETERS_KEY)
        
        elif self.get_state() == JSONState.IN_PARAMETERS_KEY:
        	self.set_state(JSONState.IN_PARAMETERS_COLON)
        
        elif self.get_state() == JSONState.IN_PARAMETERS_COLON:
        	self.set_state(JSONState.IN_PARAMETERS_VALUE)
        
        elif self.get_state() == JSONState.IN_PARAMETERS_VALUE:
        	self.set_state(JSONState.IN_END)
