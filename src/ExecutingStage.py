from typing import Any
from abc import ABC, abstractmethod


class ExecutingStage(ABC):
    """interface of json stages"""
    @abstractmethod
    def execute(self, data: Any) -> Any:
        """must implement this function on all derived class"""
        pass
