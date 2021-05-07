from abc import ABC
from typing import Dict, Any

class RLAlgo(ABC):
    ''' Interface of RL algorithm '''
    @staticmethod
    def get_config() -> Dict[str, Any]:
        ''' Get config of the algorithm '''
        raise NotImplementedError

    def __init__(self, config : Dict[str, Any]) -> None:
        ''' Initialize env, network, data collector, buffer and logger '''
        raise NotImplementedError

    def run(self) -> None:
        ''' Main loop of the algorithm, conduct regular training and interaction with the environment '''
        raise NotImplementedError