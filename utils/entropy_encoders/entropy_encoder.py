from abc import ABC, abstractmethod
from typing import List, Dict, Union, TypeAlias

Symbol: TypeAlias = Union[str, int]


class EntropyEncoder(ABC):

    @abstractmethod
    def __init__(self, symbol_frequencies: Dict[Symbol, int]):
        ...

    @abstractmethod
    def encode(self, data: List[Symbol]) -> str:
        ...

    @abstractmethod
    def decode(self, encoded_data: str) -> List[Symbol]:
        ...
