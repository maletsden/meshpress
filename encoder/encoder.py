from abc import ABC, abstractmethod
from utils.types import *


class Encoder(ABC):
    @abstractmethod
    def encode(self, model: Model) -> CompressedModel:
        ...

