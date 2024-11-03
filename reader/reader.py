from abc import ABC, abstractmethod
from utils.types import *


class Reader(ABC):
    @abstractmethod
    def read(self, path: str) -> Model:
        ...

    @staticmethod
    def read_from_file(path: str) -> Model:
        if path.endswith(".obj"):
            from .implementation.obj_reader import OBJReader
            return OBJReader().read(path)

        raise NotImplementedError("Reader is not implemented yet for provided format.")
