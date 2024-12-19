from abc import ABC, abstractmethod
from utils.types import *
from enum import IntEnum


class Packing(IntEnum):
    NONE = 0
    FIXED = 1
    BINARY_RANGE_PARTITIONING = 2
    RADIX_BINARY_TREE = 3
    HUFFMAN_ENCODER = 4
    ARITHMETIC_ENCODER = 5


class Encoder(ABC):
    @abstractmethod
    def encode(self, model: Model) -> CompressedModel:
        ...
