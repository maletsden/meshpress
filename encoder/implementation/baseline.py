from utils.types import *
import struct
from ..encoder import Encoder


class BaselineEncoder(Encoder):
    def encode(self, model: Model) -> CompressedModel:
        byte_array = bytearray()

        byte_array.extend(struct.pack('I', len(model.vertices)))
        byte_array.extend(struct.pack('I', len(model.triangles)))

        for vertex in model.vertices:
            byte_array.extend(struct.pack('f', vertex.x))
            byte_array.extend(struct.pack('f', vertex.y))
            byte_array.extend(struct.pack('f', vertex.z))
        for triangle in model.triangles:
            byte_array.extend(struct.pack('I', triangle.a))
            byte_array.extend(struct.pack('I', triangle.b))
            byte_array.extend(struct.pack('I', triangle.c))

        bits_per_vertex = len(byte_array) * 8 / len(model.vertices)
        bits_per_triangle = len(byte_array) * 8 / len(model.triangles)

        return CompressedModel(bytes(byte_array), bits_per_vertex, bits_per_triangle)
