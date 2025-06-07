from utils.types import *
from ..encoder import Encoder, Packing
import struct
from utils.bit_magic import *
import math
from utils.geometry import triangle_list_to_strip

import numpy as np
from collections import Counter


def calculate_remapping(strip, codes):
    # Step 1: Count the frequency of each integer in the array
    frequency = Counter(strip)

    # Step 2: Sort integers by frequency in descending order
    sorted_by_frequency = sorted(frequency.keys(), key=lambda x: -frequency[x])

    # Step 3: Sort codes by length in ascending order
    sorted_codes = {i: code for i, code in enumerate(codes)}
    sorted_codes = sorted(sorted_codes.items(), key=lambda item: len(item[1]))
    sorted_code_keys = [key for key, _ in sorted_codes]

    # Step 4: Create a remap dictionary by mapping each integer to the shortest available code
    indices_remap = {original: new for original, new in zip(sorted_by_frequency, sorted_code_keys)}

    return indices_remap


def reorder_vertices(vertices, triangle_strip, indices_remap):
    indices_remapping_inverse = {new: old for old, new in indices_remap.items()}
    reordered_vertices = [vertices[indices_remapping_inverse[i]] for i in range(len(vertices))]
    reordered_triangle_strip = [indices_remap[v] for v in triangle_strip]

    return reordered_vertices, reordered_triangle_strip


class SimpleQuantizator(Encoder):
    def __init__(self, pack_strip: Packing = Packing.RADIX_BINARY_TREE, pack_vertices: Packing = Packing.FIXED,
                 vertex_quantization_error: float = 0.0005,
                 allow_reorder=False,
                 verbose=False):
        self.pack_strip: Packing = pack_strip
        self.pack_vertices: Packing = pack_vertices
        self.vertex_quantization_error: float = vertex_quantization_error
        self.verbose = verbose
        self.allow_reorder = allow_reorder

    def encode(self, model: Model) -> CompressedModel:
        triangle_strip = []

        for i, strip in enumerate(model.triangle_strips):
            if i != 0:
                triangle_strip.append(strip[0])
                triangle_strip.append(strip[0])
            triangle_strip.extend(strip)
            if i != len(model.triangle_strips) - 1:
                triangle_strip.append(strip[-1])
                triangle_strip.append(strip[-1])
        vertices = model.vertices

        byte_array = bytearray()

        byte_array.extend(struct.pack('I', len(model.vertices)))
        byte_array.extend(struct.pack('I', len(triangle_strip)))

        byte_array.extend(struct.pack('f', model.aabb.min.x))
        byte_array.extend(struct.pack('f', model.aabb.min.y))
        byte_array.extend(struct.pack('f', model.aabb.min.z))

        byte_array.extend(struct.pack('f', model.aabb.max.x))
        byte_array.extend(struct.pack('f', model.aabb.max.y))
        byte_array.extend(struct.pack('f', model.aabb.max.z))

        if self.verbose:
            print("SimpleQuantizator verbose statistics:")
            print(f"- Header (in bytes): {len(byte_array)}")

        len_before_triangle_strip = len(byte_array)

        match self.pack_strip:
            case Packing.NONE:
                for index in triangle_strip:
                    byte_array.extend(struct.pack('I', index))
            case Packing.FIXED:
                extend_bytearray_with_fixed_size_values(byte_array, int(np.ceil(np.log2(len(model.vertices)))),
                                                        triangle_strip)
            case Packing.BINARY_RANGE_PARTITIONING:
                codes = calculate_codes_using_binary_range_partitioning(len(model.vertices) - 1)

                if self.allow_reorder:
                    indices_remapping = calculate_remapping(triangle_strip, codes)
                    vertices, triangle_strip = reorder_vertices(vertices, triangle_strip, indices_remapping)

                bit_codes = [codes[v] for v in triangle_strip]
                extend_bytearray_with_bit_codes(byte_array, bit_codes)
            case Packing.RADIX_BINARY_TREE:
                codes = calculate_codes_using_binary_radix_tree(len(model.vertices) - 1)

                if self.allow_reorder:
                    indices_remapping = calculate_remapping(triangle_strip, codes)
                    vertices, triangle_strip = reorder_vertices(vertices, triangle_strip, indices_remapping)

                bit_codes = [codes[v] for v in triangle_strip]
                extend_bytearray_with_bit_codes(byte_array, bit_codes)

        if self.verbose:
            print(f"- Triangles strip entropy: {calculate_entropy(triangle_strip):.3f}")
            bytes_used_for_strip = len(byte_array) - len_before_triangle_strip
            print(f"- Triangles strip average code bit length: {bytes_used_for_strip * 8 / len(triangle_strip):.3f}")
            print(f"- Triangles strip (in bytes): {bytes_used_for_strip}")

        len_before_vertices = len(byte_array)
        aabb = model.aabb
        aabb_size = model.aabb.size()
        aabb_size_largest = max(aabb_size.x, aabb_size.y, aabb_size.z)

        max_quantized_value_x = math.ceil(aabb_size.x / (aabb_size_largest * self.vertex_quantization_error))
        max_quantized_value_y = math.ceil(aabb_size.y / (aabb_size_largest * self.vertex_quantization_error))
        max_quantized_value_z = math.ceil(aabb_size.z / (aabb_size_largest * self.vertex_quantization_error))

        byte_array.extend(struct.pack('I', max_quantized_value_x))
        byte_array.extend(struct.pack('I', max_quantized_value_y))
        byte_array.extend(struct.pack('I', max_quantized_value_z))

        quantized_vertices_x = [
            np.uint32((v.x - aabb.min.x) / (aabb.max.x - aabb.min.x) * max_quantized_value_x) for v in vertices
        ]
        quantized_vertices_y = [
            np.uint32((v.y - aabb.min.y) / (aabb.max.y - aabb.min.y) * max_quantized_value_y) for v in vertices
        ]
        quantized_vertices_z = [
            np.uint32((v.z - aabb.min.z) / (aabb.max.z - aabb.min.z) * max_quantized_value_z) for v in vertices
        ]

        assert len(quantized_vertices_x) == len(vertices)
        assert len(quantized_vertices_y) == len(vertices)
        assert len(quantized_vertices_z) == len(vertices)

        for data, max_val in zip(
                (quantized_vertices_x, quantized_vertices_y, quantized_vertices_z),
                (max_quantized_value_x, max_quantized_value_y, max_quantized_value_z)
        ):
            len_byte_array_local = len(byte_array)
            match self.pack_vertices:
                case Packing.NONE:
                    extend_bytearray_with_fixed_size_values(byte_array, 32, data)
                case Packing.FIXED:
                    extend_bytearray_with_fixed_size_values(byte_array, math.ceil(math.log2(max_val)), data)
                case Packing.BINARY_RANGE_PARTITIONING:
                    codes = calculate_codes_using_binary_range_partitioning(max_val)
                    bit_codes = [codes[v] for v in data]
                    extend_bytearray_with_bit_codes(byte_array, bit_codes)
                case Packing.RADIX_BINARY_TREE:
                    codes = calculate_codes_using_binary_radix_tree(max_val)
                    bit_codes = [codes[v] for v in data]
                    extend_bytearray_with_bit_codes(byte_array, bit_codes)
            if self.verbose:
                print(f"- Max quantized value: {max_val}")
                print(f"- Average bits used for quantized value: {(len(byte_array) - len_byte_array_local) * 8 / len(data):.3f}")
        if self.verbose:
            print(
                f"- Quantized vertices entropy: {sum(calculate_entropy(d) for d in (quantized_vertices_x, quantized_vertices_y, quantized_vertices_z))}")
            bytes_used_for_vertices = len(byte_array) - len_before_vertices
            print(
                f"- Quantized vertices average code bit length: {bytes_used_for_vertices * 8 / (len(quantized_vertices_x)):.3f}")

            print(f"- Quantized vertices (in bytes): {bytes_used_for_vertices}")

        bits_per_vertex = len(byte_array) * 8 / len(model.vertices)
        bits_per_triangle = len(byte_array) * 8 / len(model.triangles)

        return CompressedModel(bytes(byte_array), bits_per_vertex, bits_per_triangle)
