from utils.types import *
from ..encoder import Encoder, Packing
import struct
from utils.bit_magic import *
from utils.geometry import triangle_list_to_generalized_strip, generalized_strip_to_triangle_list, reorder_vertices

import numpy as np


# def calculate_remapping(strip, codes):
#     # Step 1: Count the frequency of each integer in the array
#     frequency = Counter(strip)
#
#     # Step 2: Sort integers by frequency in descending order
#     sorted_by_frequency = sorted(frequency.keys(), key=lambda x: -frequency[x])
#
#     # Step 3: Sort codes by length in ascending order
#     sorted_codes = {i: code for i, code in enumerate(codes)}
#     sorted_codes = sorted(sorted_codes.items(), key=lambda item: len(item[1]))
#     sorted_code_keys = [key for key, _ in sorted_codes]
#
#     # Step 4: Create a remap dictionary by mapping each integer to the shortest available code
#     indices_remap = {original: new for original, new in zip(sorted_by_frequency, sorted_code_keys)}
#
#     return indices_remap

class GTSQuantizator(Encoder):
    def __init__(self, pack_strip: Packing = Packing.RADIX_BINARY_TREE, verbose=False):
        self.pack_strip: Packing = pack_strip
        self.verbose = verbose

    def encode(self, model: Model) -> CompressedModel:
        triangle_strip, strip_side_bits = triangle_list_to_generalized_strip([[t.a, t.b, t.c] for t in model.triangles])
        vertices = model.vertices

        self.triangle_strip = triangle_strip
        self.strip_side_bits = strip_side_bits
        self.vertices = vertices
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
            print("GTSQuantizator verbose statistics:")
            print(f"- Header (in bytes): {len(byte_array)}")

        len_before_triangle_strip = len(byte_array)

        extend_bytearray_with_fixed_size_values(byte_array, 1, strip_side_bits[3:])
        match self.pack_strip:
            case Packing.FIXED:
                extend_bytearray_with_fixed_size_values(byte_array, int(np.ceil(np.log2(len(model.vertices)))),
                                                        triangle_strip)
            case Packing.BINARY_RANGE_PARTITIONING:
                codes = calculate_codes_using_binary_range_partitioning(len(model.vertices) - 1)

                # if self.allow_reorder:
                #     indices_remapping = calculate_remapping(triangle_strip, codes)
                #     vertices, triangle_strip = reorder_vertices(vertices, triangle_strip, indices_remapping)

                bit_codes = [codes[v] for v in triangle_strip]
                extend_bytearray_with_bit_codes(byte_array, bit_codes)
            case Packing.RADIX_BINARY_TREE:
                codes = calculate_codes_using_binary_radix_tree(len(model.vertices) - 1)

                # if self.allow_reorder:
                #     indices_remapping = calculate_remapping(triangle_strip, codes)
                #     vertices, triangle_strip = reorder_vertices(vertices, triangle_strip, indices_remapping)

                bit_codes = [codes[v] for v in triangle_strip]
                extend_bytearray_with_bit_codes(byte_array, bit_codes)

        if self.verbose:
            print(
                f"- Triangles strip entropy: {(calculate_entropy(triangle_strip) + calculate_entropy(strip_side_bits[3:])):.3f}")
            bytes_used_for_strip = len(byte_array) - len_before_triangle_strip
            print(f"- Triangles strip average code bit length: {bytes_used_for_strip * 8 / len(triangle_strip):.3f}")
            print(f"- Triangles strip (in bytes): {bytes_used_for_strip}")

        len_before_vertices = len(byte_array)
        quantized_vertices = quantize_vertices(vertices, model.aabb, 12)

        extend_bytearray_with_12bit_values(byte_array, quantized_vertices)

        if self.verbose:
            print(f"- Quantized vertices entropy: {calculate_entropy(quantized_vertices)}")
            bytes_used_for_vertices = len(byte_array) - len_before_vertices
            print(
                f"- Quantized vertices average code bit length: {bytes_used_for_vertices * 8 / len(quantized_vertices):.3f}")
            print(f"- Quantized vertices (in bytes): {bytes_used_for_vertices}")

        bits_per_vertex = len(byte_array) * 8 / len(model.vertices)
        bits_per_triangle = len(byte_array) * 8 / len(model.triangles)

        return CompressedModel(bytes(byte_array), bits_per_vertex, bits_per_triangle)


class PackedGTSQuantizator(Encoder):
    def __init__(self, pack_strip: Packing = Packing.RADIX_BINARY_TREE, verbose=False):
        self.pack_strip: Packing = pack_strip
        self.verbose = verbose

    def encode(self, model: Model) -> CompressedModel:
        triangle_strip, strip_side_bits = triangle_list_to_generalized_strip([[t.a, t.b, t.c] for t in model.triangles])
        vertices = model.vertices

        def calculate_reorder_map():
            reorder_map = dict()
            reorder_index = 0
            for vertex in triangle_strip:
                if vertex in reorder_map:
                    continue
                reorder_map[vertex] = reorder_index
                reorder_index += 1

            return reorder_map

        reorder_map = calculate_reorder_map()

        vertices, triangle_list = reorder_vertices(vertices, triangle_strip, reorder_map)

        def calculate_reuse_and_increment_buffers():
            reuse_buffer = []
            used_buffer = {triangle_list[0]}
            increment_flag_buffer = [1]

            for i in range(1, len(triangle_list)):
                current_vertex = triangle_list[i]

                if current_vertex in used_buffer:
                    reuse_buffer.append(current_vertex)
                    increment_flag_buffer.append(0)
                else:
                    increment_flag_buffer.append(1)
                    used_buffer.add(current_vertex)

            return reuse_buffer, increment_flag_buffer

        reuse_buffer, increment_flag_buffer = calculate_reuse_and_increment_buffers()

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
            print("GTSQuantizator verbose statistics:")
            print(f"- Header (in bytes): {len(byte_array)}")

        len_before_triangle_strip = len(byte_array)

        extend_bytearray_with_fixed_size_values(byte_array, 1, strip_side_bits[3:])
        extend_bytearray_with_fixed_size_values(byte_array, 1, increment_flag_buffer[3:])

        max_reuse_buffer_value = np.max(reuse_buffer)
        match self.pack_strip:
            case Packing.FIXED:
                extend_bytearray_with_fixed_size_values(byte_array, int(np.ceil(np.log2(max_reuse_buffer_value))),
                                                        reuse_buffer)
            case Packing.BINARY_RANGE_PARTITIONING:
                codes = calculate_codes_using_binary_range_partitioning(max_reuse_buffer_value)
                bit_codes = [codes[v] for v in reuse_buffer]
                extend_bytearray_with_bit_codes(byte_array, bit_codes)
            case Packing.RADIX_BINARY_TREE:
                codes = calculate_codes_using_binary_radix_tree(max_reuse_buffer_value)
                bit_codes = [codes[v] for v in reuse_buffer]
                extend_bytearray_with_bit_codes(byte_array, bit_codes)

        if self.verbose:
            print(
                f"- Triangles strip entropy: {(calculate_entropy(reuse_buffer) + calculate_entropy(strip_side_bits[3:]) + calculate_entropy(increment_flag_buffer[3:])):.3f}")
            bytes_used_for_strip = len(byte_array) - len_before_triangle_strip
            print(f"- Triangles strip average code bit length: {bytes_used_for_strip * 8 / len(triangle_strip):.3f}")
            print(f"- Triangles strip (in bytes): {bytes_used_for_strip}")

        len_before_vertices = len(byte_array)
        quantized_vertices = quantize_vertices(vertices, model.aabb, 12)

        extend_bytearray_with_12bit_values(byte_array, quantized_vertices)

        if self.verbose:
            print(f"- Quantized vertices entropy: {calculate_entropy(quantized_vertices)}")
            bytes_used_for_vertices = len(byte_array) - len_before_vertices
            print(
                f"- Quantized vertices average code bit length: {bytes_used_for_vertices * 8 / len(quantized_vertices):.3f}")
            print(f"- Quantized vertices (in bytes): {bytes_used_for_vertices}")

        bits_per_vertex = len(byte_array) * 8 / len(model.vertices)
        bits_per_triangle = len(byte_array) * 8 / len(model.triangles)

        return CompressedModel(bytes(byte_array), bits_per_vertex, bits_per_triangle)
