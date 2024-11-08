from utils.types import *
from ..encoder import Encoder
import struct
from utils.bit_magic import *
from utils.huffman_encoder import *
from enum import IntEnum

import numpy as np
from collections import defaultdict, Counter


def triangle_list_to_strip(triangle_list):
    """
    Convert a triangle list to a triangle strip with degenerate triangles if needed.

    Parameters:
    triangle_list (list): A list of triangles, where each triangle is represented by three vertices [(x1, y1, z1), (x2, y2, z2), (x3, y3, z3)].

    Returns:
    list: A list of vertices representing the triangle strip, including degenerate triangles if needed.
    """
    if len(triangle_list) < 2:
        return [v for triangle in triangle_list for v in triangle]

    strip = []
    used = [False] * len(triangle_list)

    # Start with the first triangle
    strip.extend(triangle_list[0])
    used[0] = True

    current_triangle = triangle_list[0]

    while not all(used):
        found_shared = False

        # Search for the next triangle that shares an edge with the current triangle
        for i in range(len(triangle_list)):
            if used[i]:
                continue

            t2 = triangle_list[i]

            shared_vertices = [v for v in t2 if v in [strip[-1], strip[-2]]]

            if len(shared_vertices) == 2:
                # Triangles share an edge, add the new vertex while preserving order
                new_vertex = [v for v in t2 if v not in shared_vertices][0]
                # if current_triangle.index(shared_vertices[0]) < current_triangle.index(shared_vertices[1]):
                #     strip.append(new_vertex)
                # else:
                strip.append(new_vertex)
                # current_triangle = t2
                used[i] = True
                found_shared = True
                break

        if not found_shared:
            # No shared edge found, add degenerate triangles to connect
            # strip.append(current_triangle[2])  # Repeat the last vertex of the current triangle
            strip.append(strip[-1])  # Repeat the last vertex of the current triangle
            for i in range(len(triangle_list)):
                if not used[i]:
                    strip.append(triangle_list[i][0])  # Repeat the first vertex of the next triangle

                    if len(strip) % 2 == 0:
                        strip.extend([triangle_list[i][0], triangle_list[i][1], triangle_list[i][2]])

                    else:
                        strip.extend([triangle_list[i][0], triangle_list[i][2], triangle_list[i][1]])

                    # strip.extend(triangle_list[i])     # Add the new triangle
                    # current_triangle = triangle_list[i]
                    used[i] = True
                    break

    return strip


def triangle_strip_to_list(triangle_strip):
    """
    Convert a triangle strip to a triangle list.

    Parameters:
    triangle_strip (list): A list of vertices representing the triangle strip.

    Returns:
    list: A list of triangles, where each triangle is represented by three vertices.
    """
    if len(triangle_strip) < 3:
        return []

    triangle_list = []

    # Iterate through the strip to form triangles
    for i in range(2, len(triangle_strip)):
        if triangle_strip[i] == triangle_strip[i - 1] or triangle_strip[i] == triangle_strip[i - 2] or triangle_strip[
            i - 1] == triangle_strip[i - 2]:
            # Skip degenerate triangles
            continue
        if i % 2 == 0:
            triangle = (triangle_strip[i - 2], triangle_strip[i - 1], triangle_strip[i])
        else:
            triangle = (triangle_strip[i - 2], triangle_strip[i], triangle_strip[i - 1])
        triangle_list.append(triangle)

    return triangle_list


def quantize_vertices(vertices: List[Vertex], aabb: AABB):
    data = [0] * (len(vertices) * 3)

    for i in range(len(vertices)):
        quantized_vertex = (vertices[i] - aabb.min) / (aabb.max - aabb.min)
        data[i * 3 + 0] = np.uint32(quantized_vertex.x * (2 ** 12 - 1))
        data[i * 3 + 1] = np.uint32(quantized_vertex.y * (2 ** 12 - 1))
        data[i * 3 + 2] = np.uint32(quantized_vertex.z * (2 ** 12 - 1))

    return data


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


class Packing(IntEnum):
    NONE = 0
    FIXED = 1
    BINARY_RANGE_PARTITIONING = 2
    RADIX_BINARY_TREE = 3
    HUFFMAN_ENCODER = 4


class SimpleQuantizator(Encoder):

    def __init__(self, pack_strip: Packing = Packing.RADIX_BINARY_TREE, pack_vertices: Packing = Packing.FIXED,
                 allow_reorder=False,
                 verbose=False):
        self.pack_strip: Packing = pack_strip
        self.pack_vertices: Packing = pack_vertices
        self.verbose = verbose
        self.allow_reorder = allow_reorder

    def encode(self, model: Model) -> CompressedModel:
        triangle_strip = triangle_list_to_strip([[t.a, t.b, t.c] for t in model.triangles])
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

            case Packing.HUFFMAN_ENCODER:
                frequencies, huffman_codes = calculate_huffman_codes(triangle_strip)
                huffman_codes_array = [(huffman_codes[i] if i in huffman_codes else "0") for i in
                                       range(len(model.vertices))]
                extend_bytearray_with_bit_codes(byte_array, huffman_codes_array)

                bit_codes = [huffman_codes_array[v] for v in triangle_strip]
                extend_bytearray_with_bit_codes(byte_array, bit_codes)

        if self.verbose:
            print(f"- Triangles strip entropy: {calculate_entropy(triangle_strip):.3f}")
            bytes_used_for_strip = len(byte_array) - len_before_triangle_strip
            print(f"- Triangles strip average code bit length: {bytes_used_for_strip * 8 / len(triangle_strip):.3f}")
            print(f"- Triangles strip (in bytes): {bytes_used_for_strip}")

        len_before_vertices = len(byte_array)
        quantized_vertices = quantize_vertices(vertices, model.aabb)

        match self.pack_vertices:
            case Packing.NONE:
                for vertex in quantized_vertices:
                    byte_array.extend(struct.pack('I', vertex))
            case Packing.FIXED:
                extend_bytearray_with_12bit_values(byte_array, quantized_vertices)
            case Packing.BINARY_RANGE_PARTITIONING:
                codes = calculate_codes_using_binary_range_partitioning(2 ** 12 - 1)
                bit_codes = [codes[v] for v in quantized_vertices]
                extend_bytearray_with_bit_codes(byte_array, bit_codes)
            case Packing.RADIX_BINARY_TREE:
                codes = calculate_codes_using_binary_radix_tree(2 ** 12 - 1)
                bit_codes = [codes[v] for v in quantized_vertices]
                extend_bytearray_with_bit_codes(byte_array, bit_codes)

        if self.verbose:
            print(f"- Quantized vertices entropy: {calculate_entropy(quantized_vertices)}")
            bytes_used_for_vertices = len(byte_array) - len_before_vertices
            print(
                f"- Quantized vertices average code bit length: {bytes_used_for_vertices * 8 / len(quantized_vertices):.3f}")
            print(f"- Quantized vertices (in bytes): {bytes_used_for_vertices}")

        bits_per_vertex = len(byte_array) * 8 / len(model.vertices)
        bits_per_triangle = len(byte_array) * 8 / len(model.triangles)

        return CompressedModel(bytes(byte_array), bits_per_vertex, bits_per_triangle)
