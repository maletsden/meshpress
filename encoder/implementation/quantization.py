from utils.types import *
from ..encoder import Encoder
import struct

import numpy as np
from collections import defaultdict


def build_vertex_to_triangle_map(triangles):
    vertex_map = defaultdict(list)
    for i, triangle in enumerate(triangles):
        for vertex in triangle:
            vertex_map[vertex].append(i)
    return vertex_map


def find_adjacent_triangle(vertex_map, triangle_list, current_triangle, used):
    # Find an adjacent triangle that shares two vertices with the current triangle
    for vertex in current_triangle:
        for triangle_index in vertex_map[vertex]:
            if triangle_index in used:
                continue
            shared_vertices = set(triangle_list[triangle_index]) & set(current_triangle)
            if len(shared_vertices) == 2:
                return triangle_index, triangle_list[triangle_index]
    return None, None


def find_unused_triangle(triangle_list, used):
    # Find an unused triangle to start a new strip
    for i, triangle in enumerate(triangle_list):
        if i not in used:
            return i, triangle
    return None, None


def triangles_to_strip_with_degenerate(triangle_list):
    if not triangle_list:
        return []

    # Initialize with the first triangle
    used = set()
    strip = list(triangle_list[0])  # Add the first triangle vertices to the strip
    used.add(0)
    current_triangle = triangle_list[0]

    # Preprocess to build vertex-to-triangle map for faster adjacency lookup
    vertex_map = build_vertex_to_triangle_map(triangle_list)

    while len(used) < len(triangle_list):
        next_index, next_triangle = find_adjacent_triangle(vertex_map, triangle_list, current_triangle, used)

        if next_triangle is None:
            # No adjacent triangle found, start a new strip and add degenerate triangles
            next_index, next_triangle = find_unused_triangle(triangle_list, used)
            if next_triangle is None:
                break  # All triangles have been used

            # Add degenerate triangles to stitch the strip
            strip.append(strip[-1])  # Repeat the last vertex
            strip.append(next_triangle[0])  # Add the first vertex of the new triangle

            # Add the entire new triangle to the strip to restart the strip correctly
            strip.extend(next_triangle)

        else:
            # Determine the vertex that is not shared and add it to the strip
            unique_vertex = (set(next_triangle) - set(current_triangle)).pop()
            strip.append(unique_vertex)

        # Mark the new triangle as used and update the current triangle
        used.add(next_index)
        current_triangle = next_triangle

    return strip


def quantize_vertices(vertices: List[Vertex], aabb: AABB):
    data = [0] * (len(vertices) * 3)

    for i in range(len(vertices)):
        quantized_vertex = (vertices[i] - aabb.min) / (aabb.max - aabb.min)
        data[i * 3 + 0] = np.uint32(quantized_vertex.x * (2 ** 12 - 1))
        data[i * 3 + 1] = np.uint32(quantized_vertex.y * (2 ** 12 - 1))
        data[i * 3 + 2] = np.uint32(quantized_vertex.z * (2 ** 12 - 1))

    return data


def extend_bytearray_with_12bit_values(byte_data, values):
    for i in range(len(values) // 2):
        v1 = values[i * 2 + 0]
        v2 = values[i * 2 + 1]

        byte_data.append(v1 >> 4)
        byte_data.append(((v1 & 0xF) << 4) | (v2 >> 8))
        byte_data.append(v2 & 0xFF)

    if len(values) & 1:
        v1 = values[-1]

        byte_data.append(v1 >> 4)
        byte_data.append(((v1 & 0xF) << 4))
        byte_data.append(0)


class SimpleQuantizator(Encoder):
    def __init__(self, verbose = False):
        self.verbose = verbose

    def encode(self, model: Model) -> CompressedModel:
        triangle_strip = triangles_to_strip_with_degenerate([[t.a, t.b, t.c] for t in model.triangles])
        quantized_vertices = quantize_vertices(model.vertices, model.aabb)

        byte_data = bytearray()

        byte_data.extend(struct.pack('I', len(model.vertices)))
        byte_data.extend(struct.pack('I', len(triangle_strip)))

        byte_data.extend(struct.pack('f', model.aabb.min.x))
        byte_data.extend(struct.pack('f', model.aabb.min.y))
        byte_data.extend(struct.pack('f', model.aabb.min.z))

        byte_data.extend(struct.pack('f', model.aabb.max.x))
        byte_data.extend(struct.pack('f', model.aabb.max.y))
        byte_data.extend(struct.pack('f', model.aabb.max.z))

        if self.verbose:
            print("SimpleQuantizator verbose statistics:")
            print(f"- Header (in bytes): {len(byte_data)}")
            print(f"- Triangles strip (in bytes): {len(triangle_strip) * 4}")
            print(f"- Quantized vertices (in bytes): {(len(quantized_vertices) + (len(quantized_vertices) & 1)) // 2 * 3}")

        for index in triangle_strip:
            byte_data.extend(struct.pack('I', index))
        extend_bytearray_with_12bit_values(byte_data, quantized_vertices)

        return CompressedModel(bytes(byte_data))
