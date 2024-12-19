from utils.entropy_encoders.entropy_encoder import EntropyEncoder
from utils.types import *
from ..encoder import Encoder, Packing
import struct
from utils.bit_magic import *
from utils.geometry import triangle_list_to_generalized_strip, reorder_vertices
from utils.entropy_encoders import IntegerBinaryArithmeticEncoder, HuffmanEncoder
import numpy as np
from scipy.stats import norm


class GTSParallelogramPredictor(Encoder):
    def __init__(self, pack_strip: Packing = Packing.RADIX_BINARY_TREE,
                 entropy_encoder_type: Packing = Packing.ARITHMETIC_ENCODER,
                 enable_entropy_encoder_conditional_probabilities: bool = False, verbose=False):
        self.pack_strip: Packing = pack_strip
        self.verbose = verbose
        self.entropy_encoder_type = entropy_encoder_type
        self.enable_entropy_encoder_conditional_probabilities = enable_entropy_encoder_conditional_probabilities

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

        vertices, triangle_strip = reorder_vertices(vertices, triangle_strip, reorder_map)

        self.triangle_strip = triangle_strip
        self.strip_side_bits = strip_side_bits
        self.vertices = vertices

        def calculate_reuse_and_increment_buffers():
            reuse_buffer = []
            used_buffer = {triangle_strip[0]}
            increment_flag_buffer = [1]

            for i in range(1, len(triangle_strip)):
                current_vertex = triangle_strip[i]

                if current_vertex in used_buffer:
                    reuse_buffer.append(current_vertex)
                    increment_flag_buffer.append(0)
                else:
                    increment_flag_buffer.append(1)
                    used_buffer.add(current_vertex)

            return reuse_buffer, increment_flag_buffer

        reuse_buffer, increment_flag_buffer = calculate_reuse_and_increment_buffers()

        self.reuse_buffer = reuse_buffer
        self.increment_flag_buffer = increment_flag_buffer
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
        vertices_residuals = self._parallelogram_predictor_generalized_strip(vertices, triangle_strip, strip_side_bits)
        vertices_residuals_quantized, vertices_residuals_quantized_bits_needed = self._quantize_residuals(
            vertices_residuals)

        vertices_residuals_quantized_flat = vertices_residuals_quantized.flatten()
        vertices_residuals_frequencies = Counter(vertices_residuals_quantized_flat)

        self._write_packed_frequencies(byte_array, vertices_residuals_quantized_bits_needed,
                                       vertices_residuals_quantized_flat, vertices_residuals_frequencies)

        encoder: Union[EntropyEncoder, None] = None
        match self.entropy_encoder_type:
            case Packing.HUFFMAN_ENCODER:
                encoder = HuffmanEncoder(vertices_residuals_frequencies,
                                         enable_conditional_probabilities=self.enable_entropy_encoder_conditional_probabilities)

            case Packing.ARITHMETIC_ENCODER:
                encoder = IntegerBinaryArithmeticEncoder(vertices_residuals_frequencies,
                                                         enable_conditional_probabilities=self.enable_entropy_encoder_conditional_probabilities)

        encoded_stream = encoder.encode(vertices_residuals_quantized_flat.tolist())
        extend_bytearray_with_bitstream(byte_array, encoded_stream)

        if self.verbose:
            print(f"- Quantized vertices entropy: {calculate_entropy(vertices_residuals_quantized_flat)}")
            bytes_used_for_vertices = len(byte_array) - len_before_vertices
            print(
                f"- Quantized vertices average code bit length: {bytes_used_for_vertices * 8 / len(vertices_residuals_quantized_flat):.3f}")
            print(f"- Quantized vertices (in bytes): {bytes_used_for_vertices}")

        bits_per_vertex = len(byte_array) * 8 / len(model.vertices)
        bits_per_triangle = len(byte_array) * 8 / len(model.triangles)

        return CompressedModel(bytes(byte_array), bits_per_vertex, bits_per_triangle)

    def _parallelogram_predictor_generalized_strip(self, vertices, triangle_strip, side_bits):
        """
        Encodes a triangle mesh using the parallelogram predictor with a triangle strip and side bits.

        Parameters:
            vertices (np.ndarray): Array of vertex positions, shape (N, 3).
            triangle_strip (list of int): List of vertex indices defining the triangle strip.
            side_bits (list of int): List of side bits indicating the order of triangle connection (0 or 1).

        Returns:
            residuals (list of np.ndarray): Residuals calculated using the parallelogram predictor.
        """
        residuals = []  # To store residuals (differences between predicted and actual vertices)

        # First triangle needs no prediction (store raw positions or deltas from initial reference point)
        V0 = vertices[triangle_strip[0]]  # First vertex
        V1 = vertices[triangle_strip[1]]  # Second vertex
        V2 = vertices[triangle_strip[2]]  # Third vertex

        # Store the first residuals (raw position or simple difference with previous vertices)
        residuals.append(V0)
        residuals.append(V1 - V0)
        residuals.append(V2 - V1)

        # Start parallelogram prediction from the 4th vertex onward
        for i in range(3, len(triangle_strip)):
            current_vertex_index = triangle_strip[i]
            current_vertex_position = vertices[current_vertex_index]

            # Determine prediction based on the side bit
            if side_bits[i - 3] == 0:
                # Predict the next vertex assuming the triangle follows a "left" connection
                predicted_vertex = V1 + (V2 - V0)
            else:
                # Predict the next vertex assuming the triangle follows a "right" connection
                predicted_vertex = V2 + (V1 - V0)

            # Compute the residual as the difference between actual and predicted vertices
            residual = current_vertex_position - predicted_vertex
            residuals.append(residual)

            # Update vertices for the next iteration
            V0, V1, V2 = V1, V2, current_vertex_position

        return residuals

    def _quantize_residuals(self, residuals, max_error=0.001):
        """
        Quantizes residuals starting from the 3rd residual.

        Parameters:
            residuals (list of np.ndarray): Residuals to be quantized.
            max_error (float): The acceptable maximum quantization error.

        Returns:
            quantized_residuals (np.ndarray): Quantized residuals.
            bits_needed (int): Minimum bits needed for quantization with error <= max_error.
        """
        # Start quantization from the 3rd residual
        residuals = np.array([[t.x, t.y, t.z] for t in residuals])

        # Find min/max for each dimension (x, y, z)
        r_min = np.min(residuals, axis=0)
        r_max = np.max(residuals, axis=0)

        # print(f"Residual min: {r_min}")
        # print(f"Residual max: {r_max}")

        # Normalize residuals to the range [0, 1]
        normalized_residuals = (residuals - r_min) / (r_max - r_min)

        # print("\nNormalized Residuals (first 5):")
        # print(normalized_residuals[:5])

        # Determine the number of bits required for small enough error
        quantization_levels = 2
        bits_needed = 1  # Start with 1-bit quantization
        max_quantization_error = None
        while True:
            # Quantize to the current number of levels
            quantized = np.round(normalized_residuals * (quantization_levels - 1))

            # Dequantize to compute the reconstruction error
            dequantized = quantized / (quantization_levels - 1) * (r_max - r_min) + r_min

            # Calculate the maximum error
            max_quantization_error = np.max(np.abs(residuals - dequantized))

            if max_quantization_error <= max_error:
                break  # Stop if error is acceptable

            # Otherwise, increase the number of bits
            quantization_levels *= 2  # Double levels (e.g., 4 -> 8 -> 16)
            bits_needed += 1

        # print(f"\nBits Needed: {bits_needed}")
        # print(f"\nMax quantization error: {max_quantization_error}")
        # print(f"Quantized Residuals (first 5):")
        # print(quantized[:5])

        # Return quantized values and the number of bits needed
        return quantized.astype(np.int32), bits_needed

    def _write_packed_frequencies(self, byte_array, vertices_residuals_quantized_bits_needed,
                                  vertices_residuals_quantized_flat, vertices_residuals_frequencies):
        vertices_residuals_frequencies_flat = [0 for _ in range(2 ** vertices_residuals_quantized_bits_needed)]
        for symbol, frequency in vertices_residuals_frequencies.items():
            vertices_residuals_frequencies_flat[symbol] = frequency

        mean = np.mean(vertices_residuals_quantized_flat)
        std_dev = np.std(vertices_residuals_quantized_flat)

        # Fit the Normal Distribution
        # Generate a range of values for plotting the theoretical PDF
        x = np.linspace(0, 255, 256)
        pdf = norm.pdf(x, mean, np.sqrt(std_dev))
        estimated_frequencies = np.int32(np.round(pdf * np.sum(vertices_residuals_frequencies_flat)))

        symbol_frequencies_residuals = estimated_frequencies - vertices_residuals_frequencies_flat

        byte_array.extend(struct.pack('f', mean))
        byte_array.extend(struct.pack('f', std_dev))
        byte_array.extend(struct.pack('i', int(np.min(symbol_frequencies_residuals))))
        bits_needed = int(np.ceil(np.log2(np.max(symbol_frequencies_residuals - np.min(symbol_frequencies_residuals)))))
        byte_array.extend(struct.pack('b', np.int8(bits_needed)))
        extend_bytearray_with_fixed_size_values(byte_array, bits_needed,
                                                symbol_frequencies_residuals - np.min(symbol_frequencies_residuals))
