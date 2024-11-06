from typing import List, Generator
import numpy as np


def extend_bytearray_with_12bit_values(byte_array: bytearray, values: List[int]):
    for i in range(len(values) // 2):
        v1 = values[i * 2 + 0]
        v2 = values[i * 2 + 1]

        byte_array.append(v1 >> 4)
        byte_array.append(((v1 & 0xF) << 4) | (v2 >> 8))
        byte_array.append(v2 & 0xFF)

    if len(values) & 1:
        v1 = values[-1]

        byte_array.append(v1 >> 4)
        byte_array.append(((v1 & 0xF) << 4))
        byte_array.append(0)


def extend_bytearray_with_bit_codes(byte_array: bytearray, bit_codes: List[str]) -> None:
    # Initialize a binary string to accumulate bits
    binary_string = ''.join(bit_codes)

    # Make sure the binary string length is a multiple of 8 by padding with zeros if necessary
    padding_length = (8 - len(binary_string) % 8) % 8
    binary_string += '0' * padding_length

    # Convert the binary string to a bytearray
    byte_array.extend(int(binary_string[i:i + 8], 2) for i in range(0, len(binary_string), 8))


def extend_bytearray_with_fixed_size_values(byte_array: bytearray, size: int, values: List[int]) -> None:
    bit_codes: List[str] = [bin(v)[2:].rjust(size, "0") for v in values]
    extend_bytearray_with_bit_codes(byte_array, bit_codes)


def calculate_codes_using_binary_range_partitioning(number):
    def calculate_binary_range_partitioning(number):
        number_ranges = []

        bits = [int(b) for b in bin(number)[2:]]
        range_start = 0
        range_end = 0
        zeros_skipped = 0

        for i in range(len(bits)):
            range_end += bits[i] * 2 ** (len(bits) - i - 1)
            zeros_skipped += bits[i] == 0

            if bits[i] == 1 and ((i == (len(bits) - 1)) or bits[i + 1] == 0):
                number_ranges.append((range_start, range_end, len(bits) - zeros_skipped))
                range_start = range_end

        return number_ranges

    def calculate_codes_based_on_partitioned_binary_ranges(number_ranges) -> List[str]:
        if len(number_ranges) == 0:
            return []

        number = number_ranges[-1][1]
        codes = [None] * number
        max_bits = int(np.ceil(np.log2(number)))

        for numbers_range in number_ranges:
            range_start, range_end, bits_needed = numbers_range
            for i in range(range_start, range_end):
                prefix_length = (max_bits - bits_needed)
                codes[i] = "1" * prefix_length + bin(i - range_start)[2:].rjust(bits_needed - prefix_length, '0')
        return codes

    number_ranges = calculate_binary_range_partitioning(number)
    codes = calculate_codes_based_on_partitioned_binary_ranges(number_ranges)

    return codes


def calculate_codes_using_binary_radix_tree(number):
    numbers_list = list(range(0, number))
    codes = [""] * len(numbers_list)

    def split_list(numbers_list, mask):
        left = [val for val in numbers_list if (val & mask) == 0]
        right = [val for val in numbers_list if (val & mask) > 0]

        return left, right

    def calculate_codes_for_number_impl(numbers_list, numbers_code="", mask=0b1):
        if len(numbers_list) == 1:
            codes[numbers_list[0]] = numbers_code
            return
        if len(numbers_list) == 0:
            return

        left, right = split_list(numbers_list, mask)
        calculate_codes_for_number_impl(left, "0" + numbers_code, mask << 1)
        calculate_codes_for_number_impl(right, "1" + numbers_code, mask << 1)

    calculate_codes_for_number_impl(numbers_list)
    return codes