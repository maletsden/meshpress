import math
import numpy as np
from tqdm import tqdm
from .entropy_encoder import EntropyEncoder, Symbol
from typing import List, Union


class IntegerBinaryArithmeticEncoder(EntropyEncoder):
    def __init__(self, symbol_frequencies: dict, enable_conditional_probabilities: bool = False):
        """
        Initializes the Integer Binary Arithmetic Encoder.

        Parameters:
            symbol_frequencies (dict): Dictionary mapping symbols to their frequencies.
        """
        self.original_symbol_frequencies = symbol_frequencies.copy()
        self.symbol_frequencies = self.original_symbol_frequencies.copy()
        self.total = sum(symbol_frequencies.values()) + 1  # Total frequency count
        self.precision = math.ceil(math.log2(4 * self.total))
        self.high = (1 << self.precision) - 1  # Upper bound of range
        self.low = 0  # Lower bound of range
        self.output = []  # Encoded bits
        self.scale = 0  # Tracks scaling operations
        self.enable_conditional_probabilities = enable_conditional_probabilities

        # Build cumulative frequency table
        self.cumulative_frequencies = self._build_cumulative_frequencies()

    def _build_cumulative_frequencies(self):
        """
        Builds cumulative frequency table from symbol frequencies.

        Returns:
            dict: A dictionary mapping symbols to their cumulative frequencies.
        """
        items = sorted(self.symbol_frequencies.items(), key=lambda p: p[1], reverse=True)

        cumulative = {}
        total = 0
        for symbol, freq in items:
            cumulative[symbol] = total
            total += freq
        self.symbol_frequencies["EOF"] = 1
        cumulative["EOF"] = total  # Add an EOF symbol
        return cumulative

    def _update_frequencies(self, symbol):
        """
        Reduce the frequency of a symbol and recalculate cumulative frequencies.

        Parameters:
            symbol (str): The symbol whose frequency is to be reduced.
        """
        # Decrease the frequency of the encoded symbol
        if self.symbol_frequencies[symbol] > 0:
            self.symbol_frequencies[symbol] -= 1
            self.total -= 1

        # Rebuild cumulative frequencies
        self.cumulative_frequencies = self._build_cumulative_frequencies()

    def _output_bit(self, bit):
        """
        Outputs a single bit to the stream.

        Parameters:
            bit (int): The bit to output (0 or 1).
        """
        self.output.append(bit)

    def _scale_bits(self, bit):
        """
        Outputs scaled bits stored during an underflow.

        Parameters:
            bit (int): The bit to output for scaled underflows.
        """
        for _ in range(self.scale):
            self.output.append(bit)
        self.scale = 0

    def _reset_frequencies(self):
        """
        Reset the symbol frequencies to their original values and recalculate cumulative frequencies.
        """
        self.symbol_frequencies = self.original_symbol_frequencies.copy()
        self.cumulative_frequencies = self._build_cumulative_frequencies()
        self.total = sum(self.symbol_frequencies.values()) + 1

    def encode(self, data: List[Symbol]) -> str:
        """
        Encodes input data using binary arithmetic encoding.

        Parameters:
            data (list): List of symbols to encode.

        Returns:
            list: Encoded binary stream.
        """

        self._reset_frequencies()

        # Fixed-point representation with 32-bit integer
        BIT_PRECISION = self.precision
        MAX_VALUE = (1 << BIT_PRECISION) - 1  # 2^32 - 1
        HALF = 1 << (BIT_PRECISION - 1)  # 2^31
        QUARTER = 1 << (BIT_PRECISION - 2)  # 2^30
        THREE_QUARTERS = 3 << (BIT_PRECISION - 2)  # 3 * 2^30

        for symbol in tqdm(data + ['EOF']):
            # Narrow the interval for the current symbol
            symbol_low = self.cumulative_frequencies[symbol]
            symbol_high = symbol_low + self.symbol_frequencies[symbol]
            range_width = self.high - self.low + 1

            self.high = self.low + np.int32(np.int64(range_width) * symbol_high // self.total) - 1
            self.low = self.low + np.int32(np.int64(range_width) * symbol_low // self.total)

            # Normalize and output bits
            while True:
                if self.high < HALF:  # MSB is 0 [0, M/2)
                    self._output_bit(0)
                    self._scale_bits(1)
                elif self.low >= HALF:  # MSB is 1 [M/2, M)
                    self._output_bit(1)
                    self._scale_bits(0)
                    self.low -= HALF
                    self.high -= HALF
                elif self.low >= QUARTER and self.high < THREE_QUARTERS:  # Underflow [M/4, 3M/4)
                    self.scale += 1
                    self.low -= QUARTER
                    self.high -= QUARTER
                else:
                    break
                self.high = (self.high << 1) + 1
                self.low = self.low << 1

            # Dynamically update frequencies
            if self.enable_conditional_probabilities:
                self._update_frequencies(symbol)

        # Finalization
        self.scale += 1
        if self.low < QUARTER:
            self._output_bit(0)
            self._scale_bits(1)
        else:
            self._output_bit(1)
            self._scale_bits(0)

        return ''.join([str(x) for x in self.output])

    def decode(self, encoded_stream, data_length):
        """
        Decodes a binary stream using binary arithmetic decoding.

        Parameters:
            encoded_stream (list): Encoded binary stream.
            data_length (int): Length of original data (number of symbols to decode).

        Returns:
            list: Decoded list of symbols.
        """
        self._reset_frequencies()

        self.high = (1 << self.precision) - 1
        self.low = 0

        output_symbols = []
        range_width = 1 << self.precision
        value = int(''.join(map(str, encoded_stream[:self.precision])), 2)
        stream_index = self.precision

        # for _ in tqdm(range(data_length + 1)):  # Include EOF in symbols
        for _ in tqdm(range(len(encoded_stream) + 1)):
            current_range = self.high - self.low + 1
            freq = np.int32((np.int64(value - self.low + 1) * self.total - 1) // current_range)

            # Find symbol corresponding to cumulative frequency
            symbol = "EOF"
            previous = None
            for s, cumulative in self.cumulative_frequencies.items():
                if cumulative > freq:
                    symbol = previous
                    break
                previous = s

            if symbol == "EOF":  # Stop decoding at EOF
                break

            output_symbols.append(symbol)

            # Narrow range
            symbol_low = self.cumulative_frequencies[symbol]
            symbol_high = symbol_low + self.symbol_frequencies[symbol]

            self.high = self.low + np.int32(np.int64(current_range) * symbol_high // self.total) - 1
            self.low = self.low + np.int32(np.int64(current_range) * symbol_low // self.total)

            BIT_PRECISION = self.precision
            MAX_VALUE = (1 << BIT_PRECISION) - 1
            HALF = 1 << (BIT_PRECISION - 1)
            QUARTER = 1 << (BIT_PRECISION - 2)
            THREE_QUARTERS = 3 << (BIT_PRECISION - 2)

            def get_encoded_value():
                if stream_index >= len(encoded_stream):
                    return 0
                return int(encoded_stream[stream_index])

            # Normalize
            while True:
                if self.high < HALF:  # MSB is 0
                    self.high = (self.high << 1) + 1
                    self.low = self.low << 1
                    value = ((value << 1) | get_encoded_value()) & (range_width - 1)
                    stream_index += 1
                elif self.low >= HALF:  # MSB is 1
                    self.high = ((self.high - HALF) << 1) + 1
                    self.low = (self.low - HALF) << 1
                    value = ((value << 1) | get_encoded_value()) & (range_width - 1)
                    stream_index += 1
                elif self.low >= QUARTER and self.high < THREE_QUARTERS:  # Underflow
                    self.high = ((self.high - QUARTER) << 1) + 1
                    self.low = (self.low - QUARTER) << 1
                    value = (((value - QUARTER) << 1) | get_encoded_value()) & (range_width - 1)
                    stream_index += 1
                else:
                    break

            # Dynamically update frequencies
            if self.enable_conditional_probabilities:
                self._update_frequencies(symbol)

        return output_symbols
