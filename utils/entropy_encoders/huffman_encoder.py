import heapq
from .entropy_encoder import EntropyEncoder, Symbol
from typing import Dict, List, Union


class HuffmanNode:
    def __init__(self, value: Union[Symbol, None] = None, freq=0, left=None, right=None):
        self.value: Union[Symbol, None] = value
        self.freq = freq
        self.left = left
        self.right = right

    def __lt__(self, other):
        return self.freq < other.freq


class HuffmanEncoder(EntropyEncoder):
    def __init__(self, symbol_frequencies: Dict[Symbol, int], enable_conditional_probabilities: bool = False):
        """
        Initializes a Huffman Encoder.

        Parameters:
            symbol_frequencies (dict): Dictionary mapping symbols to their frequencies.
        """
        self.original_symbol_frequencies = symbol_frequencies.copy()
        self.symbol_frequencies = symbol_frequencies.copy()

        self.enable_conditional_probabilities = enable_conditional_probabilities

    def _build_tree(self):
        # Build the Huffman Tree
        heap = [HuffmanNode(value, freq) for value, freq in self.symbol_frequencies.items()]
        heapq.heapify(heap)

        # Combine nodes until there is only one root node
        while len(heap) > 1:
            left = heapq.heappop(heap)
            right = heapq.heappop(heap)
            merged_node = HuffmanNode(freq=left.freq + right.freq, left=left, right=right)
            heapq.heappush(heap, merged_node)
        self.huffman_tree = heap[0] if heap else None

    def _generate_huffman_codes(self):
        # Generate Huffman codes by traversing the tree
        self.huffman_codes = {}

        def generate_codes(node, current_code):
            if node is None:
                return
            if node.value is not None:  # Leaf node, assign code
                self.huffman_codes[node.value] = current_code
            generate_codes(node.left, current_code + "0")
            generate_codes(node.right, current_code + "1")

        generate_codes(self.huffman_tree, "")

    def _update_frequencies(self, symbol: Symbol):
        """
        Reduce the frequency of a symbol and recalculate cumulative frequencies.

        Parameters:
            symbol (str): The symbol whose frequency is to be reduced.
        """
        # Decrease the frequency of the encoded symbol
        if self.symbol_frequencies[symbol] > 0:
            self.symbol_frequencies[symbol] -= 1

        self._build_tree()
        self._generate_huffman_codes()

    def _reset_frequencies(self):
        """
        Reset the symbol frequencies to their original values and recalculate cumulative frequencies.
        """
        self.symbol_frequencies = self.original_symbol_frequencies.copy()
        self._build_tree()
        self._generate_huffman_codes()

    def encode(self, data: List[Symbol]) -> str:
        """
        Encodes a list of symbols using the Huffman codes.

        Parameters:
            data (list): List of symbols to encode.

        Returns:
            str: Encoded binary string.
        """

        self._reset_frequencies()

        encoded_symbols = []
        for symbol in data:
            encoded_symbols.append(self.huffman_codes[symbol])

            if self.enable_conditional_probabilities:
                self._update_frequencies(symbol)

        return ''.join(encoded_symbols)

    def decode(self, encoded_data: str) -> List[Symbol]:
        """
        Decodes a binary string back into the original data using the Huffman tree.

        Parameters:
            encoded_data (str): Binary string to decode.

        Returns:
            list: Decoded list of symbols.
        """
        self._reset_frequencies()

        decoded_data = []
        node = self.huffman_tree

        for bit in encoded_data:
            # Traverse the tree based on the current bit
            if bit == "0":
                node = node.left
            else:
                node = node.right

            # If it's a leaf node, decode the symbol
            if node.value is not None:
                symbol = node.value
                decoded_data.append(symbol)
                node = self.huffman_tree  # Reset to root for the next symbol

                if self.enable_conditional_probabilities:
                    self._update_frequencies(symbol)

        return decoded_data
