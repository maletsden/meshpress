import heapq
from collections import Counter


class HuffmanNode:
    def __init__(self, value=None, freq=0, left=None, right=None):
        self.value = value
        self.freq = freq
        self.left = left
        self.right = right

    def __lt__(self, other):
        return self.freq < other.freq


def build_huffman_tree(frequencies):
    heap = [HuffmanNode(value, freq) for value, freq in frequencies.items()]
    heapq.heapify(heap)

    while len(heap) > 1:
        left = heapq.heappop(heap)
        right = heapq.heappop(heap)
        merged = HuffmanNode(freq=left.freq + right.freq, left=left, right=right)
        heapq.heappush(heap, merged)

    return heap[0] if heap else None


def build_huffman_codes(node, prefix="", codes=None):
    if codes is None:
        codes = {}

    if node is not None:
        if node.value is not None:
            codes[node.value] = prefix
        build_huffman_codes(node.left, prefix + "0", codes)
        build_huffman_codes(node.right, prefix + "1", codes)

    return codes


def calculate_huffman_codes(data):
    # Calculate frequencies
    frequencies = Counter(data)

    # Build Huffman Tree
    root = build_huffman_tree(frequencies)

    # Build Huffman Codes
    huffman_codes = build_huffman_codes(root)

    return frequencies, huffman_codes
