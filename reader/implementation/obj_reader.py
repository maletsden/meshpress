from ..reader import Reader
from utils.types import Model


class OBJReader(Reader):
    """OBJ reader. Delegates to the numpy-fast loader and returns a
    numpy-backed Model. Vertex/triangle order matches the OBJ file
    (required for bit-exact BPV vs published reference numbers)."""

    def __init__(self):
        pass

    def read(self, path: str) -> Model:
        return Model.from_obj(path)
