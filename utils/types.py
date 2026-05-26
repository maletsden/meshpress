from dataclasses import dataclass, field
from typing import List, Optional, Union
import sys

import numpy as np


@dataclass
class Vertex:
    x: float = 0.0
    y: float = 0.0
    z: float = 0.0

    # Addition
    def __add__(self, other: 'Vertex') -> 'Vertex':
        return Vertex(self.x + other.x, self.y + other.y, self.z + other.z)

    # Subtraction
    def __sub__(self, other: 'Vertex') -> 'Vertex':
        return Vertex(self.x - other.x, self.y - other.y, self.z - other.z)

    # Multiplication (by scalar)
    def __mul__(self, scalar: float) -> 'Vertex':
        return Vertex(self.x * scalar, self.y * scalar, self.z * scalar)

    # Division (by scalar)
    def truediv_by_scalar(self, scalar: float) -> 'Vertex':
        if scalar == 0:
            raise ValueError("Cannot divide by zero.")
        return Vertex(self.x / scalar, self.y / scalar, self.z / scalar)

    # Division (by vertex)
    def truediv_by_vertex(self, other: 'Vertex') -> 'Vertex':
        if (other.x == 0) or (other.y == 0) or (other.z == 0):
            raise ValueError("Cannot divide by zero.")
        return Vertex(self.x / other.x, self.y / other.y, self.z / other.z)

    # Division
    def __truediv__(self, other: Union[float, 'Vertex']) -> 'Vertex':
        if type(other) == float:
            return self.truediv_by_scalar(other)
        elif type(other) == type(self):
            return self.truediv_by_vertex(other)
        raise TypeError("Provided type is not supported.")

    # Equality
    def __eq__(self, other: 'Vertex') -> bool:
        return self.x == other.x and self.y == other.y and self.z == other.z

    # Not equal
    def __ne__(self, other: 'Vertex') -> bool:
        return not self.__eq__(other)

    # Negation (unary -)
    def __neg__(self) -> 'Vertex':
        return Vertex(-self.x, -self.y, -self.z)

    # Absolute value (distance from origin)
    def __abs__(self) -> float:
        return (self.x ** 2 + self.y ** 2 + self.z ** 2) ** 0.5


@dataclass
class Triangle:
    a: int = 0
    b: int = 0
    c: int = 0


# Axis-Aligned Bounding Box
@dataclass
class AABB:
    min: Vertex = field(default_factory=Vertex)
    max: Vertex = field(default_factory=Vertex)

    # Intersection (Check if two AABBs overlap)
    def __and__(self, other: 'AABB') -> bool:
        return (
                self.max.x >= other.min.x and self.min.x <= other.max.x and
                self.max.y >= other.min.y and self.min.y <= other.max.y and
                self.max.z >= other.min.z and self.min.z <= other.max.z
        )

    # Union (Create a new AABB that encloses both)
    def __or__(self, other: 'AABB') -> 'AABB':
        new_min = Vertex(
            min(self.min.x, other.min.x),
            min(self.min.y, other.min.y),
            min(self.min.z, other.min.z)
        )
        new_max = Vertex(
            max(self.max.x, other.max.x),
            max(self.max.y, other.max.y),
            max(self.max.z, other.max.z)
        )
        return AABB(new_min, new_max)

    # Containment (Check if a point is within the AABB)
    def __contains__(self, point: Vertex) -> bool:
        return (
                self.min.x <= point.x <= self.max.x and
                self.min.y <= point.y <= self.max.y and
                self.min.z <= point.z <= self.max.z
        )

    # Expand (Expand the AABB to include a point or another AABB)
    def expand(self, other: Union[Vertex, 'AABB']) -> None:
        if isinstance(other, Vertex):
            self.min = Vertex(
                min(self.min.x, other.x),
                min(self.min.y, other.y),
                min(self.min.z, other.z)
            )
            self.max = Vertex(
                max(self.max.x, other.x),
                max(self.max.y, other.y),
                max(self.max.z, other.z)
            )
        elif isinstance(other, AABB):
            self.min = Vertex(
                min(self.min.x, other.min.x),
                min(self.min.y, other.min.y),
                min(self.min.z, other.min.z)
            )
            self.max = Vertex(
                max(self.max.x, other.max.x),
                max(self.max.y, other.max.y),
                max(self.max.z, other.max.z)
            )

    # Equality
    def __eq__(self, other: 'AABB') -> bool:
        return self.min == other.min and self.max == other.max

    # Size (returns the width, height, depth as a Vertex)
    def size(self) -> Vertex:
        return self.max - self.min


class Model:
    """Mesh container. Numpy arrays are primary storage; list-of-Vertex and
    list-of-Triangle views are lazily built (and cached) for back-compat
    with legacy encoders.

    Construct via `Model.from_arrays(verts, tris)` or `Model.from_obj(path)`.
    The two-positional-args constructor `Model(vertices, triangles)` accepts
    either numpy arrays or `List[Vertex]` / `List[Triangle]` so existing
    callers (e.g. `Model([], [])`) keep working unchanged.

    Mutating the cached list views (`model.vertices.append(...)`) does NOT
    propagate back to the numpy storage. Lists are read-only by convention.
    """

    __slots__ = ("vertices_np", "triangles_np", "triangle_strips",
                 "_vertices_cache", "_triangles_cache")

    def __init__(self,
                 vertices: Optional[Union[np.ndarray, List["Vertex"]]] = None,
                 triangles: Optional[Union[np.ndarray, List["Triangle"]]] = None,
                 triangle_strips: Optional[List[List[int]]] = None):
        self.vertices_np = self._coerce_verts(vertices)
        self.triangles_np = self._coerce_tris(triangles)
        self.triangle_strips = list(triangle_strips) if triangle_strips else []
        self._vertices_cache = None
        self._triangles_cache = None

    @staticmethod
    def _coerce_verts(v) -> np.ndarray:
        if v is None or (isinstance(v, list) and len(v) == 0):
            return np.zeros((0, 3), dtype=np.float64)
        if isinstance(v, np.ndarray):
            arr = np.ascontiguousarray(v, dtype=np.float64)
            if arr.ndim != 2 or arr.shape[1] != 3:
                raise ValueError(f"vertices ndarray must be (N, 3); got {arr.shape}")
            return arr
        return np.asarray([[p.x, p.y, p.z] for p in v], dtype=np.float64)

    @staticmethod
    def _coerce_tris(t) -> np.ndarray:
        if t is None or (isinstance(t, list) and len(t) == 0):
            return np.zeros((0, 3), dtype=np.int64)
        if isinstance(t, np.ndarray):
            arr = np.ascontiguousarray(t, dtype=np.int64)
            if arr.ndim != 2 or arr.shape[1] != 3:
                raise ValueError(f"triangles ndarray must be (M, 3); got {arr.shape}")
            return arr
        return np.asarray([[tr.a, tr.b, tr.c] for tr in t], dtype=np.int64)

    @property
    def vertices(self) -> List["Vertex"]:
        if self._vertices_cache is None:
            self._vertices_cache = [
                Vertex(float(x), float(y), float(z))
                for x, y, z in self.vertices_np
            ]
        return self._vertices_cache

    @property
    def triangles(self) -> List["Triangle"]:
        if self._triangles_cache is None:
            self._triangles_cache = [
                Triangle(int(a), int(b), int(c))
                for a, b, c in self.triangles_np
            ]
        return self._triangles_cache

    @classmethod
    def from_arrays(cls, verts: np.ndarray, tris: np.ndarray) -> "Model":
        return cls(verts, tris)

    @classmethod
    def from_obj(cls, path: str) -> "Model":
        from reader.fast_obj import load_mesh_npy
        v, t = load_mesh_npy(path)
        return cls(v, t)

    @property
    def aabb(self) -> "AABB":
        if len(self.vertices_np) == 0:
            return AABB(Vertex(), Vertex())
        lo = self.vertices_np.min(axis=0)
        hi = self.vertices_np.max(axis=0)
        return AABB(Vertex(float(lo[0]), float(lo[1]), float(lo[2])),
                    Vertex(float(hi[0]), float(hi[1]), float(hi[2])))

    def copy(self) -> "Model":
        return Model(self.vertices_np.copy(),
                     self.triangles_np.copy(),
                     list(self.triangle_strips))

@dataclass
class CompressedModel:
    data: bytes
    bits_per_vertex: float
    bits_per_triangle: float
