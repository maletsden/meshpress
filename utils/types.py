from dataclasses import dataclass, field
from typing import List, Union
import sys


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


@dataclass
class Model:
    vertices: List[Vertex] = field(default_factory=list)
    triangles: List[Triangle] = field(default_factory=list)

    @property
    def aabb(self) -> AABB:
        aabb: AABB = AABB()
        aabb.min.x = sys.float_info.max
        aabb.min.y = sys.float_info.max
        aabb.min.z = sys.float_info.max
        aabb.max.x = sys.float_info.min
        aabb.max.y = sys.float_info.min
        aabb.max.z = sys.float_info.min

        for vertex in self.vertices:
            aabb.min.x = min(vertex.x, aabb.min.x)
            aabb.min.y = min(vertex.y, aabb.min.y)
            aabb.min.z = min(vertex.z, aabb.min.z)
            aabb.max.x = max(vertex.x, aabb.max.x)
            aabb.max.y = max(vertex.y, aabb.max.y)
            aabb.max.z = max(vertex.z, aabb.max.z)

        return aabb


@dataclass
class CompressedModel:
    data: bytes
    bits_per_vertex: float
    bits_per_triangle: float
