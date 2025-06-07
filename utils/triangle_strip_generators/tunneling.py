from dataclasses import dataclass, field
from typing import List, Dict, Tuple, Optional
from collections import deque, namedtuple
from tqdm import tqdm, trange

from ..types import Vertex, Triangle

@dataclass
class TriangleMesh:
    vertices: List[Vertex]
    triangles: List[Triangle]
    tri_to_edges: Dict[int, List[int]] = field(default_factory=dict)
    edge_to_tris: Dict[int, List[int]] = field(default_factory=dict)
    edge_to_vertices: Dict[int, Tuple[int, int]] = field(default_factory=dict)

    def build_adjacency(self) -> None:
        self.tri_to_edges.clear()
        self.edge_to_tris.clear()
        self.edge_to_vertices.clear()
        edge_map: Dict[Tuple[int, int], int] = {}
        next_eid = 0
        for tidx, tri in enumerate(self.triangles):
            eids: List[int] = []
            for u, v in ((tri.a, tri.b), (tri.b, tri.c), (tri.c, tri.a)):
                key = (min(u, v), max(u, v))
                if key not in edge_map:
                    edge_map[key] = next_eid
                    self.edge_to_tris[next_eid] = []
                    self.edge_to_vertices[next_eid] = key
                    next_eid += 1
                eid = edge_map[key]
                eids.append(eid)
                self.edge_to_tris[eid].append(tidx)
            self.tri_to_edges[tidx] = eids

@dataclass
class DualGraph:
    num_triangles: int
    adj: Dict[int, List[Tuple[int, int]]] = field(default_factory=dict)
    is_strip_edge: Dict[int, bool] = field(default_factory=dict)

    @classmethod
    def from_mesh(cls, mesh: TriangleMesh) -> 'DualGraph':
        dg = cls(num_triangles=len(mesh.triangles))
        dg.adj = {t: [] for t in range(dg.num_triangles)}
        for eid, tris in mesh.edge_to_tris.items():
            dg.is_strip_edge[eid] = False
            for i in range(len(tris)):
                for j in range(i + 1, len(tris)):
                    t0, t1 = tris[i], tris[j]
                    dg.adj[t0].append((t1, eid))
                    dg.adj[t1].append((t0, eid))
        return dg

    def mark_strip(self, eid: int) -> None:
        self.is_strip_edge[eid] = True

    def unmark_strip(self, eid: int) -> None:
        self.is_strip_edge[eid] = False

    def incident_strip_count(self, tri: int) -> int:
        return sum(self.is_strip_edge[eid] for _, eid in self.adj[tri])

BFSState = namedtuple('BFSState', ['tri', 'parity', 'prev_eid', 'prev'])

class TunnelFinder:
    def __init__(self, graph: DualGraph):
        self.graph = graph

    def find_tunnel(self, start: int, max_depth: int = 100) -> Optional[List[int]]:
        if self.graph.incident_strip_count(start) != 0:
            return None
        visited = {(start, 0)}
        queue = deque([BFSState(start, 0, None, None)])
        while queue:
            state = queue.popleft()
            if len(self._reconstruct(state)) > max_depth:
                continue
            for nbr, eid in self.graph.adj[state.tri]:
                want_strip = (state.parity == 1)
                if self.graph.is_strip_edge[eid] != want_strip:
                    continue
                nxt = BFSState(nbr, 1 - state.parity, eid, state)
                key = (nbr, nxt.parity)
                if key in visited:
                    continue
                visited.add(key)
                # if nxt.parity == 1 and self.graph.incident_strip_count(nbr) == 0 and nbr != start:
                if nxt.parity == 1 and self._is_terminal(nbr) and nbr != start:
                    return self._reconstruct(nxt)
                queue.append(nxt)
        return None

    def _reconstruct(self, state: BFSState) -> List[int]:
        path = []
        while state.prev_eid is not None:
            path.append(state.prev_eid)
            state = state.prev
        return list(reversed(path))

    def _is_terminal(self, tri: int) -> bool:
        return self.graph.incident_strip_count(tri) <= 1

class TunnelingStripifier:
    def __init__(self, mesh: TriangleMesh, run_self_check: bool = False):
        self.mesh = mesh
        self.run_self_check = run_self_check
        self.mesh.build_adjacency()
        self.graph: DualGraph = DualGraph.from_mesh(mesh)
        self._init()
        self.tfinder = TunnelFinder(self.graph)
        self.terminals = {t for t in range(self.graph.num_triangles)
                          if self.graph.incident_strip_count(t) <= 1}

    def _init(self) -> None:
        for eid in self.graph.is_strip_edge:
            self.graph.unmark_strip(eid)

    def run(self, iters: int = 1000) -> None:
        for _ in trange(iters):
            moved = False
            for st in list(self.terminals):
                tun = self.tfinder.find_tunnel(st)
                if tun:
                    self._apply(tun)
                    moved = True
                    break
            if not moved:
                break

        if self.run_self_check:
            tstrips = self.extract_triangle_strips()
            self.check_connectivity(tstrips)

    def _apply(self, path: List[int]) -> None:
        affected = set()
        for eid in path:
            if self.graph.is_strip_edge[eid]:
                self.graph.unmark_strip(eid)
            else:
                self.graph.mark_strip(eid)
            affected.update(self.mesh.edge_to_tris[eid])
        for tri in affected:
            if self.graph.incident_strip_count(tri) <= 1:
                self.terminals.add(tri)
            else:
                self.terminals.discard(tri)

    def extract_triangle_strips(self) -> List[List[int]]:
        visited = set()
        strips: List[List[int]] = []
        for t in range(self.graph.num_triangles):
            if t in visited:
                continue
            deg = self.graph.incident_strip_count(t)
            if deg > 1:
                continue
            strip = [t]
            visited.add(t)
            prev = None
            cur = t
            while True:
                moves = [(nbr, eid) for nbr, eid in self.graph.adj[cur]
                         if self.graph.is_strip_edge[eid] and nbr != prev]
                if len(moves) != 1:
                    break
                nbr, eid = moves[0]
                if nbr in visited:
                    break
                strip.append(nbr)
                visited.add(nbr)
                prev, cur = cur, nbr
            strips.append(strip)
        for t in range(self.graph.num_triangles):
            if t not in visited:
                strips.append([t])
        return strips


    def check_connectivity(self, tstrips: List[List[int]]) -> bool:
        original = {tuple(sorted((tri.a, tri.b, tri.c))) for tri in self.mesh.triangles}
        reconstructed = set()
        for strip in tstrips:
            for tidx in strip:
                tri = self.mesh.triangles[tidx]
                reconstructed.add(tuple(sorted((tri.a, tri.b, tri.c))))
        if original != reconstructed:
            missing = original - reconstructed
            extra = reconstructed - original
            print(f"Connectivity mismatch! Missing: {missing}, Extra: {extra}")
            return False
        print("Connectivity stays lossless: all triangles recovered.")
        return True

    def extract_vertex_strips(self) -> List[List[int]]:
        vert_strips: List[List[int]] = []
        for strip in self.extract_triangle_strips():
            if not strip:
                continue
            first_tri = self.mesh.triangles[strip[0]]
            vs = [first_tri.a, first_tri.b, first_tri.c]
            for tidx in strip[1:]:
                tri = self.mesh.triangles[tidx]
                last2 = set(vs[-2:])
                for v in (tri.a, tri.b, tri.c):
                    if v not in last2:
                        vs.append(v)
                        break
            vert_strips.append(vs)
        return vert_strips
