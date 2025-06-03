from utils.types import *
from ..encoder import Encoder, Packing
import struct
from utils.bit_magic import *
from utils.geometry import triangle_list_to_strip

import numpy as np
from collections import Counter
import torch
import math


def calculate_remapping(strip, codes):
    # Step 1: Count the frequency of each integer in the array
    frequency = Counter(strip)

    # Step 2: Sort integers by frequency in descending order
    sorted_by_frequency = sorted(frequency.keys(), key=lambda x: -frequency[x])

    # Step 3: Sort codes by length in ascending order
    sorted_codes = {i: code for i, code in enumerate(codes)}
    sorted_codes = sorted(sorted_codes.items(), key=lambda item: len(item[1]))
    sorted_code_keys = [key for key, _ in sorted_codes]

    # Step 4: Create a remap dictionary by mapping each integer to the shortest available code
    indices_remap = {original: new for original, new in zip(sorted_by_frequency, sorted_code_keys)}

    return indices_remap


def reorder_vertices(vertices, triangle_strip, indices_remap):
    indices_remapping_inverse = {new: old for old, new in indices_remap.items()}
    reordered_vertices = [vertices[indices_remapping_inverse[i]] for i in range(len(vertices))]
    reordered_triangle_strip = [indices_remap[v] for v in triangle_strip]

    return reordered_vertices, reordered_triangle_strip


class SimpleEllipsoidFitter(Encoder):

    def __init__(self, vertex_quantization_error: float = 0.0005, max_hierarchical_fitting_depth: int = 3,
                 pack_strip: Packing = Packing.RADIX_BINARY_TREE,
                 pack_vertices: Packing = Packing.FIXED,
                 allow_reorder=False,
                 verbose=False):
        self.vertex_quantization_error: float = vertex_quantization_error
        self.max_hierarchical_fitting_depth: int = max_hierarchical_fitting_depth
        self.pack_strip: Packing = pack_strip
        self.pack_vertices: Packing = pack_vertices
        self.verbose = verbose
        self.allow_reorder = allow_reorder

    class EllipsoidNode:
        """
        Node for hierarchical ellipsoid fit.
        Attributes:
          center:   (3,) fitted center
          axes:     (3,) fitted semi‐axes
          residuals:(N_subset, 3) residual vectors
          points:   (N_subset, 3) original subset
          children: [] or [child0, child1]
        """

        def __init__(self, center, axes, residuals, points):
            self.center = center
            self.axes = axes
            self.residuals = residuals
            self.points = points
            self.children = []

    def encode(self, model: Model) -> CompressedModel:
        triangle_strip = triangle_list_to_strip([[t.a, t.b, t.c] for t in model.triangles])
        vertices = model.vertices

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
            print("SimpleQuantizator verbose statistics:")
            print(f"- Header (in bytes): {len(byte_array)}")

        len_before_triangle_strip = len(byte_array)

        match self.pack_strip:
            case Packing.NONE:
                for index in triangle_strip:
                    byte_array.extend(struct.pack('I', index))
            case Packing.FIXED:
                extend_bytearray_with_fixed_size_values(byte_array, int(np.ceil(np.log2(len(model.vertices)))),
                                                        triangle_strip)
            case Packing.BINARY_RANGE_PARTITIONING:
                codes = calculate_codes_using_binary_range_partitioning(len(model.vertices) - 1)

                if self.allow_reorder:
                    indices_remapping = calculate_remapping(triangle_strip, codes)
                    vertices, triangle_strip = reorder_vertices(vertices, triangle_strip, indices_remapping)

                bit_codes = [codes[v] for v in triangle_strip]
                extend_bytearray_with_bit_codes(byte_array, bit_codes)
            case Packing.RADIX_BINARY_TREE:
                codes = calculate_codes_using_binary_radix_tree(len(model.vertices) - 1)

                if self.allow_reorder:
                    indices_remapping = calculate_remapping(triangle_strip, codes)
                    vertices, triangle_strip = reorder_vertices(vertices, triangle_strip, indices_remapping)

                bit_codes = [codes[v] for v in triangle_strip]
                extend_bytearray_with_bit_codes(byte_array, bit_codes)

        if self.verbose:
            print(f"- Triangles strip entropy: {calculate_entropy(triangle_strip):.3f}")
            bytes_used_for_strip = len(byte_array) - len_before_triangle_strip
            print(f"- Triangles strip average code bit length: {bytes_used_for_strip * 8 / len(triangle_strip):.3f}")
            print(f"- Triangles strip (in bytes): {bytes_used_for_strip}")

        len_before_vertices = len(byte_array)
        ellipsoid_node = self.fit_ellipsoids(vertices, self.max_hierarchical_fitting_depth)
        self.write_ellipsoids_header(ellipsoid_node, byte_array)
        self.write_quantize_ellipsoid_residuals(ellipsoid_node, model, byte_array)

        if self.verbose:
            # print(f"- Quantized vertices entropy: {calculate_entropy(quantized_vertices)}")
            bytes_used_for_vertices = len(byte_array) - len_before_vertices
            print(
                f"- Quantized vertices average code bit length: {bytes_used_for_vertices * 8 / len(model.vertices):.3f}")
            print(f"- Quantized vertices (in bytes): {bytes_used_for_vertices}")

        bits_per_vertex = len(byte_array) * 8 / len(model.vertices)
        bits_per_triangle = len(byte_array) * 8 / len(model.triangles)

        return CompressedModel(bytes(byte_array), bits_per_vertex, bits_per_triangle)

    def write_ellipsoids_header(self, ellipsoid_node: EllipsoidNode, byte_array: bytearray):
        ellipsoid_node_depth = 3
        byte_array.extend(struct.pack('I', ellipsoid_node_depth))

        def recurse(root: SimpleEllipsoidFitter.EllipsoidNode):
            byte_array.extend(struct.pack('f', root.center[0]))
            byte_array.extend(struct.pack('f', root.center[1]))
            byte_array.extend(struct.pack('f', root.center[2]))

            byte_array.extend(struct.pack('f', root.axes[0]))
            byte_array.extend(struct.pack('f', root.axes[1]))
            byte_array.extend(struct.pack('f', root.axes[2]))

            for child in root.children:
                recurse(child)

        recurse(ellipsoid_node)

    def write_quantize_ellipsoid_residuals(self, ellipsoid_node: EllipsoidNode, model: Model, byte_array: bytearray):
        leafs: list[SimpleEllipsoidFitter.EllipsoidNode] = []
        self.get_tree_leafs(ellipsoid_node, leafs)

        original_model_size = model.aabb.size()
        for leaf in leafs:
            mins_res = leaf.residuals.min(axis=0)
            maxs_res = leaf.residuals.max(axis=0)
            aabb = AABB(Vertex(mins_res[0], mins_res[1], mins_res[2]), Vertex(maxs_res[0], maxs_res[1], maxs_res[2]))
            aabb_size = aabb.size()
            bits_needed_x = int(
                math.ceil(math.log2(aabb_size.x / (original_model_size.x * self.vertex_quantization_error))))
            bits_needed_y = int(
                math.ceil(math.log2(aabb_size.y / (original_model_size.y * self.vertex_quantization_error))))
            bits_needed_z = int(
                math.ceil(math.log2(aabb_size.z / (original_model_size.z * self.vertex_quantization_error))))

            N = len(leaf.residuals)
            byte_array.extend(struct.pack('I', N))
            byte_array.extend(struct.pack('I', bits_needed_x))
            byte_array.extend(struct.pack('I', bits_needed_y))
            byte_array.extend(struct.pack('I', bits_needed_z))
            byte_array.extend(struct.pack('f', aabb.min.x))
            byte_array.extend(struct.pack('f', aabb.min.y))
            byte_array.extend(struct.pack('f', aabb.min.z))
            byte_array.extend(struct.pack('f', aabb.max.x))
            byte_array.extend(struct.pack('f', aabb.max.y))
            byte_array.extend(struct.pack('f', aabb.max.z))

            normalized_residuals = (leaf.residuals - mins_res) / (maxs_res - mins_res)
            data_x = [np.uint32(v[0] * (2 ** bits_needed_x - 1)) for v in normalized_residuals]
            data_y = [np.uint32(v[1] * (2 ** bits_needed_y - 1)) for v in normalized_residuals]
            data_z = [np.uint32(v[2] * (2 ** bits_needed_z - 1)) for v in normalized_residuals]
            extend_bytearray_with_fixed_size_values(byte_array, bits_needed_x, data_x)
            extend_bytearray_with_fixed_size_values(byte_array, bits_needed_y, data_y)
            extend_bytearray_with_fixed_size_values(byte_array, bits_needed_z, data_z)

    def get_tree_leafs(self, root: EllipsoidNode, leafs: list[EllipsoidNode]):
        is_leaf = len(root.children) == 0

        if is_leaf:
            leafs.append(root)
        else:
            for child in root.children:
                self.get_tree_leafs(child, leafs)

    def fit_ellipsoids(self, vertices: List[Vertex],
                       max_depth: int = 3) -> EllipsoidNode:
        """
            Hierarchically fit ellipsoids to `points` up to `max_depth` levels,
            using nonlinear least‐squares at each node.

            Returns the root EllipsoidNode.
            """

        points = np.array([[v.x, v.y, v.z] for v in vertices], dtype=np.float32)

        def recurse(sub_pts: np.ndarray, depth: int) -> SimpleEllipsoidFitter.EllipsoidNode:
            # 1) Nonlinear ellipsoid fit
            center, axes = self.fit_axis_aligned_ellipsoid_nn(sub_pts, num_iters=2000)
            # 2) Compute residuals
            residuals = self.compute_ellipsoid_residuals(sub_pts, center, axes)
            node = self.EllipsoidNode(center, axes, residuals, sub_pts)

            # 3) Split if not at max depth
            if depth < max_depth and sub_pts.shape[0] > 10:
                pts0, pts1 = self.split_points_along_longest_axis(residuals, center, axes)
                if pts0.shape[0] > 0 and pts1.shape[0] > 0:
                    child0 = recurse(pts0, depth + 1)
                    child1 = recurse(pts1, depth + 1)
                    node.children = [child0, child1]
            return node

        return recurse(points, depth=1)

    def fit_axis_aligned_ellipsoid_nn(
            self,
            points: np.ndarray,
            num_iters: int = 1000,
            lr: float = 1e-2,
            device: str = "cpu"
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Fit an axis‐aligned ellipsoid to `points` by treating (x0,y0,z0,a,b,c) as
        learnable parameters in PyTorch and minimizing ∑[( (x−x0)/a )^2 + ((y−y0)/b)^2 + ((z−z0)/c)^2 − 1 ]^2
        via gradient descent.

        Returns:
            center: np.ndarray of shape (3,)  -- learned [x0, y0, z0]
            axes:   np.ndarray of shape (3,)  -- learned [a, b, c] (all > 0)
        """
        # Convert points to torch tensor on the chosen device
        pts = torch.from_numpy(points.astype(np.float32)).to(device)  # (N,3)

        # Initialize parameters:
        # center (x0,y0,z0) = centroid of points
        centroid = points.mean(axis=0).astype(np.float32)
        # axes (a,b,c) = half‐ranges in x,y,z
        mins = points.min(axis=0).astype(np.float32)
        maxs = points.max(axis=0).astype(np.float32)
        half_extents = ((maxs - mins) / 2.0).astype(np.float32)
        # Prevent zero initial axes:
        a0, b0, c0 = np.maximum(half_extents, 1e-3)

        # Create torch parameters with requires_grad=True
        x0 = torch.tensor(centroid[0], requires_grad=True, dtype=torch.float32, device=device)
        y0 = torch.tensor(centroid[1], requires_grad=True, dtype=torch.float32, device=device)
        z0 = torch.tensor(centroid[2], requires_grad=True, dtype=torch.float32, device=device)
        # To enforce positivity of a,b,c, we parametrize them via softplus of unconstrained variables:
        sa = torch.tensor(np.log(np.exp(a0) - 1.0), requires_grad=True, dtype=torch.float32, device=device)
        sb = torch.tensor(np.log(np.exp(b0) - 1.0), requires_grad=True, dtype=torch.float32, device=device)
        sc = torch.tensor(np.log(np.exp(c0) - 1.0), requires_grad=True, dtype=torch.float32, device=device)

        # Build optimizer over all parameters
        optimizer = torch.optim.Adam([x0, y0, z0, sa, sb, sc], lr=lr)

        for it in range(num_iters):
            optimizer.zero_grad()

            # Recover actual a,b,c via softplus to keep them > 0
            a = torch.nn.functional.softplus(sa)
            b = torch.nn.functional.softplus(sb)
            c = torch.nn.functional.softplus(sc)

            # Center points
            xc = (pts[:, 0] - x0) / a
            yc = (pts[:, 1] - y0) / b
            zc = (pts[:, 2] - z0) / c

            # Ellipsoid residual: f_i = xc^2 + yc^2 + zc^2 - 1
            f = xc * xc + yc * yc + zc * zc - 1.0  # (N,)

            # Loss = mean squared residual
            loss = torch.mean(f * f)

            loss.backward()
            optimizer.step()

            # # Optional: print every 200 iterations
            # if (it + 1) % 200 == 0 or it == 0:
            #     with torch.no_grad():
            #         print(f"Iter {it + 1}/{num_iters}, Loss = {loss.item():.6e}, "
            #               f"a={a.item():.4f}, b={b.item():.4f}, c={c.item():.4f}")

        # After optimization, extract final center and axes
        with torch.no_grad():
            center = torch.stack((x0, y0, z0)).cpu().numpy()
            a = torch.nn.functional.softplus(sa).cpu().numpy()
            b = torch.nn.functional.softplus(sb).cpu().numpy()
            c = torch.nn.functional.softplus(sc).cpu().numpy()
            axes = np.array([a, b, c], dtype=np.float32)

        return center, axes

    def compute_ellipsoid_residuals(
            self,
            points: np.ndarray,
            center: np.ndarray,
            axes: np.ndarray
    ) -> np.ndarray:
        """
        For each point p, compute the vector residual to the fitted ellipsoid:
          1) p_c = p − center
          2) denom = sqrt((x_c/a)^2 + (y_c/b)^2 + (z_c/c)^2)
             if denom > ε: t = 1/denom else t = 1
          3) p_surf = t * p_c
          4) residual = p_surf − p_c
        Returns array of shape (N,3).
        """
        pts_c = points - center.reshape(1, 3)
        a, b, c = axes
        x_c = pts_c[:, 0]
        y_c = pts_c[:, 1]
        z_c = pts_c[:, 2]

        denom = np.sqrt((x_c / a) ** 2 + (y_c / b) ** 2 + (z_c / c) ** 2).reshape(-1, 1)
        eps = 1e-12
        t = np.zeros_like(denom)
        nonzero = (denom.reshape(-1) > eps)
        t[nonzero] = 1.0 / denom[nonzero]
        t[~nonzero] = 1.0

        p_surf = pts_c * t
        residuals = p_surf - pts_c
        return residuals

    def split_points_along_longest_axis(
            self,
            points: np.ndarray,
            center: np.ndarray,
            axes: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Split `points` into two subsets along the ellipsoid’s longest axis:
          1) idx = argmax(axes)
          2) split_val = center[idx]
          3) pts0: points[:,idx] < split_val; pts1: points[:,idx] >= split_val
        """
        idx = np.argmax(axes)
        coord = points[:, idx]
        split_val = (coord.max() + coord.min()) / 2.0
        mask0 = coord < split_val
        return points[mask0], points[~mask0]
