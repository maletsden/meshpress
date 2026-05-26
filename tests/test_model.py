import numpy as np
import pytest

from utils.types import Model, Vertex, Triangle, AABB


def test_model_from_arrays_stores_numpy():
    v = np.array([[0.0, 1.0, 2.0], [3.0, 4.0, 5.0]], dtype=np.float64)
    t = np.array([[0, 1, 0]], dtype=np.int64)
    m = Model.from_arrays(v, t)
    assert isinstance(m.vertices_np, np.ndarray)
    assert m.vertices_np.shape == (2, 3)
    assert m.vertices_np.dtype == np.float64
    assert m.triangles_np.shape == (1, 3)
    assert m.triangles_np.dtype == np.int64


def test_model_legacy_list_iteration():
    v = np.array([[0.0, 1.0, 2.0], [3.0, 4.0, 5.0]], dtype=np.float64)
    t = np.array([[0, 1, 0]], dtype=np.int64)
    m = Model.from_arrays(v, t)
    assert isinstance(m.vertices, list)
    assert isinstance(m.vertices[0], Vertex)
    assert m.vertices[0].x == 0.0 and m.vertices[1].z == 5.0
    assert isinstance(m.triangles[0], Triangle)
    assert m.triangles[0].a == 0 and m.triangles[0].b == 1


def test_model_legacy_constructor_empty():
    m = Model([], [])
    assert m.vertices_np.shape == (0, 3)
    assert m.triangles_np.shape == (0, 3)
    assert m.vertices_np.dtype == np.float64
    assert m.triangles_np.dtype == np.int64


def test_model_legacy_constructor_from_lists():
    verts = [Vertex(0.0, 1.0, 2.0), Vertex(3.0, 4.0, 5.0)]
    tris = [Triangle(0, 1, 0)]
    m = Model(verts, tris)
    assert m.vertices_np.shape == (2, 3)
    assert m.vertices_np[1, 2] == 5.0
    assert m.triangles_np.shape == (1, 3)
    assert m.triangles_np[0, 1] == 1


def test_model_aabb_uses_numpy():
    v = np.array([[0.0, -1.0, 2.0], [3.0, 4.0, -5.0]], dtype=np.float64)
    t = np.array([[0, 1, 0]], dtype=np.int64)
    m = Model.from_arrays(v, t)
    box = m.aabb
    assert isinstance(box, AABB)
    assert box.min.x == 0.0 and box.min.y == -1.0 and box.min.z == -5.0
    assert box.max.x == 3.0 and box.max.y == 4.0 and box.max.z == 2.0


def test_model_copy_independent():
    v = np.array([[0.0, 1.0, 2.0]], dtype=np.float64)
    t = np.array([[0, 0, 0]], dtype=np.int64)
    m = Model.from_arrays(v, t)
    m2 = m.copy()
    m2.vertices_np[0, 0] = 99.0
    assert m.vertices_np[0, 0] == 0.0


def test_obj_reader_matches_fast_obj():
    from pathlib import Path
    from reader import Reader
    from reader.fast_obj import load_mesh_npy

    asset = Path("assets/stanford-bunny.obj")
    if not asset.exists():
        pytest.skip("stanford-bunny.obj not present")
    m = Reader.read_from_file(str(asset))
    v_ref, t_ref = load_mesh_npy(str(asset))
    assert np.array_equal(m.vertices_np, v_ref)
    assert np.array_equal(m.triangles_np, t_ref)
