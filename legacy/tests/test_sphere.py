import numpy as np
import pyvista as pv
from skimage import measure

# --- 1. The Math: SDF & Projection ---

def smin(a, b, k):
    """Polynomial Smooth Minimum (The 'Blob' function)."""
    h = np.maximum(k - np.abs(a - b), 0.0) / k
    return np.minimum(a, b) - h * h * h * k * (1.0 / 6.0)

def sdf_sphere(p, center, radius):
    return np.linalg.norm(p - center, axis=1) - radius

class BlobModel:
    def __init__(self):
        self.res = 50  # Grid resolution (higher = smoother but slower)
        limit = 3.0
        x = np.linspace(-limit, limit, self.res)
        y = np.linspace(-limit, limit, self.res)
        z = np.linspace(-limit, limit, self.res)
        self.X, self.Y, self.Z = np.meshgrid(x, y, z, indexing='ij')
        self.grid_points = np.stack([self.X.ravel(), self.Y.ravel(), self.Z.ravel()], axis=1)

        # Fake "Original Mesh" vertices to demonstrate projection
        # We put them slightly outside where the blob will be
        theta = np.linspace(0, 2*np.pi, 20)
        self.targets = np.column_stack([
            2.2 * np.cos(theta),
            np.zeros_like(theta),
            2.2 * np.sin(theta)
        ])

    def evaluate(self, sep, scale, smooth):
        """
        Calculates the Signed Distance Field for the hierarchy.
        Returns: Distance field array
        """
        # Level 0: Root Sphere (Center)
        d_root = sdf_sphere(self.grid_points, np.array([0,0,0]), 1.0)

        # Level 1: Two Children (Left and Right)
        # Note: In a real compressor, these positions would be optimized.
        c1_pos = np.array([sep, 0, 0])
        c2_pos = np.array([-sep, 0, 0])
        child_rad = 1.0 * scale

        d_c1 = sdf_sphere(self.grid_points, c1_pos, child_rad)
        d_c2 = sdf_sphere(self.grid_points, c2_pos, child_rad)

        # Smooth Union of Children
        d_children = smin(d_c1, d_c2, smooth)

        # Smooth Union with Root
        d_final = smin(d_root, d_children, smooth)

        return d_final

    def get_mesh(self, sep, scale, smooth):
        """Generates a PyVista mesh from the SDF."""
        vol = self.evaluate(sep, scale, smooth).reshape((self.res, self.res, self.res))

        try:
            # Marching Cubes to get vertices/faces
            verts, faces, normals, values = measure.marching_cubes(vol, level=0.0)

            # Map grid coordinates back to world coordinates
            verts = verts * (6.0 / (self.res - 1)) - 3.0

            # Create PyVista PolyData
            # PyVista requires faces to have a size indicator (3 for triangles)
            faces_pv = np.column_stack((np.full(len(faces), 3), faces)).flatten()
            mesh = pv.PolyData(verts, faces_pv)
            return mesh
        except:
            return None

    def get_projections(self, mesh, sep, scale, smooth):
        """
        Demonstrates the 'Compression Residuals'.
        Finds the closest point on the blob for our target vertices.
        """
        if mesh is None or mesh.n_points == 0:
            return None

        # PyVista has a fast KD-Tree locator
        closest_points = []
        for target in self.targets:
            # Find closest point on the generated blob surface
            idx = mesh.find_closest_point(target)
            closest_points.append(mesh.points[idx])

        return np.array(closest_points)

# --- 2. Visualization Setup ---

model = BlobModel()
plotter = pv.Plotter()

# Initial State
init_sep = 1.2
init_scale = 0.7
init_smooth = 0.5

# Placeholders for actors (the 3D objects)
mesh_actor = None
lines_actor = None
points_actor = None

def callback(value=None):
    """Called whenever a slider moves."""
    global mesh_actor, lines_actor, points_actor

    # Get slider values
    sep = plotter.slider_widgets[0].GetRepresentation().GetValue()
    scale = plotter.slider_widgets[1].GetRepresentation().GetValue()
    smooth = plotter.slider_widgets[2].GetRepresentation().GetValue()

    # 1. Generate the Blob Mesh
    mesh = model.get_mesh(sep, scale, smooth)
    if mesh is None: return

    # 2. Calculate "Residuals" (Projections)
    projected_points = model.get_projections(mesh, sep, scale, smooth)

    # --- Update the Scene ---

    # Remove old actors
    if mesh_actor: plotter.remove_actor(mesh_actor)
    if lines_actor: plotter.remove_actor(lines_actor)
    if points_actor: plotter.remove_actor(points_actor)

    # Add new Blob
    mesh_actor = plotter.add_mesh(mesh, color="lightblue", smooth_shading=True, specular=0.5)

    # Add Target Points (Red dots - The "Original Mesh")
    points_actor = plotter.add_points(model.targets, color="red", point_size=10, render_points_as_spheres=True)

    # Add Projection Lines (White lines - The "Residuals")
    # Build lines array: [2, id_target, id_proj, 2, id_target, id_proj...]
    lines = []
    all_pts = []
    curr_idx = 0
    for i, target in enumerate(model.targets):
        all_pts.append(target)
        all_pts.append(projected_points[i])
        lines.append([2, curr_idx, curr_idx+1])
        curr_idx += 2

    lines_poly = pv.PolyData(np.array(all_pts))
    lines_poly.lines = np.hstack(lines)
    lines_actor = plotter.add_mesh(lines_poly, color="white", line_width=2)

# --- 3. UI Controls ---

plotter.add_slider_widget(callback, [0.0, 2.5], value=init_sep, title="Child Offset", pointa=(0.025, 0.9), pointb=(0.25, 0.9))
plotter.add_slider_widget(callback, [0.1, 1.2], value=init_scale, title="Child Scale", pointa=(0.025, 0.75), pointb=(0.25, 0.75))
plotter.add_slider_widget(callback, [0.1, 1.5], value=init_smooth, title="Smoothness", pointa=(0.025, 0.6), pointb=(0.25, 0.6))

plotter.add_text("Algorithm Visualization:\n1. Adjust Sliders to fit Blob to Red Dots.\n2. White Lines = Residuals (To be compressed)", position='upper_right', font_size=10)

# Initial Draw
callback()

plotter.show()