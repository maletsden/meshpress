from ..reader import Reader
from utils.types import *
import pywavefront


class OBJReader(Reader):
    def __init__(self):
        pass

    def read(self, path: str) -> Model:
        model: Model = Model([], [])
        try:
            # model_obj = pywavefront.Wavefront(path, collect_faces=True)

            with open(path, 'r') as file:
                for line in file:
                    # Strip whitespace and check line type
                    stripped_line = line.strip()

                    if stripped_line.startswith('v '):  # Vertex line
                        parts = stripped_line.split()
                        # Parse vertex data (skip the first part 'v')
                        vertex = tuple(map(float, parts[1:]))
                        model.vertices.append(Vertex(*vertex))

                    elif stripped_line.startswith('f '):  # Face line
                        parts = stripped_line.split()
                        # Parse face data (skip the first part 'f')
                        indices = list(map(lambda x: int(x.split('/')[0]) - 1, parts[1:]))
                        # Triangulate: fan from first vertex for polygons with >3 verts
                        for i in range(1, len(indices) - 1):
                            model.triangles.append(Triangle(indices[0], indices[i], indices[i + 1]))

            # return vertices, faces
            #
            # model.vertices = list(map(lambda v: Vertex(*v), model_obj.vertices))
            # model.triangles = list(map(lambda f: Triangle(*f), model_obj.mesh_list[0].faces))

            return model

        except FileNotFoundError:
            print(f"ObjReader: {path} not found.")
        except:
            print("ObjReader: An error occurred while loading the shape.")
