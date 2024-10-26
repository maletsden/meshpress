from ..reader import Reader, Model, Vertex, Triangle
import pywavefront


class OBJReader(Reader):
    def __init__(self):
        pass

    def read(self, path: str) -> Model:
        model: Model = Model([], [])
        try:
            model_obj = pywavefront.Wavefront(path, collect_faces=True)

            model.vertices = list(map(lambda v: Vertex(*v), model_obj.vertices))
            model.triangles = list(map(lambda f: Triangle(*f), model_obj.mesh_list[0].faces))

            return model

        except FileNotFoundError:
            print(f"ObjReader: {path} not found.")
        except:
            print("ObjReader: An error occurred while loading the shape.")
