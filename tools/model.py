import os
import sklearn
from sklearn.model_selection import train_test_split
from sklearn.multioutput import MultiOutputRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.linear_model import RidgeCV
import vtk
import numpy as np

train_path = "./train_data/PersonalizedPelvisStructures/"
landmark_path = "./train_data/Landmarks/"
os.makedirs(landmark_path, exist_ok=True)

file_list = os.listdir(train_path)
file_list = [f for f in file_list if os.path.isfile(os.path.join(train_path, f)) and f.endswith("ply")]
file_sample_mesh = file_list[0]

class Model:
    def __init__(self, ctrl, state) -> None:
        self.ctrl = ctrl
        self.state = state
        mesh_source = vtk.vtkPLYReader()
        mesh_source.SetFileName(os.path.join(train_path, file_sample_mesh))
        self.ref_mesh = mesh_source.GetOutput()
    
    # ====== Sub-functions =======
    def load_mesh(self, file_path: str) -> vtk.vtkPolyData:
        try:
            reader = vtk.vtkPLYReader()
            reader.SetFileName(file_path)
            reader.Update()
            poly_data = reader.GetOutput()

            # Tạo transform để scale 1000 lần
            transform = vtk.vtkTransform()
            transform.Scale(1000, 1000, 1000)  # Scale đều theo cả 3 trục

            # Áp dụng transform lên polydata
            transform_filter = vtk.vtkTransformPolyDataFilter()
            transform_filter.SetInputData(poly_data)
            transform_filter.SetTransform(transform)
            transform_filter.Update()

            return transform_filter.GetOutput()
        except Exception as e:
            print(f"Error loading mesh: {e}")
            return vtk.vtkPolyData()

    def get_pca_transform(self, polydata: vtk.vtkPolyData) -> vtk.vtkTransform:
        points = polydata.GetPoints()
        np_points = np.array([points.GetPoint(i) for i in range(points.GetNumberOfPoints())])
        centroid = np.mean(np_points, axis=0)
        centered = np_points - centroid
        u, s, vh = np.linalg.svd(centered)
        rotation = vh.T  # principal axes
        transform = vtk.vtkTransform()
        transform.PostMultiply()
        transform.Translate(-centroid[0], -centroid[1], -centroid[2])
        transform.Concatenate(rotation.flatten())
        transform.Translate(centroid[0], centroid[1], centroid[2])
        return transform

    def mesh_align(self, ref_mesh, target_mesh):
        source_t = self.get_pca_transform(ref_mesh)
        target_t = self.get_pca_transform(target_mesh)

        # Align target to source orientation
        align_transform = vtk.vtkTransform()
        align_transform.PostMultiply()
        align_transform.Concatenate(target_t.GetInverse())
        align_transform.Concatenate(source_t)

        transform_filter = vtk.vtkTransformPolyDataFilter()
        transform_filter.SetInputData(target_mesh)
        transform_filter.SetTransform(align_transform)
        transform_filter.Update()
        return transform_filter.GetOutput(), align_transform

    def find_corresponding_points(self, ref_mesh, new_mesh, landmark_coords, search_radius=5.0):
        """
        Tìm điểm tương đồng nhanh và chính xác
        - landmark_coords: list of [x, y, z] trên original_mesh
        - search_radius: giới hạn vùng tìm kiếm (đơn vị: cùng đơn vị mesh)
        """
        # Bước 1: Align thô bằng PCA
        aligned_new_mesh, _ = self.mesh_align(ref_mesh, new_mesh)

        # Bước 2: Dùng StaticPointLocator (nhanh hơn CellLocator nếu chỉ cần điểm)
        locator = vtk.vtkStaticPointLocator()
        locator.SetDataSet(aligned_new_mesh)
        locator.BuildLocator()

        corresponding = []
        for pt in landmark_coords:
            dist2 = 0.0
            closest_id = locator.FindClosestPointWithinRadius(search_radius, pt, dist2)
            if closest_id == -1:
                # Nếu không tìm thấy trong bán kính, fallback toàn cục
                closest_id = locator.FindClosestPoint(pt)
            closest_pt = aligned_new_mesh.GetPoint(closest_id)
            corresponding.append(closest_pt)

        return corresponding
    # ====== Main functions =======
    def cross_landmark(self):
        ref_landmarks = self.state.picked_vertices
        ref_mesh = self.ref_mesh
        if len(ref_landmarks) < 3:
            print("Please pick at least 3 landmarks on the reference mesh.")
            return
        for file_name in file_list:
            file_path = os.path.join(train_path, file_name)
            new_mesh = self.load_mesh(file_path)
            if new_mesh.GetNumberOfPoints() == 0:
                print(f"Failed to load mesh from {file_path}. Skipping.")
                continue
            corresponding_points = self.find_corresponding_points(ref_mesh, new_mesh, ref_landmarks)
            landmark_file = os.path.join(landmark_path, file_name.replace(".ply", "_landmarks.txt"))
            np.savetxt(landmark_file, np.array(corresponding_points))


            
