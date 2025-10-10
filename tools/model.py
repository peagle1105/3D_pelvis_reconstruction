# Import necessary libraries
import vtk
import numpy as np
import os
import io
import trimesh
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error
from sklearn.multioutput import MultiOutputRegressor
from sklearn.linear_model import RidgeCV
from scipy.spatial import KDTree
import joblib
import shutil
import zipfile
import tempfile

train_path = "./train_data/PersonalizedPelvisStructures"
file_list = os.listdir(train_path)

# Model
class Model:
    def __init__(self, ctrl, state):
        self.ctrl = ctrl
        self.state = state
        self.template_mesh = self.state.template_mesh
    
    # ===== Sub-functions =====
    def load_mesh(self, mesh_path):
        if not os.path.exists(mesh_path):
            raise FileNotFoundError(f"Mesh file not found: {mesh_path}")
        
        reader = vtk.vtkPLYReader()
        reader.SetFileName(mesh_path)
        reader.Update()

        transformer = vtk.vtkTransform()
        transformer.Scale(1000, 1000, 1000)  # Scale from meters to millimeters

        transformer_filter = vtk.vtkTransformPolyDataFilter()
        transformer_filter.SetTransform(transformer)
        transformer_filter.SetInputData(reader.GetOutput())
        transformer_filter.Update()

        mesh = transformer_filter.GetOutput()

        if mesh is None:
            raise ValueError("Failed to load mesh from the provided path.")
        
        return mesh
    
    def vtk_to_trimesh(self, vtk_mesh: vtk.vtkPolyData) -> trimesh.Trimesh:
        points = np.array([vtk_mesh.GetPoint(i) for i in range(vtk_mesh.GetNumberOfPoints())])

        faces = []
        for i in range(vtk_mesh.GetNumberOfCells()):
            cell = vtk_mesh.GetCell(i)
            ids = cell.GetPointIds()
            face = [ids.GetId(j) for j in range(ids.GetNumberOfIds())]
            if len(face) == 3:  # Chỉ lấy tam giác
                faces.append(face)

        return trimesh.Trimesh(vertices=points, faces=np.array(faces))

    def trimesh_to_vtk(self, tri_mesh: trimesh.Trimesh) -> vtk.vtkPolyData:
        vtk_points = vtk.vtkPoints()
        for point in tri_mesh.vertices:
            vtk_points.InsertNextPoint(point)

        vtk_cells = vtk.vtkCellArray()
        for face in tri_mesh.faces:
            id_list = vtk.vtkIdList()
            for idx in face:
                id_list.InsertNextId(idx)
            vtk_cells.InsertNextCell(id_list)

        vtk_mesh = vtk.vtkPolyData()
        vtk_mesh.SetPoints(vtk_points)
        vtk_mesh.SetPolys(vtk_cells)

        return vtk_mesh

    def computeBarycentricLandmarks(self, template_mesh: vtk.vtkPolyData, landmarkPoints: np.ndarray):
        """
        Given 3D landmark points and a mesh, compute the triangle indices and
        barycentric coordinates for use in trimesh.registration.nricp_amberg.

        Args:
            mesh (vtk.vtkPolyData): The mesh where landmarks will be localized.
            landmarkPoints (np.ndarray): (n, 3) array of 3D landmark points.

        Returns:
            tuple:
                triIndices (np.ndarray): (n,) int array of triangle indices.
                baryCoords (np.ndarray): (n, 3) float array of barycentric coordinates.
        """
        mesh = self.vtk_to_trimesh(template_mesh)
        closest = trimesh.proximity.closest_point(mesh, landmarkPoints)
        triIndices = closest[2]      # (n,) array of triangle indices
        
        baryCoords = []
        for point, triIndex in zip(landmarkPoints, triIndices):
            triVerts = mesh.triangles[triIndex]  # shape (3, 3)
            v0, v1, v2 = triVerts

            # Compute barycentric coordinates
            T = np.column_stack([v1 - v0, v2 - v0])
            v = point - v0
            A = np.dot(T.T, T)
            b = np.dot(T.T, v)

            try:
                x = np.linalg.solve(A, b)
            except np.linalg.LinAlgError:
                x = np.zeros(2)

            w1 = 1.0 - x[0] - x[1]
            w2 = x[0]
            w3 = x[1]
            baryCoords.append([w1, w2, w3])

        return np.array(triIndices), np.array(baryCoords)
    
    def reconstructLandmarksFromBarycentric(self, target_mesh: trimesh.Trimesh, baryIndices: np.ndarray, baryCoords: np.ndarray) -> np.ndarray:
        """
        Reconstruct 3D points from barycentric coordinates and triangle indices.

        Args:
            mesh (vtk.vtkPolyData): The source mesh.
            triIndices (np.ndarray): (n,) array of triangle indices.
            baryCoords (np.ndarray): (n, 3) array of barycentric coordinates.

        Returns:
            np.ndarray: (n, 3) array of reconstructed 3D points.
        """
        if baryIndices.shape[0] != baryCoords.shape[0]:
            raise ValueError("triIndices and baryCoords must have the same number of entries.")
        
        mesh = target_mesh
        # Get the triangles from the mesh
        tris = mesh.triangles[baryIndices]  # shape (n, 3, 3)

        # Unpack vertices: V = w1*V0 + w2*V1 + w3*V2
        v0 = tris[:, 0, :]  # (n, 3)
        v1 = tris[:, 1, :]
        v2 = tris[:, 2, :]

        w1 = baryCoords[:, 0][:, np.newaxis]
        w2 = baryCoords[:, 1][:, np.newaxis]
        w3 = baryCoords[:, 2][:, np.newaxis]

        points = w1 * v0 + w2 * v1 + w3 * v2  # (n, 3)
        return points

    def estimateNearestIndicesKDTreeBased(self, inSourcePoints, inTargetPoints, inThreshold=1e-6):
        # Prepare buffers
        sourcePoints = np.array(inSourcePoints)
        targetPoints = np.array(inTargetPoints)

        # Create a KD-tree for the body vertices
        targetPointTree = KDTree(targetPoints)
        
        # Find the distances from each head vertex to the nearest body vertex
        distances, indices = targetPointTree.query(sourcePoints)
        
        # Return buffer
        return indices
    # ===== Main functions =====
    def train(self):
        # Load template mesh
        template_mesh = self.template_mesh
        template_mesh_NoMuscle = trimesh.load_mesh("./train_data/TempPelvisBoneMesh.ply")
        if template_mesh is None:
            raise ValueError("Template mesh is not set in the model state.")

        # Example landmark points (replace with actual data)
        landmarks = []
        for vertex in self.state.picked_vertices:
            x, y, z = float(vertex["x"]), float(vertex["y"]), float(vertex["z"])
            landmark = (x, y, z)
            landmarks.append(landmark)
        
        landmarks = np.array(landmarks)

        # Compute barycentric coordinates and triangle indices
        tri_template_mesh = self.vtk_to_trimesh(template_mesh)
        pelvisBoneVertexIndices = self.estimateNearestIndicesKDTreeBased(template_mesh_NoMuscle.vertices, tri_template_mesh.vertices)
        pelvicFeatureBaryIndices, pelvicFeatureBaryCoords = self.computeBarycentricLandmarks(template_mesh, landmarks)

        vertices = []
        features = []

        for file in file_list:
            mesh = self.load_mesh(f"{train_path}/{file}")
            tri_mesh = self.vtk_to_trimesh(mesh)
            pelvicFeatures = self.reconstructLandmarksFromBarycentric(tri_mesh, pelvicFeatureBaryIndices, pelvicFeatureBaryCoords)
            vertices.append(tri_mesh.vertices.flatten())
            features.append(pelvicFeatures.flatten())

        trainingPelvicFeatureData, validPelvicFeatureData, trainingPelvicVertexData, validPelvicVertexData = train_test_split(features, vertices, test_size= 0.2, random_state= 42, shuffle= True)

        numComps = self.state.n_components
        # Scale the pelvic feature and vertex data
        yScaler = StandardScaler().fit(trainingPelvicVertexData)
        xScaler = StandardScaler().fit(trainingPelvicFeatureData)
        scaledYData = yScaler.transform(trainingPelvicVertexData)
        scaledXData = xScaler.transform(trainingPelvicFeatureData)

        self.state.train_data_status = 'success'

        # Control the number of components
        targetNumComps = numComps
        xDims = scaledXData.shape[1]
        yDims = scaledYData.shape[1]
        xNumComps = min(xDims, targetNumComps)
        yNumComps = min(yDims, targetNumComps)

        # Parameterize the scaled pelvic vertex and feature data
        xSSM = PCA(n_components=xNumComps)
        xSSM.fit(scaledXData)
        trainXParamData = xSSM.transform(scaledXData)
        ySSM = PCA(n_components=yNumComps)
        ySSM.fit(scaledYData)
        trainYParamData = ySSM.transform(scaledYData)

        # Linear regression from feature parameters to vertex parameters
        model = MultiOutputRegressor(RidgeCV())
        model.fit(trainXParamData, trainYParamData)

        # Computing errors on the validating data
        ## Scale and parameterize the validation data
        scaledValidXData = xScaler.transform(validPelvicFeatureData)
        validXParams = xSSM.transform(scaledValidXData)
        ## Try to predict the pelvic vertex params from the pelvic feature params
        predYParams = model.predict(validXParams)
        ## Inverse transform to the scaled Y data
        predYParams = np.array(predYParams).reshape(-1, yNumComps)
        predScaledYData = ySSM.inverse_transform(predYParams)
        predYData = yScaler.inverse_transform(predScaledYData)
        ## Computing validating errors
        avgP2PDists = []
        for v, predY in enumerate(predYData):
            # Getting the valiation data and predicted data
            validY = validPelvicVertexData[v]
            validPelvicStructureVertices = validY.reshape(-1, 3)
            predPelvicStructureVertices = predY.reshape(-1, 3)

            # Compute accuracy only on the bone vertices because we do not have ground truth of the muscle
            validPelvisBoneVertices = validPelvicStructureVertices[pelvisBoneVertexIndices]
            predPelvisBoneVertices = predPelvicStructureVertices[pelvisBoneVertexIndices]

            # Compute points to points distances
            avgP2PDist = mean_absolute_error(validPelvisBoneVertices, predPelvisBoneVertices)
            avgP2PDists.append(avgP2PDist)
        avgP2PDists = np.array(avgP2PDists)
        mse = np.mean(avgP2PDists)
        self.state.eval_mse = mse
        print(avgP2PDists)

        return model, xScaler, yScaler, xSSM, ySSM
    
    def save_model(self, model, xScaler, yScaler, xSSM, ySSM):
        temp_dir = "./temp_model_path"
        zip_path = "./model_archive.zip"

        # Bước 1: Tạo thư mục tạm
        os.makedirs(temp_dir, exist_ok=True)

        # Bước 2: Lưu các mô hình vào thư mục
        joblib.dump(model, os.path.join(temp_dir, "model.pkl"))
        joblib.dump(xScaler, os.path.join(temp_dir, "xScaler.pkl"))
        joblib.dump(yScaler, os.path.join(temp_dir, "yScaler.pkl"))
        joblib.dump(xSSM, os.path.join(temp_dir, "xSSM.pkl"))
        joblib.dump(ySSM, os.path.join(temp_dir, "ySSM.pkl"))

        # Bước 3: Nén thư mục thành file ZIP
        with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zipf:
            for root, dirs, files in os.walk(temp_dir):
                for file in files:
                    file_path = os.path.join(root, file)
                    arcname = os.path.relpath(file_path, temp_dir)
                    zipf.write(file_path, arcname)

        # Bước 4: Đọc nội dung file ZIP
        with open(zip_path, "rb") as f:
            zip_content = f.read()

        # Bước 5: Xóa thư mục tạm và file ZIP
        shutil.rmtree(temp_dir)
        os.remove(zip_path)

        return zip_content

    def load_model(self, zip_bytes):
        # Create a temple folder for extracting
        temp_dir = tempfile.mkdtemp(prefix="model_load_")

        try:
            # Write the zip's content into temple file
            zip_path = os.path.join(temp_dir, "model.zip")
            with open(zip_path, "wb") as f:
                f.write(zip_bytes)

            # Extract zip
            with zipfile.ZipFile(zip_path, "r") as zipf:
                zipf.extractall(temp_dir)

            # Check and load model
            def load_component(name):
                path = os.path.join(temp_dir, f"{name}.pkl")
                if os.path.exists(path):
                    return joblib.load(path)
                else:
                    print(f"[DEBUG] Thiếu file: {name}.pkl")
                    return None

            model   = load_component("model")
            xScaler = load_component("xScaler")
            yScaler = load_component("yScaler")
            xSSM    = load_component("xSSM")
            ySSM    = load_component("ySSM")

            return model, xScaler, yScaler, xSSM, ySSM

        finally:
            # Delete temple file
            shutil.rmtree(temp_dir)

    def create_vertices_polydata(self, points_array) -> vtk.vtkPolyData:
        points = vtk.vtkPoints()
        vertices = vtk.vtkCellArray()

        for i, pt in enumerate(points_array):
            pid = points.InsertNextPoint(pt)
            vertices.InsertNextCell(1)
            vertices.InsertCellPoint(pid)

        polydata = vtk.vtkPolyData()
        polydata.SetPoints(points)
        polydata.SetVerts(vertices)

        return polydata

    def create_mesh_polydata(self, points_array, faces) -> vtk.vtkPolyData :
        points_array = np.asarray(points_array)
        
        # Kiểm tra shape
        if points_array.ndim != 2 or points_array.shape[1] != 3:
            raise ValueError(f"points_array must have shape (n, 3), got {points_array.shape}")
        
        points = vtk.vtkPoints()
        for pt in points_array:
            # Chuyển đổi sang float để đảm bảo type safety
            x, y, z = float(pt[0]), float(pt[1]), float(pt[2])
            points.InsertNextPoint(x, y, z)

        polys = vtk.vtkCellArray()
        
        # Xử lý cả hai trường hợp: vtkCellArray và danh sách thông thường
        if isinstance(faces, vtk.vtkCellArray):
            # Nếu là vtkCellArray, sao chép trực tiếp
            polys.DeepCopy(faces)
        else:
            # Nếu là danh sách thông thường, xây dựng vtkCellArray
            for face in faces:
                face_ids = vtk.vtkIdList()
                for pid in face:
                    face_ids.InsertNextId(int(pid))
                polys.InsertNextCell(face_ids)

        polydata = vtk.vtkPolyData()
        polydata.SetPoints(points)
        polydata.SetPolys(polys)

        return polydata
        