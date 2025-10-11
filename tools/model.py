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
from scipy.spatial import KDTree, procrustes
import joblib
import shutil
import zipfile
import tempfile

train_path = "./train_data/PersonalizedPelvisStructures"
file_list = os.listdir(train_path)

# Model
class Model:
    def __init__(self, ctrl, state, template_mesh):
        self.ctrl = ctrl
        self.state = state
        self.template_mesh = template_mesh
    
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
    
    ## Evaluation
    def _compute_mesh_volume(self, vertices_flat: np.ndarray):
        """ Tính thể tích của mesh. Cần thư viện mesh. """
        # Giả định: Volume/Scale được tính toán dựa trên bounding box cho mục đích minh họa
        # THỰC TẾ: NÊN DÙNG THỂ TÍCH HOẶC DIỆN TÍCH BỀ MẶT THỰC CỦA MESH
        vertices_3d = vertices_flat.reshape(-1, 3)
        min_coords = np.min(vertices_3d, axis=0)
        max_coords = np.max(vertices_3d, axis=0)
        size = np.prod(max_coords - min_coords)
        return size if size > 0 else 1.0 # Trả về thể tích

    def _find_nearest_vertices(self, mesh_vertices: np.ndarray, landmark_coords: np.ndarray, radius: float = 0.05):
        """ Tìm chỉ số các đỉnh trong một bán kính R quanh một landmark. """
        distances = np.linalg.norm(mesh_vertices - landmark_coords, axis=1)
        return np.where(distances < radius)[0]

    def compute_inter_mesh_consistency(self, features: np.ndarray):
        """
        Đánh giá độ nhất quán sau khi đã căn chỉnh (Procrustes Analysis) để bỏ qua
        sự khác biệt về vị trí, hướng, và tỷ lệ tổng thể.
        """
        n_meshes = features.shape[0]
        # Lấy mẫu ngẫu nhiên
        selected_indices = np.random.choice(n_meshes, size=min(50, n_meshes), replace=False)
        selected_features = features[selected_indices]
        
        n_landmarks = selected_features.shape[1] // 3
        
        # Reshape về dạng (n_meshes, n_landmarks, 3)
        landmark_positions = selected_features.reshape(-1, n_landmarks, 3)
        
        # 1. Generalized Procrustes Analysis (GPA)
        # Chọn mesh đầu tiên làm reference (hoặc tính mean shape)
        reference_shape = landmark_positions[0]
        aligned_shapes = []
        
        # Thực hiện căn chỉnh cho tất cả các meshes còn lại so với reference
        for i in range(landmark_positions.shape[0]):
            target_shape = landmark_positions[i]
            
            try:
                mtx1, mtx2_aligned, _ = procrustes(reference_shape, target_shape)
                aligned_shapes.append(mtx2_aligned)
            except ValueError:
                continue

        if not aligned_shapes:
            return 0.0
            
        aligned_landmarks = np.stack(aligned_shapes, axis=0) # shape: (N_aligned, n_landmarks, 3)
        
        # 2. Tính độ lệch chuẩn trên các landmark đã được căn chỉnh
        landmark_std_aligned = np.std(aligned_landmarks, axis=0)  # shape: (n_landmarks, 3)
        norms_aligned = np.linalg.norm(landmark_std_aligned, axis=1)
        
        # Giá trị càng thấp càng tốt
        avg_std_per_landmark_aligned = np.mean(norms_aligned)
        
        print(f"Inter-mesh consistency (avg std per landmark, Procrustes-aligned): {avg_std_per_landmark_aligned:.6f}")
        return avg_std_per_landmark_aligned

    def compute_landmark_shape_correlation(self, vertices: np.ndarray, features: np.ndarray, k_neighbors: int = 20):
        """
        Tính tương quan cục bộ giữa landmark và các đỉnh lân cận (K-NN) để đánh giá tính gắn kết cục bộ.
        """
        n_meshes = vertices.shape[0]
        # Tăng kích thước mẫu để có thống kê tốt hơn
        selected_indices = np.random.choice(n_meshes, size=min(100, n_meshes), replace=False) 
        
        sample_vertices = vertices[selected_indices]  # shape: (N_sample, n_vertices*3)
        sample_features = features[selected_indices]  # shape: (N_sample, n_landmarks*3)
        
        correlations = []
        n_landmarks = sample_features.shape[1] // 3
        n_vertices = sample_vertices.shape[1] // 3
        
        # Reshape về dạng (N_sample, n_landmarks, 3)
        landmarks_reshaped = sample_features.reshape(-1, n_landmarks, 3) 
        
        # --- 1. Xây dựng KDTree từ MESH MẪU ---
        
        # Chọn mesh đầu tiên làm đại diện để tìm các đỉnh lân cận
        reference_mesh_idx = 0 
        # Vị trí 3D của tất cả vertices trên mesh mẫu: shape (n_vertices, 3)
        reference_vertices_3d = sample_vertices[reference_mesh_idx].reshape(n_vertices, 3)
        
        # Xây dựng cây KDTree để tìm lân cận nhanh
        tree = KDTree(reference_vertices_3d) 
        
        # --- 2. Vòng lặp qua các Landmark ---
        for i in range(n_landmarks):
            # Tọa độ của landmark thứ i trên tất cả meshes: shape (N_sample, 3)
            landmark_i = landmarks_reshaped[:, i, :]  
            
            # Lấy vị trí landmark i trên mesh mẫu đầu tiên
            current_landmark_coords = landmarks_reshaped[reference_mesh_idx, i, :]
            
            # Tìm K đỉnh gần nhất (K-NN) trong không gian Euclide
            distances, sampled_vertex_indices = tree.query(current_landmark_coords, k=k_neighbors)
            
            # --- 3. Vòng lặp qua các Đỉnh Lân cận (Đã được tìm thấy) ---
            for vertex_index in sampled_vertex_indices: # type: ignore
                
                # Đảm bảo chỉ số không nằm ngoài giới hạn (mặc dù tree.query đã xử lý)
                if vertex_index >= n_vertices: continue 

                # Lấy tọa độ (X, Y, Z) của đỉnh này trên tất cả N_sample meshes
                start_col = vertex_index * 3
                end_col = vertex_index * 3 + 3
                
                # vtx_coords_all_meshes: shape (N_sample, 3)
                # Đây là tọa độ của đỉnh 'vertex_index' trên tất cả các mesh đã chọn
                vtx_coords_all_meshes = sample_vertices[:, start_col:end_col]
                
                # --- 4. Tính correlation cho từng dimension (x, y, z) ---
                for dim in range(3):
                    
                    L_dim = landmark_i[:, dim]
                    V_dim = vtx_coords_all_meshes[:, dim]

                    # Cần đảm bảo std > 0 để corrcoef hoạt động
                    # Điều kiện này ngăn việc tính tương quan nếu dữ liệu không thay đổi
                    if (np.std(L_dim) > 1e-10 and np.std(V_dim) > 1e-10):
                        
                        # np.corrcoef trả về ma trận 2x2. Ta chỉ cần giá trị tương quan [0, 1]
                        corr = np.corrcoef(L_dim, V_dim)[0, 1]
                        
                        if not np.isnan(corr):
                            correlations.append(abs(corr))
        
        shape_correlation = np.mean(correlations) if correlations else 0
        print(f"Shape correlation score (K-NN local, K={k_neighbors}): {shape_correlation:.6f}")
        
        # Kiểm tra cuối cùng: Nếu correlations rỗng
        if not correlations:
            print("CẢNH BÁO: Không có tương quan nào được tính (có thể do lỗi indexing hoặc std=0).")
            
        return shape_correlation

    def compute_feature_reconstruction_capability(self, vertices: np.ndarray, features: np.ndarray):
        """
        Optimized feature reconstruction capability using regular PCA and consistent feature size
        """
        n = vertices.shape[0]
        random_list = np.random.choice(n, size=50, replace=False)
        sample_vertices = vertices[random_list]
        sample_features = features[random_list]
        
        # Convert to arrays and ensure consistent dimensions
        all_landmarks = np.array(sample_features)
        all_shapes = np.array(sample_vertices)
        
        # Check dimensions
        if all_landmarks.shape[0] < 2 or all_shapes.shape[0] < 2:
            print("Warning: Not enough samples for PCA")
            return 0.5  # Return neutral score
        
        # Use regular PCA instead of IncrementalPCA for stability
        n_components = min(5, all_landmarks.shape[0] - 1, all_landmarks.shape[1])
        
        if n_components < 1:
            return 0.5
        
        try:
            # PCA on landmarks
            pca_landmarks = PCA(n_components=n_components)
            landmark_pcs = pca_landmarks.fit_transform(all_landmarks)
            
            # PCA on shapes
            pca_shapes = PCA(n_components=n_components)
            shape_pcs = pca_shapes.fit_transform(all_shapes)
            
            # Vectorized correlation calculation
            correlations = []
            for i in range(min(landmark_pcs.shape[1], shape_pcs.shape[1])):
                if (np.std(landmark_pcs[:, i]) > 1e-10 and np.std(shape_pcs[:, i]) > 1e-10):
                    corr = np.corrcoef(landmark_pcs[:, i], shape_pcs[:, i])[0, 1]
                    if not np.isnan(corr):
                        correlations.append(abs(corr))
            
            pca_correlation = np.mean(correlations) if correlations else 0
            
            # Variance ratio (first 3 components for speed)
            variance_ratio = np.sum(pca_landmarks.explained_variance_ratio_[:min(3, n_components)])
            
            reconstruction_score = (pca_correlation + variance_ratio) / 2
            
            print(f"Feature reconstruction capability: {reconstruction_score:.6f}")
            return reconstruction_score
            
        except Exception as e:
            print(f"PCA computation failed: {e}")
            return 0.5 # Return neutral score on failure

    def compute_anatomical_sensitivity(self, vertices: np.ndarray, features: np.ndarray):
        """
        Tính độ nhạy cảm với biến đổi giải phẫu.
        """
        n_meshes = vertices.shape[0]
        selected_indices = np.random.choice(n_meshes, size=min(50, n_meshes), replace=False)
        
        sample_vertices = vertices[selected_indices]  # shape: (50, n_vertices*3)
        sample_features = features[selected_indices]  # shape: (50, n_landmarks*3)
        
        # 1. Tính kích thước mesh nội tại (Volume)
        mesh_volumes = []
        for i in range(sample_vertices.shape[0]):
            # Sử dụng hàm giả định để tính Volume/Area (THAY THẾ BẰNG HÀM THỰC TẾ)
            volume = self._compute_mesh_volume(sample_vertices[i])
            mesh_volumes.append(volume)
        
        mesh_volumes = np.array(mesh_volumes)  # shape: (50,)
        
        # Chuẩn hóa volume để giảm ảnh hưởng của độ lớn tuyệt đối
        mesh_volumes_normalized = (mesh_volumes - np.mean(mesh_volumes)) / (np.std(mesh_volumes) + 1e-8)
        
        # Reshape features to (50, n_landmarks, 3)
        n_landmarks = sample_features.shape[1] // 3
        landmarks_reshaped = sample_features.reshape(-1, n_landmarks, 3)  # (50, n_landmarks, 3)
        
        # 2. Tính correlation giữa Volume và tọa độ landmark
        correlations = []
        
        for landmark_idx in range(n_landmarks):
            landmark_coords = landmarks_reshaped[:, landmark_idx, :]
            
            # Tính correlation cho từng tọa độ (x, y, z)
            for coord_idx in range(3):
                landmark_dim_coords = landmark_coords[:, coord_idx]
                
                # Cần đảm bảo std > 0
                if (np.std(mesh_volumes_normalized) > 1e-10 and np.std(landmark_dim_coords) > 1e-10):
                    # Tính tương quan giữa Volume và tọa độ của landmark
                    corr = np.corrcoef(mesh_volumes_normalized, landmark_dim_coords)[0, 1]
                    if not np.isnan(corr):
                        correlations.append(abs(corr))
        
        anatomical_sensitivity = np.mean(correlations) if correlations else 0
        
        print(f"Anatomical variation sensitivity (Volume-based): {anatomical_sensitivity:.6f}")
        return anatomical_sensitivity

    def evaluate_cross_landmarking_performance(self, vertices: np.ndarray, features: np.ndarray):
        """
        Optimized comprehensive evaluation
        """
        evaluation_metrics = {}
        
        # Đảm bảo chúng ta có đủ samples
        n_meshes = vertices.shape[0]
        if n_meshes < 2:
            print("Warning: Not enough meshes for evaluation")
            return {}
        
        # Run evaluations
        inter_mesh_consistency = self.compute_inter_mesh_consistency(features)
        shape_correlation = self.compute_landmark_shape_correlation(vertices, features)
        reconstruction_capability = self.compute_feature_reconstruction_capability(vertices, features)
        anatomical_sensitivity = self.compute_anatomical_sensitivity(vertices, features)
        
        # Convert consistency to score (lower std = better)
        consistency_score = 1.0 / (1.0 + inter_mesh_consistency)
        evaluation_metrics['consistency_score'] = consistency_score
        evaluation_metrics['inter_mesh_std'] = inter_mesh_consistency
        evaluation_metrics['shape_correlation'] = shape_correlation
        evaluation_metrics['feature_reconstruction'] = reconstruction_capability
        evaluation_metrics['anatomical_sensitivity'] = anatomical_sensitivity
        
        return evaluation_metrics

    # ===== Main functions =====
    def cross_landmark(self):
        """
        Optimized main cross-landmark function
        """
        # Load template mesh
        template_mesh = self.template_mesh

        if template_mesh is None:
            raise ValueError("Template mesh is not set in the model state.")

        # Extract landmark points
        landmarks = np.array([
            [float(vertex["x"]), float(vertex["y"]), float(vertex["z"])]
            for vertex in self.state.picked_vertices
        ])
        
        if len(landmarks) == 0:
            raise ValueError("No landmarks provided for cross-landmarking.")

        # Compute barycentric coordinates
        pelvicFeatureBaryIndices, pelvicFeatureBaryCoords = self.computeBarycentricLandmarks(template_mesh, landmarks)

        # Process files
        all_vertices = []
        all_features = []

        for file in file_list:
            mesh = self.load_mesh(f"{train_path}/{file}")
            tri_mesh = self.vtk_to_trimesh(mesh)
            pelvicFeatures = self.reconstructLandmarksFromBarycentric(
                tri_mesh, pelvicFeatureBaryIndices, pelvicFeatureBaryCoords
            )
            all_vertices.append(tri_mesh.vertices.flatten())
            all_features.append(pelvicFeatures.flatten())

        # Convert to arrays
        vertices_array = np.array(all_vertices)  # shape: (n_meshes, n_vertices*3)
        features_array = np.array(all_features)  # shape: (n_meshes, n_landmarks*3)
        
        # Evaluate performance
        if vertices_array.shape[0] > 0:
            performance_metrics = self.evaluate_cross_landmarking_performance(
                vertices=vertices_array,
                features=features_array
            )
        else:
            performance_metrics = {}

        return all_vertices, all_features, performance_metrics
    
    def train(self, vertices, features):
        # Load template mesh
        template_mesh = self.template_mesh
        template_mesh_NoMuscle = trimesh.load_mesh("./train_data/TempPelvisBoneMesh.ply")
        # Estimate the nearest indices
        tri_template_mesh = self.vtk_to_trimesh(template_mesh)
        pelvisBoneVertexIndices = self.estimateNearestIndicesKDTreeBased(template_mesh_NoMuscle.vertices, tri_template_mesh.vertices)
        
        trainingPelvicFeatureData, validPelvicFeatureData, trainingPelvicVertexData, validPelvicVertexData = train_test_split(features, vertices, test_size= 0.2, random_state= 42, shuffle= True)

        numComps = self.state.n_components
        # Scale the pelvic feature and vertex data
        yScaler = StandardScaler().fit(trainingPelvicVertexData)
        xScaler = StandardScaler().fit(trainingPelvicFeatureData)
        scaledYData = yScaler.transform(trainingPelvicVertexData)
        scaledXData = xScaler.transform(trainingPelvicFeatureData)

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
        