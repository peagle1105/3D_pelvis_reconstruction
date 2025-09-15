from vtk import vtkWorldPointPicker

class Mouse:
    def __init__(self, control, state, renderer_2d, sphere_actor, get_dicom_reader_callback) -> None:
        self.ctrl = control
        self.state = state
        self.renderer_2d = renderer_2d  # Đảm bảo đây là vtkRenderer của viewport 2D
        self.sphere_actor = sphere_actor
        self.get_dicom_reader = get_dicom_reader_callback
        self.picker = vtkWorldPointPicker()  # Thêm picker để chuyển đổi tọa độ chính xác

    def on_mouse_move(self, obj, event):
        iren = obj
        mouse_pos = iren.GetEventPosition()
        
        # Lấy renderer từ interactor
        renderer = iren.FindPokedRenderer(mouse_pos[0], mouse_pos[1])
        if renderer != self.renderer_2d:
            return  # Chỉ xử lý nếu chuột trong renderer 2D

        dicom_reader = self.get_dicom_reader()
        if not self.state.data_loaded or dicom_reader is None:
            return

        # Sử dụng picker để chuyển đổi sang world coordinates
        self.picker.Pick(mouse_pos[0], mouse_pos[1], 0, renderer)
        world_point = list(self.picker.GetPickPosition())

        # Hiệu chỉnh theo hướng slice và vị trí hiện tại
        output = dicom_reader.GetOutput()
        origin = output.GetOrigin()
        spacing = output.GetSpacing()
        slice_index = self.state.slice_index
        orientation = self.state.current_view

        if orientation == "axial":
            world_point[2] = origin[2] + slice_index * spacing[2]
        elif orientation == "sagittal":
            world_point[0] = origin[0] + slice_index * spacing[0]
        elif orientation == "coronal":
            world_point[1] = origin[1] + slice_index * spacing[1]

        # Chuyển đổi sang voxel coordinates
        voxel_coords = [
            int((world_point[0] - origin[0]) / spacing[0]),
            int((world_point[1] - origin[1]) / spacing[1]),
            int((world_point[2] - origin[2]) / spacing[2])
        ]

        # Kiểm tra giới hạn voxel
        dimensions = output.GetDimensions()
        if (0 <= voxel_coords[0] < dimensions[0] and 
            0 <= voxel_coords[1] < dimensions[1] and 
            0 <= voxel_coords[2] < dimensions[2]):
            voxel_value = output.GetScalarComponentAsDouble(
                voxel_coords[0], voxel_coords[1], voxel_coords[2], 0
            )
            print(f"Voxel coordinates: {voxel_coords}, Value: {voxel_value}")
        else:
            print("Voxel coordinates out of bounds")
        world_point = tuple(world_point)
        # Cập nhật vị trí sphere
        self.sphere_actor.SetPosition(world_point)
        self.sphere_actor.SetVisibility(True)
        self.ctrl.view_update_3d()