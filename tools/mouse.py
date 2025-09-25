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
        if not self.state.point_picking_enabled:
            return

        iren = obj
        mouse_pos = iren.GetEventPosition()

        dicom_reader = self.get_dicom_reader()
        if not self.state.data_loaded or dicom_reader is None:
            return

        # Use picker to get world coordinates
        self.picker.Pick(mouse_pos[0], mouse_pos[1], 0, self.renderer_2d)
        world_point = list(self.picker.GetPickPosition())

        # Adjust based on current slice and orientation
        output = dicom_reader.GetOutput()
        origin = output.GetOrigin()
        spacing = output.GetSpacing()
        slice_index = self.state.slice_index
        orientation = self.state.current_view

        # Calculate the actual world coordinates based on orientation
        if orientation == "axial":
            world_point[2] = origin[2] + slice_index * spacing[2]
        elif orientation == "sagittal":
            world_point[0] = origin[0] + slice_index * spacing[0]
        elif orientation == "coronal":
            world_point[1] = origin[1] + slice_index * spacing[1]
            
        # Cập nhật vị trí sphere
        self.sphere_actor.SetPosition(world_point)
        self.sphere_actor.SetVisibility(True)
        self.ctrl.view_update_3d()