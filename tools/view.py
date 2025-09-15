from config.config import viewer_2d
from tools.plane import Plane

class View:
    def __init__(self, control, state, get_dicom_reader_callback, plane, renderer_2d) -> None:
        self.ctrl = control
        self.state = state
        self.get_dicom_func = get_dicom_reader_callback
        self.plane = plane
        self.renderer_2d = renderer_2d

    def update_view_orientation(self):
        """Update the slice orientation based on current view selection"""
        dicom_reader = self.get_dicom_func()
        global viewer_2d

        if viewer_2d is None or dicom_reader is None:
            return
        else:        
            # Update slice range
            output = dicom_reader.GetOutput()
            dims = output.GetDimensions()
            
            max_slice = 0
            if self.state.current_view == "axial":
                max_slice = dims[2] - 1
            elif self.state.current_view == "sagittal":
                max_slice = dims[0] - 1
            elif self.state.current_view == "coronal":
                max_slice = dims[1] - 1

            self.state.slice_max = max_slice
            
            # Reset slice index to middle of new range
            new_slice_index = max_slice
            self.state.slice_index = new_slice_index
            # Set orientation
            if self.state.current_view == "axial":
                viewer_2d.SetSliceOrientationToXY()
            elif self.state.current_view == "sagittal":
                viewer_2d.SetSliceOrientationToYZ()
            else:
                viewer_2d.SetSliceOrientationToXZ()
            
            # Update plane position and size
            self.plane.update_plane_position()
            self.plane.update_plane_size()

        # Reset camera and render
        self.renderer_2d.ResetCamera()
        self.ctrl.view_update_2d()