from vtk import vtkTransform, vtkTransformPolyDataFilter
from vtkmodules.vtkFiltersSources import vtkPlaneSource

class Plane:
    def __init__(self, control, state, plane_source: vtkTransformPolyDataFilter, volume, get_dicom_reader_callback):
        self.ctrl = control
        self.state = state
        self.plane_source = plane_source
        self.volume = volume
        self.get_dicom_func = get_dicom_reader_callback
        self.init_plane = vtkPlaneSource()
        self.transform = vtkTransform()
        
        # Initialize with a default plane
        self.init_plane.SetOrigin(0, 0, 0 - self.volume.z_height_calc(self.state.slice_min))
        self.init_plane.SetPoint1(100, 0, 0)
        self.init_plane.SetPoint2(0, 100, 0)
        self.init_plane.SetNormal(0, 0, 1)
        self.init_plane.Update()

    def update_plane_position(self):
        """Update plane position based on current view"""
        dicom_reader = self.get_dicom_func()
        if not dicom_reader:
            return

        output = dicom_reader.GetOutput()
        if not output:
            return

        bounds = output.GetBounds()
        if not bounds or len(bounds) < 6:
            return
            
        x_min, x_max, y_min, y_max, z_min, z_max = bounds
        
        # Make sure we have valid slice indices
        if self.state.slice_max <= 0:
            return
            
        # Calculate the actual position in world coordinates
        slice_ratio = self.state.slice_index / self.state.slice_max
        
        if self.state.current_view == "axial":
            z_pos = z_min + (z_max - z_min) * slice_ratio
            dz = z_pos - self.state.z_height
            self.transform.Identity()
            self.transform.Translate(0, 0, dz)
        elif self.state.current_view == "sagittal":
            x_pos = x_min + (x_max - x_min) * slice_ratio
            dx = x_pos - self.state.x_height
            self.transform.Identity()
            self.transform.Translate(dx, 0, 0)
        else:  # coronal view
            y_pos = y_min + (y_max - y_min) * slice_ratio
            dy = y_pos - self.state.y_height
            self.transform.Identity()
            self.transform.Translate(0, dy, 0)

        self.plane_source.SetTransform(self.transform)
        self.plane_source.SetInputConnection(self.init_plane.GetOutputPort())
        self.plane_source.Update()

    def update_plane_size(self):
        """Update plane size based on volume bounds"""
        dicom_reader = self.get_dicom_func()
        if not dicom_reader:
            return
        
        output = dicom_reader.GetOutput()
        if not output:
            return
        
        bounds = output.GetBounds()
        if not bounds or len(bounds) < 6:
            return
            
        x_min, x_max, y_min, y_max, z_min, z_max = bounds
        
        # Add a margin to ensure the plane is always visible
        margin = 10.0
        
        # Set plane size based on current view orientation
        if self.state.current_view == "axial":
            # XY plane at Z position
            self.init_plane.SetOrigin(x_min - margin, y_min - margin, 0)
            self.init_plane.SetPoint1(x_max + margin, y_min - margin, 0)
            self.init_plane.SetPoint2(x_min - margin, y_max + margin, 0)
            self.init_plane.SetNormal(0, 0, 1)
        elif self.state.current_view == "sagittal":
            # YZ plane at X position
            self.init_plane.SetOrigin(0, y_min - margin, z_min - margin)
            self.init_plane.SetPoint1(0, y_max + margin, z_min - margin)
            self.init_plane.SetPoint2(0, y_min - margin, z_max + margin)
            self.init_plane.SetNormal(1, 0, 0)
        else:  # coronal view
            # XZ plane at Y position
            self.init_plane.SetOrigin(x_min - margin, 0, z_min - margin)
            self.init_plane.SetPoint1(x_max + margin, 0, z_min - margin)
            self.init_plane.SetPoint2(x_min - margin, 0, z_max + margin)
            self.init_plane.SetNormal(0, 1, 0)
        
        self.init_plane.Update()
        self.plane_source.Update()