from vtk import vtkPiecewiseFunction
from vtkmodules.vtkRenderingCore import vtkColorTransferFunction

class Volume:
    def __init__(self, control, state, get_dicom_reader_callback):
        self.ctrl = control
        self.state = state
        self.get_dicom_func = get_dicom_reader_callback

    def update_volume_rendering(self, volume_3d):
        dicom_reader = self.get_dicom_func()
        
        if volume_3d is None or dicom_reader is None:
            return
        
        # Get scalar range for transfer functions
        scalar_range = dicom_reader.GetOutput().GetScalarRange()
        min_val, max_val = scalar_range
        
        # Create color transfer function
        color_function = vtkColorTransferFunction()
        color_function.AddRGBPoint(min_val, 0.0, 0.0, 0.0)   # Black
        color_function.AddRGBPoint((min_val + max_val) * 0.3, 0.67, 0.91, 0.67)  # Blue
        color_function.AddRGBPoint((min_val + max_val) * 0.6, 0.0, 0.4, 0.0)  # Green
        color_function.AddRGBPoint(max_val, 1.0, 1.0, 1.0)    # White
        
        # Create opacity transfer function
        opacity_function = vtkPiecewiseFunction()
        opacity_function.AddPoint(min_val, 0.0)
        opacity_function.AddPoint((min_val + max_val) * 0.3, 0.0)
        opacity_function.AddPoint((min_val + max_val) * 0.5, self.state.opacity * 0.5)
        opacity_function.AddPoint((min_val + max_val) * 0.7, self.state.opacity)
        opacity_function.AddPoint(max_val, self.state.opacity)
        
        # Update volume properties
        volume_property = volume_3d.GetProperty()
        volume_property.SetColor(color_function)
        volume_property.SetScalarOpacity(opacity_function)
        volume_property.ShadeOn()
        volume_property.SetInterpolationTypeToLinear()
        
        self.ctrl.view_update_3d()        

    def z_height_calc(self, slice_index) -> float:
        dicom_reader = self.get_dicom_func()

        if dicom_reader is None:
            print("Error: No dicom reader")
            return 0.0
        else:
            try:
                image_data = dicom_reader.GetOutput()
                origin = image_data.GetOrigin()  # Get the origin
                spacing = image_data.GetSpacing()
                
                # Calculate height considering origin
                z_height = float(slice_index * spacing[2] + origin[2])
                return z_height
            except Exception as e:
                print(f"Error: {e}")
                return 0.0

    def x_height_calc(self, slice_index) -> float:
        dicom_reader = self.get_dicom_func()

        if dicom_reader is None:
            print("Error: No dicom reader")
            return 0.0
        else:
            try:
                image_data = dicom_reader.GetOutput()
                origin = image_data.GetOrigin()  # Get the origin
                spacing = image_data.GetSpacing()
                
                # Calculate height considering origin
                x_height = float(slice_index * spacing[0] + origin[0])
                return x_height
            except Exception as e:
                print(f"Error: {e}")
                return 0.0

    def y_height_calc(self, slice_index) -> float:
        dicom_reader = self.get_dicom_func()

        if dicom_reader is None:
            print("Error: No dicom reader")
            return 0.0
        else:
            try:
                image_data = dicom_reader.GetOutput()
                origin = image_data.GetOrigin()  # Get the origin
                spacing = image_data.GetSpacing()
                
                # Calculate height considering origin
                y_height = float(slice_index * spacing[1] + origin[1])
                return y_height
            except Exception as e:
                print(f"Error: {e}")
                return 0.0