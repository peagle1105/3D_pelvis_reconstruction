# Import system libraries
import os
from pathlib import Path

# Import trame and vtk modules
from trame.app import get_server
from vtkmodules.vtkRenderingCore import (
    vtkRenderer,
    vtkRenderWindow,
    vtkRenderWindowInteractor,
    vtkVolume,
    vtkVolumeProperty,
    vtkColorTransferFunction,
)
from vtk import vtkPiecewiseFunction, vtkTransformPolyDataFilter, vtkCommand
from vtkmodules.vtkIOImage import vtkDICOMImageReader
from vtkmodules.vtkInteractionStyle import vtkInteractorStyleImage
from vtkmodules.vtkInteractionImage import vtkImageViewer2
from vtkmodules.vtkRenderingVolume import vtkFixedPointVolumeRayCastMapper
from vtkmodules.vtkFiltersSources import vtkSphereSource
from vtkmodules.vtkRenderingCore import vtkActor, vtkPolyDataMapper
import vtkmodules.vtkRenderingOpenGL2
import vtkmodules.vtkInteractionStyle

# Import tools
from tools.slice import Slice
from tools.mouse import Mouse
from tools.voxel import Volume
from tools.file_handler import load_files
from tools.plane import Plane
from tools.view import View
from tools.point_picker import PointPickingTool

#---------------------------------------------------------
# Define constant
#---------------------------------------------------------
temp_path = "./temp_folder/"
path = Path(temp_path)
os.makedirs(path, exist_ok=True)

# Global VTK objects
from config.config import (
    viewer_2d,
)
volume_3d = None
dicom_reader = None
renderer_2d = None

#---------------------------------------------------------
# Create server
#---------------------------------------------------------
server = get_server(client_type="vue2")
ctrl = server.controller # type: ignore
state = server.state # type: ignore

# Controller functions for increment/decrement
slice = Slice(ctrl, state)
ctrl.add("increment_slice")(slice.increment_slice)
ctrl.add("decrement_slice")(slice.decrement_slice)

# Add shared state variables
state.uploaded_files = None
state.data_loaded = False
state.slice_index = 0
state.slice_min = 0
state.slice_max = 100
state.current_view = "axial"  # Track current view orientation
state.opacity = 0.5  # Volume rendering opacity
state.z_height = 0.0
state.x_height = 0.0  # Add x position for sagittal view
state.y_height = 0.0  # Add y position for coronal view
state.point_picking_enabled = False  # Track if point picking is enabled
state.picked_points = []  # Store picked points
state.selected_points = []  # Store selected points for operations
state.point_picking_mode = False  # Track if we are in point picking mode
state.point_dialog = False
state.status = ""
state.picked_points_content = ""  # Content to be saved in the file
state.file_content = ""  # Content of the uploaded .pp file

#---------------------------------------------------------
# Rendering setup
#---------------------------------------------------------
# Create separate renderers for 2D and 3D views
renderer_3d = vtkRenderer()
renderer_3d.SetBackground(0.8, 0.8, 0.92)

renderer_2d = vtkRenderer()
renderer_2d.SetBackground(0,0,0)

# Create separate render windows
render_window_3d = vtkRenderWindow()
render_window_3d.AddRenderer(renderer_3d)

render_window_2d = vtkRenderWindow()
render_window_2d.AddRenderer(renderer_2d)

# Create interactors with different styles
interactor_3d = vtkRenderWindowInteractor()
interactor_3d.SetRenderWindow(render_window_3d)
interactor_3d.GetInteractorStyle().SetCurrentStyleToTrackballCamera() # type: ignore

interactor_2d = vtkRenderWindowInteractor()
interactor_2d.SetRenderWindow(render_window_2d)
interactor_2d.SetInteractorStyle(vtkInteractorStyleImage()) # Use image-specific interactor style

#---------------------------------------------------------
# Element renderer
#---------------------------------------------------------
# ===== Mouse ======
sphere_source = vtkSphereSource()
sphere_source.SetRadius(2)  # Adjust size based on data
sphere_source.SetPhiResolution(16)
sphere_source.SetThetaResolution(16)

sphere_mapper = vtkPolyDataMapper()
sphere_mapper.SetInputConnection(sphere_source.GetOutputPort())

sphere_actor = vtkActor()
sphere_actor.SetMapper(sphere_mapper)
sphere_actor.GetProperty().SetColor(1, 1, 1)  # White color
sphere_actor.SetVisibility(False)  # Hidden initially

renderer_3d.AddActor(sphere_actor)

# ===== Plane =====
plane_source = vtkTransformPolyDataFilter()

plane_mapper = vtkPolyDataMapper()
plane_mapper.SetInputConnection(plane_source.GetOutputPort())

plane_actor = vtkActor()
plane_actor.SetMapper(plane_mapper)
plane_actor.GetProperty().SetCoatColor(0.765, 0.949, 0.741)
plane_actor.GetProperty().SetLineWidth(2)
plane_actor.GetProperty().SetOpacity(0.5)

renderer_3d.AddActor(plane_actor)

#---------------------------------------------------------
# Tools initialization
#---------------------------------------------------------
# get dicom reader function
def get_dicom_reader():
    return dicom_reader

# ===== Mouse ======
mouse = Mouse(
    control=ctrl, 
    state=state,
    renderer_2d=renderer_2d,
    sphere_actor=sphere_actor,
    get_dicom_reader_callback = get_dicom_reader
)
# ===== Volume =====
volume = Volume(
    control= ctrl,
    state= state,
    get_dicom_reader_callback= get_dicom_reader,
)
# ===== Plane =====
plane = Plane(
    control= ctrl,
    state= state,
    plane_source= plane_source,
    volume= volume,
    get_dicom_reader_callback= get_dicom_reader
)
# ===== View =====
view = View(
    control = ctrl,
    state= state,
    get_dicom_reader_callback = get_dicom_reader,
    plane= plane,
    renderer_2d= renderer_2d
)

# ===== Point Picking Tool =====
point_picker = PointPickingTool(
    control=ctrl,
    state=state,
    renderer_2d=renderer_2d,
    renderer_3d=renderer_3d,
    sphere_actor=sphere_actor,
    get_dicom_reader_callback=get_dicom_reader
)

# Add observer for mouse movement
interactor_2d.AddObserver(vtkCommand.MouseMoveEvent, mouse.on_mouse_move)

# Add controller functions for point picking
ctrl.add("toggle_point_picking")(lambda: setattr(state, "point_picking_enabled", not state.point_picking_enabled))
ctrl.add("delete_selected_points")(point_picker.delete_selected_points)
ctrl.add("delete_all_points")(point_picker.delete_all_points)
ctrl.add("save_points")(point_picker.save_points)
ctrl.add("load_points")(point_picker.load_points)
@ctrl.add("upload_new_series")
def upload_new_series():
    """Xóa toàn bộ file trong temp_folder và reset state"""
    # Clear uploaded files
    state.uploaded_files = None
    state.data_loaded = False
    state.slice_index = 0
    
    # Clear renderers but preserve essential elements
    renderer_3d.RemoveAllViewProps()
    renderer_2d.RemoveAllViewProps() # type: ignore
    
    # Re-add essential elements to renderers
    renderer_3d.AddActor(sphere_actor)
    renderer_3d.AddActor(plane_actor)
    
    # Reset global variables
    global viewer_2d, volume_3d, dicom_reader
    viewer_2d = None
    volume_3d = None
    dicom_reader = None
    
    # Hide sphere and plane until new data is loaded
    sphere_actor.SetVisibility(False)
    plane_actor.SetVisibility(False)
    
    # Update views
    ctrl.view_update_2d()
    ctrl.view_update_3d()

    # Clear temp directory
    try:
        for f in path.iterdir():
            if f.is_file():
                f.unlink()
    except Exception as e:
        print(f"❌ Failed to clear temp folder: {e}")

#---------------------------------------------------------
# Callback functions
#---------------------------------------------------------
# ===== Slice change ====
@state.change("slice_index")
def on_slice_change(slice_index, **kwargs):
    global viewer_2d
    if not state.data_loaded or viewer_2d is None:
        return
    
    # Cập nhật slice trong vtkImageViewer2
    viewer_2d.SetSlice(int(slice_index))
    
    # Cập nhật vị trí mặt phẳng
    plane.update_plane_size()
    plane.update_plane_position()
    
    # Cập nhật sphere theo slice mới
    sphere_actor.SetVisibility(True)
    ctrl.view_update_2d()
    ctrl.view_update_3d()

# ===== View orientation change =====
@state.change("current_view")
def on_view_change(current_view, **kwargs):
    global viewer_2d, dicom_reader
    if not state.data_loaded or viewer_2d is None:
        return
    if dicom_reader is not None:
        output = dicom_reader.GetOutput()
        dims = output.GetDimensions()
        
        max_slice = 0
        if current_view == "axial":
            max_slice = dims[2] - 1
        elif current_view == "sagittal":
            max_slice = dims[0] - 1
        elif current_view == "coronal":
            max_slice = dims[1] - 1

        state.slice_max = max_slice
        state.slice_index = max_slice
        if current_view == "axial":
            viewer_2d.SetSliceOrientationToXY()
        elif current_view == "sagittal":
            viewer_2d.SetSliceOrientationToYZ()
        else:
            viewer_2d.SetSliceOrientationToXZ()
        plane.update_plane_size()
        plane.update_plane_position()
        ctrl.view_update_2d()
    else:
        print("❌ dicom_reader is None in on_view_change")

# ===== Opacity change =====
@state.change("opacity")
def on_opacity_change(opacity, **kwargs):
    if state.data_loaded:
        volume.update_volume_rendering(volume_3d= volume_3d)

# ===== Point Picking =====
@state.change("point_picking_mode")
def on_point_picking_mode_change(point_picking_mode, **kwargs):
    if point_picking_mode:
        # Enable picking and show points
        point_picker.enable_picking()
        point_picker.color_change_pick_points()
        ctrl.view_update_2d()
        ctrl.view_update_3d()
    else:
        # Disable picking and hide points
        point_picker.disable_picking()
        point_picker.hide_pick_points()
        ctrl.view_update_2d()
        ctrl.view_update_3d()

@state.change("picked_points")
def on_picked_points_change(**kwargs):
    if state.point_picking_mode:
        point_picker.recreate_all_points()
        state.picked_points_content = point_picker.save_points()
        ctrl.view_update_3d()

@state.change("selected_points")
def on_selected_points_change(selected_points, **kwargs):
    # selected_points là list tên điểm (string)
    selected_names = selected_points if selected_points else []
    updated_points = []
    picked_points = state.picked_points if state.picked_points else []
    # Cập nhật thuộc tính `selected` trong picked_points
    for point in picked_points:
        point_copy = point.copy()
        point_copy["selected"] = point["name"] in selected_names
        updated_points.append(point_copy)
    
    # Gán lại picked_points để kích hoạt reactivity
    state.picked_points = updated_points
    
    # Update selected spheres
    point_picker.create_selected_sphere()
    ctrl.view_update_3d()

    # In số lượng điểm đang được chọn
    print(len(selected_names))

@state.change("file_content")
def on_file_content_change(file_content, **kwargs):
    """Xử lý khi nội dung file thay đổi"""
    if file_content:
        point_picker.load_points(file_content)
        # Reset state sau khi xử lý
        state.file_content = None

#---------------------------------------------------------
# VTK Pipeline
#---------------------------------------------------------
@state.change("uploaded_files")
def on_file_upload(uploaded_files, **kwargs):
    global viewer_2d, volume_3d, dicom_reader, renderer_2d
    
    # Handle file clearance
    if not uploaded_files:
        if volume_3d:
            renderer_3d.RemoveVolume(volume_3d)
            volume_3d = None
        if viewer_2d:
            renderer_2d.RemoveActor(viewer_2d.GetImageActor()) # type: ignore
            viewer_2d = None
        dicom_reader = None
        state.data_loaded = False
        sphere_actor.SetVisibility(False)  # Hide sphere when no data
        plane_actor.SetVisibility(False)   # Hide plane when no data
        ctrl.view_update_2d()
        ctrl.view_update_3d()
        return

    # Clean temp directory
    try:
        for f in path.iterdir():
            if f.is_file():
                f.unlink()
    except Exception as e:
        print(f"❌ Failed to clear temp folder: {e}")
        return

    # Save new files
    load_files(temp_path= path, file_list= uploaded_files)

    # Clear previous objects
    if volume_3d:
        renderer_3d.RemoveVolume(volume_3d)
        volume_3d = None
    if viewer_2d:
        renderer_2d.RemoveActor(viewer_2d.GetImageActor()) # type: ignore
        viewer_2d = None
    dicom_reader = None

    try:
        # Read DICOM series
        dicom_reader = vtkDICOMImageReader()
        dicom_reader.SetDirectoryName(str(path))
        dicom_reader.Update()
        output = dicom_reader.GetOutput()

        if not output or output.GetNumberOfPoints() == 0:
            print("❌ No DICOM data loaded")
            state.data_loaded = False
            ctrl.view_update_2d()
            ctrl.view_update_3d()
            return

        # =============== 3D Volume Pipeline ===============
        # Create volume mapper
        volume_mapper = vtkFixedPointVolumeRayCastMapper()
        volume_mapper.SetInputConnection(dicom_reader.GetOutputPort())
        
        # Get scalar range for transfer functions
        scalar_range = output.GetScalarRange()
        min_val, max_val = scalar_range
        
        # Create color transfer function
        color_function = vtkColorTransferFunction()
        color_function.AddRGBPoint(min_val, 0.0, 0.0, 0.0)   # Black
        color_function.AddRGBPoint((min_val + max_val) * 0.3, 0.0, 0.0, 1.0)  # Blue
        color_function.AddRGBPoint((min_val + max_val) * 0.6, 0.0, 1.0, 0.0)  # Green
        color_function.AddRGBPoint(max_val, 1.0, 1.0, 1.0)    # White
        
        # Create opacity transfer function
        opacity_function = vtkPiecewiseFunction()
        opacity_function.AddPoint(min_val, 0.0)
        opacity_function.AddPoint((min_val + max_val) * 0.3, 0.0)
        opacity_function.AddPoint((min_val + max_val) * 0.5, 0.5 * state.opacity if state.opacity is not None else 0)
        opacity_function.AddPoint((min_val + max_val) * 0.7, state.opacity if state.opacity is not None else 0)
        opacity_function.AddPoint(max_val, state.opacity if state.opacity is not None else 0)
        
        # Create volume properties
        volume_property = vtkVolumeProperty()
        volume_property.SetColor(color_function)
        volume_property.SetScalarOpacity(opacity_function)
        volume_property.ShadeOn()
        volume_property.SetInterpolationTypeToLinear()
        
        # Create volume
        volume_3d = vtkVolume()
        volume_3d.SetMapper(volume_mapper)
        volume_3d.SetProperty(volume_property)
        
        # Add volume to renderer
        renderer_3d.AddVolume(volume_3d)

        # =============== 2D Pipeline ===============
        # Create image viewer
        viewer_2d = vtkImageViewer2()
        viewer_2d.SetInputConnection(dicom_reader.GetOutputPort())
        viewer_2d.SetRenderWindow(render_window_2d)
        viewer_2d.SetRenderer(renderer_2d)
        viewer_2d.SetupInteractor(interactor_2d)
        
        # Get and add the image actor to our renderer
        image_actor = viewer_2d.GetImageActor()
        renderer_2d.AddActor(image_actor) # type: ignore

        # Update slice range
        dims = output.GetDimensions()
        state.slice_max = dims[2] - 1
        state.slice_index = min(0, state.slice_max)
        viewer_2d.SetSlice(state.slice_index)

        # Calculate heights for different views
        state.z_height = volume.z_height_calc(slice_index = state.slice_index)
        state.x_height = volume.x_height_calc(slice_index = state.slice_index)
        state.y_height = volume.y_height_calc(slice_index = state.slice_index)
        
        # Update plane position and size
        plane.update_plane_position()
        plane.update_plane_size()
        
        # Show plane
        plane_actor.SetVisibility(True)

        # Reset cameras
        renderer_2d.ResetCamera() # type: ignore
        renderer_3d.ResetCamera()
        renderer_3d.GetActiveCamera().Azimuth(30)
        renderer_3d.GetActiveCamera().Elevation(30)
        renderer_3d.ResetCameraClippingRange()

        # Set initial view
        state.current_view = "axial"
        view.update_view_orientation()

        state.data_loaded = True

    except Exception as e:
        print(f"❌ Processing error: {e}")
        state.data_loaded = False
        import traceback
        traceback.print_exc()

    ctrl.view_update_2d()
    ctrl.view_update_3d()

def get_trame_objects():
    return server, ctrl, state, render_window_2d, render_window_3d, interactor_2d, interactor_3d