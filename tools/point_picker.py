from vtk import vtkCommand, vtkWorldPointPicker, vtkSphereSource, vtkPolyDataMapper, vtkActor, vtkTextActor3D
import xml.etree.ElementTree as ET
from datetime import datetime

class PointPickingTool:
    def __init__(self, control, state, renderer_2d, renderer_3d, sphere_actor, get_dicom_reader_callback):
        self.ctrl = control
        self.state = state
        self.renderer_2d = renderer_2d
        self.renderer_3d = renderer_3d
        self.sphere_actor = sphere_actor
        self.get_dicom_reader = get_dicom_reader_callback
        self.next_point_id = 1
        
        # Pickers for 2D and 3D
        self.picker_2d = vtkWorldPointPicker()
        self.picker_3d = vtkWorldPointPicker()
        self.point_actors = []
        self.label_actors = []
        self.selected_sphere_actors = []
        self.observer_2d = None
        self.observer_3d = None
    
    # ============ Sub-functions ============
    def enable_picking(self):
        """Enable point picking by adding event observers"""
        interactor_2d = self.renderer_2d.GetRenderWindow().GetInteractor()
        interactor_3d = self.renderer_3d.GetRenderWindow().GetInteractor()
        
        # Add observers for right-click events
        self.observer_2d = interactor_2d.AddObserver(vtkCommand.RightButtonPressEvent, self.on_click_2d)
        self.observer_3d = interactor_3d.AddObserver(vtkCommand.RightButtonPressEvent, self.on_click_3d)
        
    def disable_picking(self):
        """Disable point picking by removing event observers"""
        interactor_2d = self.renderer_2d.GetRenderWindow().GetInteractor()
        interactor_3d = self.renderer_3d.GetRenderWindow().GetInteractor()
        
        # Remove observers if they exist
        if self.observer_2d:
            interactor_2d.RemoveObserver(self.observer_2d)
            self.observer_2d = None
        if self.observer_3d:
            interactor_3d.RemoveObserver(self.observer_3d)
            self.observer_3d = None

    def add_point(self, world_point, source, select=False):
        """Add a new point to the list and optionally select it"""
        point_name = f"{self.next_point_id}"
        self.next_point_id += 1

        point_data = {
            "name": point_name,
            "x": round(world_point[0], 2),
            "y": round(world_point[1], 2),
            "z": round(world_point[2], 2),
            "selected": select,  # Đảm bảo thuộc tính selected được thiết lập đúng
            "source": source
        }

        # Update state with new point - use state assignment for reactivity
        current_points = self.state.picked_points.copy()
        current_points.append(point_data)
        self.state.picked_points = current_points
        
        # Force state update
        self.state.flush()
        
        return point_name

    def update_selected_points(self):
        """Update the selected property in picked_points based on selected_points"""
        
        selected_names = self.state.selected_points if self.state.selected_points else []
        updated_points = []
        
        for point in self.state.picked_points:
            point_copy = point.copy()
            point_copy["selected"] = point["name"] in selected_names
            updated_points.append(point_copy)
        
        # Use state assignment for reactivity
        self.state.picked_points = updated_points
        self.state.flush()  # Force state update after modifying points

    def color_change_pick_points(self):
        dicom_reader = self.get_dicom_reader()
        if not dicom_reader:
            return
        
        img_data = dicom_reader.GetOutput()
        if not img_data:
            return
        
        normal_color = (1.0, 1.0, 0.0)  # Yellow color for normal points
        selected_color = (1.0, 0.0, 0.0)  # Red color for selected points

        # Add spheres and labels for each picked point
        for point in self.state.picked_points:
            if not point or not all(key in point for key in ['x', 'y', 'z']):
                continue
            
            try:
                x, y, z = float(point["x"]), float(point["y"]), float(point["z"])
                
                # Create a sphere at the point location
                sphere_source = vtkSphereSource()
                sphere_source.SetRadius(2)
                sphere_source.SetCenter(x, y, z)
                sphere_source.SetPhiResolution(16)
                sphere_source.SetThetaResolution(16)
                
                sphere_mapper = vtkPolyDataMapper()
                sphere_mapper.SetInputConnection(sphere_source.GetOutputPort())
                
                sphere_actor = vtkActor()
                sphere_actor.SetMapper(sphere_mapper)
                
                # Set color based on selection status
                sphere_actor.GetProperty().SetColor(list(normal_color))

                sphere_actor.GetProperty().SetOpacity(0.7)
                
                self.renderer_3d.AddActor(sphere_actor)
                self.point_actors.append(sphere_actor)
                
                # Create label for this point
                self.create_label(point)
                
            except (ValueError, TypeError):
                continue
        
        self.ctrl.view_update_3d()
    
    def hide_pick_points(self):
        """Ẩn tất cả các sphere point và label đã được thêm vào renderer_3d"""
        for actor in self.point_actors:
            self.renderer_3d.RemoveActor(actor)
        for actor in self.label_actors:
            self.renderer_3d.RemoveActor(actor)
        for actor in self.selected_sphere_actors:
            self.renderer_3d.RemoveActor(actor)
            
        self.point_actors.clear()  # Xóa danh sách sphere actors
        self.label_actors.clear()  # Xóa danh sách label actors
        self.selected_sphere_actors.clear()  # Xóa danh sách selected sphere actors
        self.ctrl.view_update_3d()

    def recreate_all_points(self):
        """Recreate all points from the state"""
        self.hide_pick_points()  # Clear existing points
        self.color_change_pick_points()  # Recreate all points
        self.create_selected_sphere()  # Recreate selected spheres

    def pp_file_format(self, point_list):
        """Return points in XML format as string"""
        try:
            # Tạo phần tử gốc
            root = ET.Element("PickedPoints")
            root.text = "\n"

            # Tạo phần tử DocumentData
            doc_data = ET.SubElement(root, "DocumentData")
            doc_data.text = "\n  "
            doc_data.tail = "\n"

            # Thêm các phần tử con.
            ET.SubElement(doc_data, "DateTime", {"date": datetime.now().strftime("%d-%m-%Y"), "time": datetime.now().strftime("%H:%M:%S")}).tail = "\n  "
            ET.SubElement(doc_data, "User", {"name": "Guest"}).tail = "\n  "
            ET.SubElement(doc_data, "DataFileName", {"name": "Temp"}).tail = "\n  "
            ET.SubElement(doc_data, "templateName", {"name": ""}).tail = "\n"
            for point in point_list:
                ET.SubElement(root, "point", {
                    "name": point["name"],
                    "y": str(point["y"]),
                    "z": str(point["z"]),
                    "activate": "1" if point["selected"] else "0",
                    "x": str(point["x"]),
                }).tail = "\n"
            # Chuyển cây XML thành chuỗi (không có khai báo XML)
            xml_str = ET.tostring(root, encoding="unicode")

            # Thêm dòng DOCTYPE thủ công
            doctype = "<!DOCTYPE PickedPoints>\n"
            final_content = doctype + xml_str

            return final_content

        except Exception as e:
            print(f"Error creating .pp content: {e}")
            return None
    
    def load_pp_file(self, content):
        try:
            # Parse XML từ string thay vì file
            root = ET.fromstring(content)
            loaded_points = []
            for point_elem in root.findall("point"):
                name = point_elem.get("name")
                x = float(point_elem.get("x", "0"))
                y = float(point_elem.get("y", "0"))
                z = float(point_elem.get("z", "0"))
                selected = point_elem.get("activate") == "1"
                source = "PP File"

                loaded_points.append({
                    "name": name,
                    "x": round(x, 2),
                    "y": round(y, 2),
                    "z": round(z, 2),
                    "selected": selected,
                    "source": source
                })

            self.next_point_id = max(self.next_point_id, int(name) + 1) # type: ignore
            return loaded_points
        except Exception as e:
            print(f"Error loading .pp file from content: {e}")
            return []
            
    def create_label(self, point):
        """Create a label for a specific point"""
        label_actor = vtkTextActor3D()
        label_actor.SetInput(f"{point['name']}")
        x = float(point["x"])
        y = float(point["y"])
        z = float(point["z"]) + 3  # Offset to place label above the point
        label_actor.SetPosition(x, y, z)
        
        # Set label properties (customize as needed)
        text_property = label_actor.GetTextProperty()
        text_property.SetFontSize(8)
        text_property.SetColor(1, 1, 1)  # White color
        text_property.SetBackgroundColor(0, 0, 0)  # Black background
        text_property.SetBackgroundOpacity(0.5)
        
        self.renderer_3d.AddActor(label_actor)
        self.label_actors.append(label_actor)
        return label_actor

    def create_selected_sphere(self):
        """Create or update selected sphere indicators"""
        # First remove any existing selected spheres
        for actor in self.selected_sphere_actors:
            self.renderer_3d.RemoveActor(actor)
        self.selected_sphere_actors.clear()
        
        for point in self.state.selected_points:
            if point or all(key in point for key in ['x', 'y', 'z']):
                try:
                    x, y, z = float(point["x"]), float(point["y"]), float(point["z"])
                    
                    # Create a sphere at the point location
                    sphere_source = vtkSphereSource()
                    sphere_source.SetRadius(3)  # Larger radius for selected points
                    sphere_source.SetCenter(x, y, z)
                    sphere_source.SetPhiResolution(16)
                    sphere_source.SetThetaResolution(16)
                    
                    sphere_mapper = vtkPolyDataMapper()
                    sphere_mapper.SetInputConnection(sphere_source.GetOutputPort())
                    
                    sphere_actor = vtkActor()
                    sphere_actor.SetMapper(sphere_mapper)
                    
                    # Set color for selected points (red)
                    sphere_actor.GetProperty().SetColor(1.0, 0.0, 0.0)
                    sphere_actor.GetProperty().SetOpacity(0.7)
                    
                    self.renderer_3d.AddActor(sphere_actor)
                    self.selected_sphere_actors.append(sphere_actor)
                    
                except (ValueError, TypeError):
                    continue
        
        self.ctrl.view_update_3d()
        
        self.ctrl.view_update_3d()
    # ============ Main functions ============
    def on_click_2d(self, obj, event):
        """Handle click on 2D DICOM view"""
        if not self.state.point_picking_enabled:
            return

        iren = obj
        mouse_pos = iren.GetEventPosition()

        dicom_reader = self.get_dicom_reader()
        if not self.state.data_loaded or dicom_reader is None:
            return

        # Use picker to get world coordinates
        self.picker_2d.Pick(mouse_pos[0], mouse_pos[1], 0, self.renderer_2d)
        world_point = list(self.picker_2d.GetPickPosition())

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

        print(f"Picked point at {world_point} in {orientation} view")

        # Add point to the list and select it
        point_name = self.add_point(world_point, "DICOM", select=True)
        
        # Update selected points and refresh UI
        # Sửa dòng này: gán trực tiếp danh sách mới thay vì sử dụng append
        self.update_selected_points()  # Đảm bảo cập nhật thuộc tính selected
        self.color_change_pick_points()

        # Update sphere position
        self.sphere_actor.SetPosition(world_point)
        self.sphere_actor.SetVisibility(True)
        
        # Force state update
        self.state.flush()
        self.ctrl.view_update_3d()
        self.ctrl.view_update_2d()  # Add this to update 2D view as well

    def on_click_3d(self, obj, event):
        """Handle click on 3D view"""
        if not self.state.point_picking_enabled:
            return

        iren = obj
        mouse_pos = iren.GetEventPosition()

        # Use picker to get world coordinates
        self.picker_3d.Pick(mouse_pos[0], mouse_pos[1], 0, self.renderer_3d)
        world_point = self.picker_3d.GetPickPosition()

        print(f"Picked point at {world_point} in 3D view")

        # Add point to the list and select it
        point_name = self.add_point(world_point, "3D", select=True)
        
        # Update selected points and refresh UI
        self.update_selected_points()  # Đảm bảo cập nhật thuộc tính selected
        self.color_change_pick_points()

        # Update sphere position
        self.sphere_actor.SetPosition(world_point)
        self.sphere_actor.SetVisibility(True)
        
        # Force state update
        self.state.flush()
        self.ctrl.view_update_3d()

    def delete_point(self, point_name):
        """Delete a point by name"""
        self.state.picked_points = [p for p in self.state.picked_points if p["name"] != point_name]
    
    def delete_selected_points(self):
        """Delete all selected points"""
        if self.state.selected_points and len(self.state.selected_points) > 0:
            selected_names = self.state.selected_points
            self.state.picked_points = [p for p in self.state.picked_points if p not in selected_names]
            self.state.selected_points = []
    
    def delete_all_points(self):
        """Delete all points"""
        self.state.picked_points = []
        self.state.selected_points = []
        self.next_point_id = 1
    
    def save_points(self):
        """Save points and return XML content"""
        if not self.state.picked_points:
            print("No points to save")
            return None
        
        try:
            return self.pp_file_format(self.state.picked_points)
        except Exception as e:
            print(f"Error saving points: {e}")
            return None
    
    def load_points(self, file_content):
        try:
            loaded_points = self.load_pp_file(file_content)
            self.state.picked_points = loaded_points
        except Exception as e:
            print(f"Error loading points: {e}")