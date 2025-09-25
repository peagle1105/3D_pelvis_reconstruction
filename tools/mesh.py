from vtk import vtkCommand, vtkCellPicker, vtkSphereSource, vtkPolyDataMapper, vtkActor, vtkTextActor3D

class Mesh:
    def __init__(self, ctrl, state, mesh_renderer, sphere_actor, interactor_3d):
        self.ctrl = ctrl
        self.state = state
        self.mesh_renderer = mesh_renderer
        self.sphere_actor = sphere_actor
        self.interactor_3d = interactor_3d  # Lưu interactor_3d
        self.picker_3d = vtkCellPicker()
        self.point_actors = []
        self.label_actors = []
        self.selected_sphere_actors = []
        self.observer_3d = None 
        self.next_vertices_id = 1

    # ===== Sub-function =====
    def enable_point_picking_3d(self):
        """Enable point picking for 3D mesh"""
        if self.interactor_3d is None:
            print("Interactor 3D is not available!")
            return
            
        # Remove existing observer if any
        if self.observer_3d:
            self.interactor_3d.RemoveObserver(self.observer_3d)
            
        # Add new observer
        self.observer_3d = self.interactor_3d.AddObserver(
            vtkCommand.RightButtonPressEvent, self.on_click_3d)
        print("✅ 3D point picking enabled")

    def disable_point_picking_3d(self):
        """Disable point picking by removing event observers"""
        if self.interactor_3d and self.observer_3d:
            self.interactor_3d.RemoveObserver(self.observer_3d)
            self.observer_3d = None
            print("✅ 3D point picking disabled")

    def add_point(self, world_point, source, select=False):
        """Add a new point to the list and optionally select it"""
        vertices_name = f"{self.next_vertices_id}"
        self.next_vertices_id += 1

        vertices_data = {
            "name": vertices_name,
            "x": round(world_point[0], 2),
            "y": round(world_point[1], 2),
            "z": round(world_point[2], 2),
            "selected": select,  # Đảm bảo thuộc tính selected được thiết lập đúng
            "source": source
        }

        # Update state with new point - use state assignment for reactivity
        current_vertices = self.state.picked_vertices.copy()
        current_vertices.append(vertices_data)
        self.state.picked_vertices = current_vertices
        
        # Force state update
        self.state.flush()
        
        return vertices_name

    def update_selected_points(self):
        """Update the selected property in picked_points based on selected_points"""
        
        selected_names = self.state.selected_vertices if self.state.selected_vertices else []
        updated_vertices = []
        
        for vertex in self.state.picked_vertices:
            vertex_copy = vertex.copy()
            vertex_copy["selected"] = vertex["name"] in selected_names
            updated_vertices.append(vertex_copy)
        
        # Use state assignment for reactivity
        self.state.picked_vertices = updated_vertices
        self.state.flush()  # Force state update after modifying points

    def color_change_pick_points(self):
        normal_color = (1.0, 1.0, 0.0)  # Yellow color for normal points

        # Add spheres and labels for each picked point
        for vertex in self.state.picked_vertices:
            if not vertex or not all(key in vertex for key in ['x', 'y', 'z']):
                continue
            
            try:
                x, y, z = float(vertex["x"]), float(vertex["y"]), float(vertex["z"])
                
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
                
                self.mesh_renderer.AddActor(sphere_actor)
                self.point_actors.append(sphere_actor)
                
                # Create label for this point
                self.create_label(vertex)
                
            except (ValueError, TypeError):
                continue
        
        self.ctrl.view_update_3d()
    
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
        text_property.SetColor(0, 0, 0)  # White color
        text_property.SetBackgroundColor(1, 1, 1)  # Black background
        
        self.mesh_renderer.AddActor(label_actor)
        self.label_actors.append(label_actor)
        return label_actor
    
    def create_selected_sphere(self):
        """Create or update selected sphere indicators"""
        # First remove any existing selected spheres
        for actor in self.selected_sphere_actors:
            self.mesh_renderer.RemoveActor(actor)
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
                    
                    self.mesh_renderer.AddActor(sphere_actor)
                    self.selected_sphere_actors.append(sphere_actor)
                    
                except (ValueError, TypeError):
                    continue
        
        self.ctrl.view_update_2d()
        self.ctrl.view_update_3d()

    def hide_pick_points(self):
        """Ẩn tất cả các sphere point và label đã được thêm vào renderer_3d"""
        for actor in self.point_actors:
            self.mesh_renderer.RemoveActor(actor)
        for actor in self.label_actors:
            self.mesh_renderer.RemoveActor(actor)
        for actor in self.selected_sphere_actors:
            self.mesh_renderer.RemoveActor(actor)
            
        self.point_actors.clear()  # Xóa danh sách sphere actors
        self.label_actors.clear()  # Xóa danh sách label actors
        self.selected_sphere_actors.clear()  # Xóa danh sách selected sphere actors
        self.ctrl.view_update_3d()

    def recreate_all_points(self):
        """Recreate all points from the state"""
        self.hide_pick_points()  # Clear existing points
        self.color_change_pick_points()  # Recreate all points
        self.create_selected_sphere()  # Recreate selected spheres

    # ===== Main function =====
    def on_click_3d(self, obj, event):
        """Handle click on 3D view"""
        # ✅ Sửa điều kiện kiểm tra
        if not self.state.create_model_mode:
            return

        iren = obj
        mouse_pos = iren.GetEventPosition()

        # Kiểm tra mesh_renderer có trong render window không
        render_window = self.mesh_renderer.GetRenderWindow()
        if not render_window or self.mesh_renderer not in render_window.GetRenderers():
            print("❌ Mesh renderer not in active render window")
            return

        # Use picker to get world coordinates
        success = self.picker_3d.Pick(mouse_pos[0], mouse_pos[1], 0, self.mesh_renderer)
        if not success:
            print("❌ Pick failed")
            return
            
        world_point = self.picker_3d.GetPickPosition()
        print(f"✅ Picked point at {world_point} in 3D view")

        # Add point to the list and select it
        point_name = self.add_point(world_point, "3D", select=True)
        
        # Update selected points and refresh UI
        self.update_selected_points()
        self.recreate_all_points()  # ✅ Sử dụng hàm recreate thay vì color_change

        # Update sphere position
        self.sphere_actor.SetPosition(world_point)
        self.sphere_actor.SetVisibility(True)
        
        # Force state update
        self.state.flush()
        self.ctrl.view_update_3d()

    def delete_vertices(self, vertex_name):
        """Delete a point by name"""
        self.state.picked_points = [p for p in self.state.picked_points if p["name"] != vertex_name]
    
    def delete_selected_vertices(self):
        """Delete all selected points"""
        if self.state.selected_vertices and len(self.state.selected_vertices) > 0:
            selected_names = self.state.selected_vertices
            self.state.picked_vertices = [p for p in self.state.picked_vertices if p not in selected_names]
            self.state.selected_vertices = []
    
    def delete_all_vertices(self):
        """Delete all points"""
        self.state.picked_vertices = []
        self.state.selected_vertices = []
        self.next_vertices_id = 1