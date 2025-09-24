import os
import base64
from pipeline import get_trame_objects

server, ctrl, state, render_window_2d, render_window_3d, interactor_2d, interactor_3d = get_trame_objects()

from trame.ui.vuetify import SinglePageLayout
from trame.widgets import vtk, vuetify

state.header = [
    {'text': 'Name', 'value': 'name', 'width': '30%'},
    {'text': 'X', 'value': 'x', 'width': '15%'},
    {'text': 'Y', 'value': 'y', 'width': '15%'},
    {'text': 'Z', 'value': 'z', 'width': '15%'},
    {'text': 'Source', 'value': 'source', 'width': '25%'}
]
logo_path = "./static/logo.png"
with open(logo_path, "rb") as f:
    img_data = f.read()
img_base64 = base64.b64encode(img_data).decode('utf-8')
#---------------------------------------------------------
# UI Setup
#---------------------------------------------------------
with SinglePageLayout(server, drawer = None) as layout:
    layout.title.set_text("DICOM 3D Viewer")
    with layout.icon:
        from trame.widgets import html
        html.Img(
            src=f"data:image/png;base64,{img_base64}",
            style="height: 50px !important; width: auto !important; margin-left: 10px !important;"
        )
    # Main content with custom structure
    with layout.content:
        # Custom toolbar placed right below the title
        with vuetify.VToolbar(flat=True, dense=True, style="border-bottom: 1px solid #e0e0e0; margin-bottom: 16px;", v_if=("data_loaded")):
            # File menu
            with vuetify.VMenu(offset_y=True, open_on_hover=True):
                with vuetify.Template(v_slot_activator="{ on, attrs }"):
                    vuetify.VBtn(
                        "File",
                        v_bind="attrs",
                        v_on="on",
                        text=True,
                        classes ="text-none px-3",
                        style="min-width: auto; border-radius: 0; height: 36px;",
                    )
                with vuetify.VList(dense=True):
                    vuetify.VListItem(
                        "new series",
                        click=ctrl.upload_new_series,
                        style = "font-size: 14px;"
                    )
                    vuetify.VListItem(
                        "export mesh",
                        click="export_dialog = true",
                        style = "font-size: 14px;"
                    )
            
            # View menu
            with vuetify.VMenu(offset_y=True, open_on_hover=True):
                with vuetify.Template(v_slot_activator="{ on, attrs }"):
                    vuetify.VBtn(
                        "View",
                        v_bind="attrs",
                        v_on="on",
                        text=True,
                        classes ="text-none px-3",
                        style="min-width: auto; border-radius: 0; height: 36px;",
                    )
                with vuetify.VList(dense=True):
                    with vuetify.VList(v_model=("current_view", "axial"), classes="ma-2"):
                        vuetify.VListItem("axial", value="axial", click = "current_view = 'axial'", active_class="primary--text", style = "font-size: 14px;")
                        vuetify.VListItem("sagittal", value="sagittal", click = "current_view = 'sagittal'", active_class="primary--text", style = "font-size: 14px;")
                        vuetify.VListItem("coronal", value="coronal", click = "current_view = 'coronal'", active_class="primary--text", style = "font-size: 14px;")

            # Edit menu
            with vuetify.VMenu(offset_y=True, open_on_hover=True):
                with vuetify.Template(v_slot_activator="{ on, attrs }"):
                    vuetify.VBtn(
                        "Model",
                        v_bind="attrs",
                        v_on="on",
                        text=True,
                        classes ="text-none px-3",
                        style="min-width: auto; border-radius: 0; height: 36px;",
                    )
                with vuetify.VList(dense=True):
                    vuetify.VListItem(
                        "create model",
                        click = "point_picking_mode = false; point_picking_enabled = false, create_model_mode = true",
                        style = "font-size: 14px;"
                    )
                    vuetify.VListItem(
                        "reconstruct",
                        click="point_picking_mode = true; point_picking_enabled = true, create_model_mode = false",
                        style = "font-size: 14px;"
                    )
        
        # Dialog for export mesh
        with vuetify.VDialog(v_model=("export_dialog", False), width="500", persistent=True):
            with vuetify.VCard():
                with vuetify.VCardTitle(
                    "Export Mesh", 
                    classes="text-h5 grey lighten-2 d-flex justify-space-between align-center",
                    style="padding: 16px;"
                ):
                    vuetify.VIcon("mdi-download", color="primary", style="font-size: 24px;")
                    vuetify.VSpacer()
                    vuetify.VIcon(
                        "mdi-close", 
                        click="export_dialog = false",
                        style="cursor: pointer;"
                    )

                with vuetify.VCardText(classes="pa-4"):
                    vuetify.VTextField(
                        label="File name",
                        v_model=("file_mesh_name",""),
                        hide_details=True,
                        outlined=True,
                        dense=True,
                        placeholder="Enter file name",
                        prepend_icon="mdi-file-outline",
                        classes="mb-4"
                    )
                    vuetify.VSelect(
                        label="File format",
                        v_model=("file_mesh_extend","PLY"),
                        items=("['PLY', 'OBJ', 'STL']",),
                        hide_details=True,
                        outlined=True,
                        dense=True,
                        prepend_icon="mdi-format-list-bulleted-type",
                        classes="mb-2"
                    )

                vuetify.VDivider()
                
                with vuetify.VCardActions(classes="pa-4 d-flex justify-end"):
                    vuetify.VBtn(
                        "Cancel", 
                        click="export_dialog = false",
                        color="grey",
                        text=True,
                        classes="mr-2",
                    )
                    vuetify.VBtn(
                        "Save", 
                        click= ctrl.save_file(),
                        disabled=(" file_mesh_name == '' ",),
                        color="primary",
                        depressed=True
                    )
        
        # Content area with DICOM viewer
        with vuetify.VContainer(
            v_if=("data_loaded", False),
            fluid=True,
            classes="pa-0 fill-height",
        ):
            # Normal layout (2 columns)
            with vuetify.VRow(no_gutters=True, classes="fill-height", v_if=("!point_picking_mode && !create_model_mode",)):
                # Left column - 2D view with controls
                with vuetify.VCol(cols=6, classes="pa-2 d-flex flex-column"):
                    # 2D View
                    with vuetify.VCard(classes="flex-grow-1", style="position: relative;"):
                        view_2d = vtk.VtkRemoteView(
                            render_window_2d,
                            interactor=interactor_2d,
                            style="width: 100%; height: 100%;",
                        )
                        ctrl.view_update_2d = view_2d.update
                    
                    # Slider and controls
                    with vuetify.VCard(classes="mt-2 pa-2"):
                        with vuetify.VRow(align="center", no_gutters=True):
                            with vuetify.VCol(cols=1):
                                with vuetify.VBtn(
                                    icon=True,
                                    click=ctrl.decrement_slice,
                                    disabled=("!data_loaded || slice_index <= slice_min",),
                                    classes="mr-1",
                                ): vuetify.VIcon("mdi-minus")
                            
                            with vuetify.VCol(cols=8):
                                vuetify.VSlider(
                                    v_model=("slice_index",),
                                    min=("slice_min", 0),
                                    max=("slice_max", 100),
                                    hide_details=True,
                                    dense=True,
                                    disabled=("!data_loaded",),
                                )
                            
                            with vuetify.VCol(cols=1):
                                with vuetify.VBtn(
                                    icon=True,
                                    click=ctrl.increment_slice,
                                    disabled=("!data_loaded || slice_index >= slice_max",),
                                    classes="ml-1",
                                ):
                                    vuetify.VIcon("mdi-plus")
                            
                            with vuetify.VCol(cols=2):
                                vuetify.VTextField(
                                    value=("`${slice_index}/${slice_max}`",),
                                    outlined=True,
                                    dense=True,
                                    hide_details=True,
                                    readonly = True,
                                    disabled=("!data_loaded",),
                                    style="max-width: 90px;",
                                    classes="ml-2",
                                )
                        
                        with vuetify.VRow(align="center", no_gutters=True, classes="mt-2"):
                            with vuetify.VCol(cols=3):
                                vuetify.VItem("Opacity:")
                            
                            with vuetify.VCol(cols=9):
                                vuetify.VSlider(
                                    v_model=("opacity", 0.5),
                                    min=0.1,
                                    max=1.0,
                                    step=0.05,
                                    hide_details=True,
                                    thumb_label=True,
                                    dense=True,
                                    disabled=("!data_loaded",),
                                )
                
                # Right column - 3D view and point management
                with vuetify.VCol(cols=6, classes="pa-2 d-flex flex-column"):
                    # 3D view
                    with vuetify.VCard(classes="flex-grow-1"):
                        view_3d = vtk.VtkRemoteView(
                            render_window_3d,
                            interactor=interactor_3d,
                            style="width: 100%; height: 100%;",
                        )
                        ctrl.view_update_3d = view_3d.update
                        ctrl.on_server_ready.add(view_3d.update)

            # Reconstruct layout (3 columns)
            with vuetify.VRow(no_gutters=True, classes="fill-height", v_if=("point_picking_mode",)):
                # DICOM view (3 columns)
                with vuetify.VCol(cols=5, classes="pa-2 d-flex flex-column"):
                    with vuetify.VCard(classes="flex-grow-1", style="position: relative;"):
                        view_2d = vtk.VtkRemoteView(
                            render_window_2d,
                            interactor=interactor_2d,
                            style="width: 100%; height: 100%;",
                        )
                        ctrl.view_update_2d = view_2d.update
                    with vuetify.VCard(classes="mt-2 pa-2"):
                        with vuetify.VRow(align="center", no_gutters=True):
                            with vuetify.VCol(cols=1):
                                with vuetify.VBtn(
                                    icon=True,
                                    click=ctrl.decrement_slice,
                                    disabled=("!data_loaded || slice_index <= slice_min",),
                                    classes="mr-1",
                                ): vuetify.VIcon("mdi-minus")
                            
                            with vuetify.VCol(cols=8):
                                vuetify.VSlider(
                                    v_model=("slice_index",),
                                    min=("slice_min", 0),
                                    max=("slice_max", 100),
                                    hide_details=True,
                                    dense=True,
                                    disabled=("!data_loaded",),
                                )
                            
                            with vuetify.VCol(cols=1):
                                with vuetify.VBtn(
                                    icon=True,
                                    click=ctrl.increment_slice,
                                    disabled=("!data_loaded || slice_index >= slice_max",),
                                    classes="ml-1",
                                ):
                                    vuetify.VIcon("mdi-plus")
                            
                            with vuetify.VCol(cols=2):
                                vuetify.VTextField(
                                    v_model_number=("slice_index",),
                                    type="number",
                                    outlined=True,
                                    dense=True,
                                    hide_details=True,
                                    disabled=("!data_loaded",),
                                    style="max-width: 80px;",
                                    classes="ml-2",
                                )
                        
                        with vuetify.VRow(align="center", no_gutters=True, classes="mt-2"):
                            with vuetify.VCol(cols=2):
                                vuetify.VItem(children=["Opacity:"])
                            
                            with vuetify.VCol(cols=7):
                                vuetify.VSlider(
                                    v_model=("opacity", 0.5),
                                    min=0.1,
                                    max=1.0,
                                    step=0.05,
                                    hide_details=True,
                                    thumb_label=True,
                                    dense=True,
                                    disabled=("!data_loaded",),
                                )

                # 3D view (5 columns)
                with vuetify.VCol(cols=5, classes="pa-2 d-flex flex-column"):
                    with vuetify.VCard(classes="flex-grow-1"):
                        view_3d = vtk.VtkRemoteView(
                            render_window_3d,
                            interactor=interactor_3d,
                            style="width: 100%; height: 100%;",
                        )
                        ctrl.view_update_3d = view_3d.update

                # Point management panel (2 columns)
                with vuetify.VCol(cols=2, classes="pa-2 d-flex flex-column"):
                    with vuetify.VCard(classes="d-flex flex-column", style="height: 100%;"):
                        with vuetify.VCardTitle(classes="py-2 primary white--text d-flex align-center"):
                            vuetify.VIcon("mdi-map-marker", left=True, dark=True)
                            vuetify.VCardTitle("Point Management", classes="text-body-1 white--text")
                            vuetify.VSpacer()
                            with vuetify.VBtn(
                                icon=True,
                                dark=True,
                                click="point_picking_mode = false; point_picking_enabled = false; create_model_mode = false",
                                classes="ml-2",
                                small=True,
                            ): 
                                vuetify.VIcon("mdi-close")
                        
                        # Point list with fixed height
                        with vuetify.VCardText(classes="pa-2 flex-grow-1", style="overflow-y: auto; height: calc(100% - 120px);"):
                            with vuetify.VContainer(fluid=True, style="height: 400px;"):
                                vuetify.VDataTable(
                                    headers=("header",),
                                    items=("picked_points",),
                                    v_model=("selected_points", []),
                                    item_key="name",
                                    show_select=True,
                                    hide_default_footer=True,
                                    dense=True,
                                    height="100%",
                                    style="height: 100%;",
                                    classes="point-table",
                                )

                        # Button actions with improved layout
                        with vuetify.VCardActions(classes="pa-2 pt-0"):
                            with vuetify.VContainer(fluid=True, classes="pa-0"):
                                with vuetify.VRow(dense=True, no_gutters=True):
                                    with vuetify.VCol(cols=6, classes="pr-1"):
                                        vuetify.VBtn(
                                            "Del Sel",
                                            click= ctrl.delete_selected_points,
                                            small=True,
                                            block=True,
                                            color="error",
                                            disabled=("!selected_points || selected_points.length === 0",),
                                            style="font-size: 11px;",
                                        )
                                    with vuetify.VCol(cols=6, classes="pl-1"):
                                        vuetify.VBtn(
                                            "Del All",
                                            click=ctrl.delete_all_points,
                                            small=True,
                                            block=True,
                                            color="error",
                                            disabled=("!picked_points || picked_points.length === 0",),
                                            style="font-size: 11px;",
                                        )
                                
                                with vuetify.VRow(dense=True, no_gutters=True, classes="mt-1"):
                                    with vuetify.VCol(cols=6, classes="pr-1"):
                                        vuetify.VBtn(
                                            "Save",
                                            small=True,
                                            block=True,
                                            color="success",
                                            disabled=("!picked_points || picked_points.length === 0",),
                                            style="font-size: 11px;",
                                            click=(
                                                "const content = picked_points_content;"
                                                "const blob = new Blob([content], { type: 'text/plain' });"
                                                "const url = URL.createObjectURL(blob);"
                                                "const link = document.createElement('a');"
                                                "link.href = url;"
                                                "link.download = 'picked_points.pp';"
                                                "document.body.appendChild(link);"
                                                "link.click();"
                                                "document.body.removeChild(link);"
                                                "URL.revokeObjectURL(url);"
                                                "status = '✅ Đã tải về thành công!';"
                                            ),
                                        )

                                    with vuetify.VCol(cols=6, classes="pl-1"):
                                        vuetify.VBtn(
                                            "Load",
                                            click="""$refs.fileInput.$el.querySelector('input[type=file]').click()""", 
                                            small=True,
                                            block=True,
                                            color="success",
                                            style="font-size: 11px;",
                                        )

                                    # Thêm ref cho VFileInput
                                    vuetify.VFileInput(
                                        v_model=("points_file", None),
                                        ref="fileInput",  # Thêm ref
                                        style="display: none;",
                                        multiple=False,
                                        accept=".pp",
                                        change=(
                                            "function(file) {"
                                            "  if (file) {"
                                            "    const reader = new FileReader();"
                                            "    reader.onload = function(e) {"
                                            "      set('file_content', e.target.result);" 
                                            "    };"
                                            "    reader.readAsText(file);"
                                            "  }"
                                            "}"
                                        ),
                                    )
                                with vuetify.VRow(dense=True, no_gutters=True, classes="mt-1"):
                                    vuetify.VBtn(
                                        "Reconstruct",
                                        click="",
                                        small=True,
                                        block=True,
                                        color="primary",
                                        disabled=("!picked_points || picked_points.length === 0",),
                                        style="font-size: 11px;",
                                    )

            # Create model layout
            with vuetify.VRow(no_gutters=True, classes="fill-height", v_if=("create_model_mode",)):
                # Windows containing mesh
                with vuetify.VCol(cols=9, classes="pa-0 d-flex flex-column"):
                    with vuetify.VCard(classes="flex-grow-1"):
                        view_3d = vtk.VtkRemoteView(
                            render_window_3d,
                            interactor=interactor_3d,
                            style="width: 100%; height: 100%;",
                        )
                        ctrl.view_update_3d = view_3d.update
                # Functioning part
                with vuetify.VCol(cols=3, classes="pa-0 d-flex flex-column"):
                    with vuetify.VCard(classes="d-flex flex-column", style="height: 100%;"):
                        with vuetify.VCardTitle(classes="py-2 primary white--text d-flex align-center"):
                            vuetify.VIcon("mdi-map-marker", left=True, dark=True)
                            vuetify.VCardTitle("Vertices Management", classes="text-body-1 white--text")
                            vuetify.VSpacer()
                            with vuetify.VBtn(
                                icon=True,
                                dark=True,
                                click="point_picking_mode = false; point_picking_enabled = false; create_model_mode = false",
                                classes="ml-2",
                                small=True,
                            ): 
                                vuetify.VIcon("mdi-close")
                        
                        # Point list with fixed height
                        with vuetify.VCardText(classes="pa-2 flex-grow-1", style="overflow-y: auto; height: calc(100% - 120px);"):
                            with vuetify.VContainer(fluid=True, style="height: 400px;"):
                                vuetify.VDataTable(
                                    headers=("header",),
                                    items=("picked_vertices",),
                                    v_model=("selected_vertices", []),
                                    item_key="name",
                                    show_select=True,
                                    hide_default_footer=True,
                                    dense=True,
                                    height="100%",
                                    style="height: 100%;",
                                    classes="point-table",
                                )
                                

                        # Button actions with improved layout
                        with vuetify.VCardActions(classes="pa-2 pt-0"):
                            with vuetify.VContainer(fluid=True, classes="pa-0"):
                                with vuetify.VRow(dense=True, no_gutters=True):
                                    with vuetify.VCol(cols=6, classes="pr-1"):
                                        vuetify.VBtn(
                                            "Del Sel",
                                            click= "",
                                            small=True,
                                            block=True,
                                            color="error",
                                            disabled=("!selected_vertices || selected_vertices.length === 0",),
                                            style="font-size: 11px;",
                                        )
                                    with vuetify.VCol(cols=6, classes="pl-1"):
                                        vuetify.VBtn(
                                            "Del All",
                                            click="",
                                            small=True,
                                            block=True,
                                            color="error",
                                            disabled=("!picked_vertices || picked_vertices.length === 0",),
                                            style="font-size: 11px;",
                                        )
                                
                                with vuetify.VRow(dense=True, no_gutters=True, classes="mt-1"):
                                    vuetify.VBtn(
                                        "Train",
                                        small=True,
                                        block=True,
                                        color="success",
                                        disabled=("!picked_vertices || picked_vertices.length === 0",),
                                        style="font-size: 11px;",
                                        click="",
                                    )
                                with vuetify.VRow(dense=True, no_gutters=True, classes="mt-1"):
                                    vuetify.VBtn(
                                        "Save",
                                        click="",
                                        small=True,
                                        block=True,
                                        color="primary",
                                        disabled=("!picked_vertices || picked_vertices.length === 0",),
                                        style="font-size: 11px;",
                                    )

        # Upload prompt when no data is loaded
        with vuetify.VContainer(
            v_if=("!data_loaded", True),
            fluid=True,
            classes="d-flex flex-column align-center justify-center fill-height",
            style="margin-top: -50px;",
        ):
            vuetify.VIcon("mdi-folder-upload", classes="mb-4", style="font-size: 64px; color: #999;")
            vuetify.VCardTitle("Upload DICOM Series")
            vuetify.VCardSubtitle("Click to upload a full DICOM series for 3D reconstruction")

            # Ẩn VFileInput nhưng vẫn giữ chức năng
            vuetify.VFileInput(
                v_model=("uploaded_files", None),
                multiple=True,
                hide_input=True,
                accept=".dcm, .dicom, application/dicom, image/*",
                style="display: none",
                id = "hidden_file_input"
            )

            # Button thay thế - có icon và đẹp
            with vuetify.VBtn(
                color="primary",
                elevation=2,
                large=True,
                click="document.getElementById('hidden_file_input').click()",  # Khi click → trigger click vào VFileInput
                classes="mb-4",
            ):
                vuetify.VIcon("mdi-upload", left=True)
                html.Span("Upload DICOM Files")

            vuetify.VDivider(classes="my-4")

            vuetify.VChip(
                "Tip: Use a full series (e.g., CT slices) for best results",
                color="blue-grey",
                dark=True,
                small=True,
            )

#---------------------------------------------------------
# Start server
#---------------------------------------------------------
if __name__ == "__main__":
    server.start()  # type: ignore