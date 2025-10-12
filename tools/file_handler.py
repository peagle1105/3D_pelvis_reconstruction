import base64
from pathlib import Path, PurePath
import zipfile
from io import BytesIO
import vtk
import os
import tempfile
from config.config import resource_path

temp_path = resource_path("temp_folder/")

# Sub-function
def get_ply_content(mesh_data):
    try:
        # Sử dụng vtkPLYWriter với StringIO
        ply_writer = vtk.vtkPLYWriter()
        ply_writer.SetFileTypeToASCII()
        ply_writer.SetInputData(mesh_data)
        
        with tempfile.NamedTemporaryFile(suffix='.ply', delete=False, mode='w') as temp_file:
            temp_filename = temp_file.name
        
        ply_writer.SetFileName(temp_filename)
        ply_writer.Write()
        
        # Đọc nội dung file
        with open(temp_filename, 'r') as f:
            content = f.read()
        
        # Xóa file tạm
        os.unlink(temp_filename)
        return content
    except Exception as e:
        print(f"Error: {e}")
        return None

def get_obj_content(mesh_data):
    try:
        # Sử dụng vtkOBJWriter
        obj_writer = vtk.vtkOBJWriter()
        obj_writer.SetInputData(mesh_data)
        
        with tempfile.NamedTemporaryFile(suffix='.obj', delete=False, mode='w') as temp_file:
            temp_filename = temp_file.name
        
        obj_writer.SetFileName(temp_filename)
        obj_writer.Write()
        
        # Đọc nội dung file
        with open(temp_filename, 'r') as f:
            content = f.read()
        
        # Xóa file tạm
        os.unlink(temp_filename)
        return content
    except Exception as e:
        print(f"Error: {e}")
        return None

def get_stl_content(mesh_data, ascii_format=True):
    try:
        # Sử dụng vtkSTLWriter
        stl_writer = vtk.vtkSTLWriter()
        stl_writer.SetInputData(mesh_data)
        
        # Chọn định dạng ASCII hoặc Binary
        if ascii_format:
            stl_writer.SetFileTypeToASCII()
        else:
            stl_writer.SetFileTypeToBinary()

        with tempfile.NamedTemporaryFile(suffix='.stl', delete=False, mode='w') as temp_file:
            temp_filename = temp_file.name
        
        stl_writer.SetFileName(temp_filename)
        stl_writer.Write()
        
        # Đọc nội dung file
        mode = 'r' if ascii_format else 'rb'
        with open(temp_filename, mode) as f:
            content = f.read()
        
        # Xóa file tạm
        os.unlink(temp_filename)
        return content
    except Exception as e:
        print(f"Error: {e}")
        return None

# Main function
def load_files(temp_path, file_list):
    """Save uploaded files to temporary directory"""

    for file in file_list:
        original_name = file["name"]
        clean_name = PurePath(original_name).name
        if not clean_name or clean_name == ".":
            clean_name = "unknown.dcm"
        if "." not in clean_name:
            clean_name += ".dcm"

        file_path = temp_path / clean_name
        content = file["content"]
        
        if isinstance(content, str):
            # Handle base64 data URI
            if "," in content:
                _, base64_part = content.split(",", 1)
            else:
                base64_part = content
            try:
                binary_data = base64.b64decode(base64_part)
            except Exception as e:
                print(f"❌ Base64 decode failed: {e}")
                continue
        elif isinstance(content, bytes):
            binary_data = content
        else:
            print(f"❌ Unsupported content type: {type(content)}")
            continue

        try:
            with open(file_path, "wb") as f:
                f.write(binary_data)
            # Optional DICOM validation
            try:
                import pydicom
                pydicom.dcmread(file_path, stop_before_pixels=True)
            except ImportError:
                pass  # Pydicom not available
            except Exception as e:
                print(f"⚠️ DICOM validation warning: {e}")
        except Exception as e:
            print(f"❌ Failed to save file: {e}")

def load_zip_file(temp_path, zip_file):
    """
    Giải nén file ZIP (dưới dạng base64 hoặc bytes) vào thư mục temp_path.
    zip_file: dict như {'name': 'data.zip', 'content': '...base64...'}
    """
    temp_path = Path(temp_path)
    temp_path.mkdir(parents=True, exist_ok=True)
    content = zip_file["content"]

    # Xử lý base64 nếu cần
    if isinstance(content, str):
        if "," in content:
            _, base64_part = content.split(",", 1)
        else:
            base64_part = content
        try:
            binary_data = base64.b64decode(base64_part)
        except Exception as e:
            print(f"❌ Base64 decode failed: {e}")
            return False
    elif isinstance(content, bytes):
        binary_data = content
    else:
        print(f"❌ Unsupported content type: {type(content)}")
        return False

    try:
        # Mở file ZIP từ dữ liệu nhị phân
        with zipfile.ZipFile(BytesIO(binary_data), 'r') as zip_ref:
            for member in zip_ref.infolist():
                # Bỏ qua thư mục
                if member.is_dir():
                    continue

                # Lấy tên file gốc trong ZIP
                original_name = member.filename
                clean_name = Path(original_name).name

                # Đảm bảo tên file hợp lệ
                if not clean_name or clean_name == ".":
                    clean_name = "unknown.dcm"
                if "." not in clean_name:
                    clean_name += ".dcm"

                # Đường dẫn lưu file
                file_path = temp_path / clean_name

                # Ghi file ra đĩa
                with open(file_path, "wb") as f:
                    f.write(zip_ref.read(member))

                # (Tùy chọn) Kiểm tra file DICOM
                try:
                    import pydicom
                    pydicom.dcmread(file_path, stop_before_pixels=True)
                except ImportError:
                    pass  # Không có pydicom → bỏ qua
                except Exception as e:
                    print(f"⚠️ DICOM validation warning for {clean_name}: {e}")

        print(f"✅ Giải nén thành công {len(zip_ref.filelist)} file vào {temp_path}")
        return True

    except zipfile.BadZipFile:
        print("❌ File không phải định dạng ZIP hợp lệ")
        return False
    except Exception as e:
        print(f"❌ Lỗi khi giải nén: {e}")
        return False

def export_mesh(state, mesh_data):
    match state.file_mesh_extend:
        case 'ply':
            content = get_ply_content(mesh_data= mesh_data)
        case 'obj':
            content = get_obj_content(mesh_data= mesh_data)
        case 'stl':
            content = get_stl_content(mesh_data= mesh_data)
        case _:
            content = None
    return content