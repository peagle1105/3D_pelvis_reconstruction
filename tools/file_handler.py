import base64
from pathlib import Path, PurePath
import zipfile
from io import BytesIO
import vtk
import os
import tempfile

temp_path = "./temp_folder/"

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

def export_mesh(ctrl, mesh, file_name, extend):
    """
    Export mesh and trigger browser download — NO physical file saved.
    
    Parameters:
    - mesh: vtk.vtkPolyData
    - file_name: str, base name (without extension), e.g., "my_mesh"
    - extend: str, one of "PLY", "OBJ", "STL"
    """
    if not hasattr(mesh, "GetNumberOfPoints"):
        raise TypeError("Input mesh must be a vtkPolyData object.")

    # Tạo đường dẫn tạm (chỉ để VTK ghi vào — sẽ đọc lại và xóa ngay)
    suffix = f".{extend.lower()}"
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
        tmp_path = tmp.name

    try:
        # Chọn writer theo định dạng
        match extend.upper():
            case "PLY":
                writer = vtk.vtkPLYWriter()
                writer.SetFileName(tmp_path)
                writer.SetInputData(mesh)
                writer.SetFileTypeToBinary()  # hoặc SetFileTypeToASCII()
                mime_type = "application/octet-stream"

            case "OBJ":
                writer = vtk.vtkOBJWriter()
                writer.SetFileName(tmp_path)
                writer.SetInputData(mesh)
                # OBJ không hỗ trợ binary trong VTK → luôn là ASCII
                mime_type = "text/plain"

            case "STL":
                writer = vtk.vtkSTLWriter()
                writer.SetFileName(tmp_path)
                writer.SetInputData(mesh)
                writer.SetFileTypeToBinary()
                mime_type = "application/octet-stream"

            case _:
                raise ValueError(f"Unsupported format: {extend}. Use 'PLY', 'OBJ', or 'STL'.")

        # Ghi file tạm
        writer.Write()

        # Đọc nội dung file
        with open(tmp_path, "rb") as f:
            content = f.read()

        # Trigger download trong Trame
        full_file_name = f"{file_name}{suffix}"
        ctrl.download(content, full_file_name, mime_type)
        print(f"✅ Download triggered: {full_file_name}")

    except Exception as e:
        print(f"❌ Export failed: {e}")
        raise e

    finally:
        # Dọn file tạm
        if os.path.exists(tmp_path):
            os.unlink(tmp_path)