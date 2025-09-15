import base64
from pathlib import Path, PurePath

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
