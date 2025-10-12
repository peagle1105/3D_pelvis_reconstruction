viewer_2d = None
volume_3d = None
dicom_reader = None
renderer_2d = None

import os
import sys
def resource_path(relative_path):
    """Lấy đường dẫn tuyệt đối đến file khi chạy từ .exe hoặc từ mã nguồn"""
    base_path = getattr(sys, '_MEIPASS', os.path.abspath("."))
    return os.path.join(base_path, relative_path)