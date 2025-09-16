# 🦴 DICOM 3D Viewer

## 📌 Overview

**DICOM 3D Viewer** is a medical imaging system designed to reconstruct a 3D mesh of the female pelvis from anatomical landmarks. It provides an intuitive interface for uploading DICOM data and selecting key points for mesh generation.

## 🚀 Getting Started

Follow these steps to run the project locally:

```bash
# 1. Clone the repository
git clone https://github.com/peagle1105/3D_pelvis_reconstruction.git
cd dicom-3d-viewer

# 2. Create and activate a virtual environment
python -m venv venv

# On macOS/Linux:
source venv/bin/activate

# On Windows:
venv\Scripts\activate

# 3. Install required dependencies
pip install -r requirements.txt

# 4. Launch the web interface
python ui.py
```

## 📖 Usage Guide

**Once the web interface is running:**

- Upload Data

- Click the Upload button upload either:
  All of .dcm files
  A .zip archive containing DICOM data

- Add Landmark Points:
  Manually select key anatomical points using the Edit > pick point
  Or upload a .pp file containing pre-defined points by clicking "Load" at pick point windows

- Reconstruct Mesh
  Click the Reconstruct button

==> The system will generate and display a 3D mesh of the pelvis

- Interact with the Mesh:
  Rotate, zoom, and pan the 3D model using mouse controls

- Export mesh: file > export

## 🛠 System Requirements

- Python 3.8 or higher
- Modern web browser (Chrome, Firefox, Edge)
- Minimum 8GB RAM recommended for large DICOM datasets
