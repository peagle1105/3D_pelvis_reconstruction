# 🦴 DICOM 3D Viewer

## 📌 Overview

**DICOM 3D Viewer** is a medical imaging system designed to reconstruct a 3D mesh of the female pelvis from anatomical landmarks. It provides an intuitive interface for uploading DICOM data and selecting key points for mesh generation.

### ✨ Features

- Upload individual DICOM files or `.zip` archives containing pelvic scan data
- Select anatomical landmarks manually via the interface or upload a `.pp` point file
- Click **Reconstruct** to generate a complete 3D mesh of the pelvis

---

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


_How to use_

- Run the project
- Upload DiCOM files or a zip file contain DICOM files
- There are some simple functions:
  \*Pick feature points:
  Edit > pick points > Left click to check the position > Right click to pick
  *Choose the orientation of DICOM files
  View > Axial/Sagittal/Coronal
  *Upload another set of DICOM files
  File > new series
```
