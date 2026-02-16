# Thermal Image Animal Segmentation and Boundary Extraction

This project performs **animal segmentation** from thermal images using both a **classical HSV-based computer vision pipeline** and the **Segment Anything Model 2 (SAM2)**. The goal is to extract the object boundary and compare classical and deep learning approaches visually and quantitatively using IoU and Boundary F1 scores.

---

## Repository Structure

Computer Vision Module 4/
│
├─ sbhattarai9_cv_4.py # Main Python script for segmentation
├─ thermal1.jpg # Test thermal image 1
├─ thermal3.jpg # Test thermal image 2
├─ sam2/ # SAM2 cloned repository
│ ├─ sam2/
│ │ └─ configs/...
│ └─ checkpoints/...
├─ CV-Venv/ # Optional: Python virtual environment (ignored in git)
└─ README.md # This file


---

## Installation

1. **Clone this repository**

```powershell
git clone https://github.com/sukibhattarai2026-arch/Computer_vision_module4.git
cd "Computer Vision Module 4"


python -m venv CV-Venv
.\CV-Venv\Scripts\activate


pip install -r requirements.txt
pip install opencv-python numpy torch torchvision
pip install opencv-python numpy torch torchvision
git clone https://github.com/facebookresearch/sam2.git


Running the Script

python "D:\Assignment\Computer Vision\Computer Vision Module 4\sbhattarai9_cv_4.py" `
--run_sam2 `
--sam2_cfg "D:\Assignment\Computer Vision\Computer Vision Module 4\sam2\sam2\configs\sam2.1\sam2.1_hiera_l.yaml" `
--sam2_checkpoint "D:\Assignment\Computer Vision\Computer Vision Module 4\sam2\checkpoints\sam2.1_hiera_large.pt" `
--device cpu `
--images thermal1.jpg thermal3.jpg

