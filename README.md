# 📌 Project Title

> ### **Fall Detection System**



<br><br>
## 📖 Overview

This system is designed to rapidly detect falls, especially among the elderly population, and immediately notify caregivers to enable prompt response and intervention.  
Unlike typical vision-based approaches, it detects falls **without any wearable sensors**, using only a camera module.
The system extracts skeleton keypoints and treats them as time-series data—similar to gyroscope readings—enabling fall detection through temporal analysis rather than standard image classification.  
A functional prototype was also implemented and deployed on a Raspberry Pi for real-world demonstration.


<br><br>
## 🧩 Data Preprocessing
- **Dataset**: 3,782 video samples
- **Challenge**: Detecting falls using **only computer vision** (no physical sensors) was difficult
- **Inspiration**: Adopted the concept of **gyroscope data** from wearable devices to capture motion patterns
- **Skeleton Extraction**:
  - Used `MediaPipe Pose` to estimate human body keypoints
  - Each joint represented by its **X, Y, Z coordinates** in 3D space
- Skeleton data served as a **sensor-like input** for the model


<br><br>
## 🛠️ Tech Stack

| Category        | Tools / Frameworks                |
|----------------|-----------------------------------|
| OS              | Windows 11 HOME, Raspberry PI OS |
| Language        | Python 3.8                       |
| Libraries       | torch, sklearn, mediapipe, picamera2, opencv |
| Environment     | Jupyter Notebook / VSCode |
| Hardware        | GPU, Raspberry PI                |


<br><br>
## 📂 Project Structure

```bash
.
├── Final_model/                    # TorchScript model 
├── Raspberry_Pi_Test/              # Raspberry PI translation Prototype
│ 
│   # Preprocessing
├── Data_Preprocessing.ipynb        # Data Preprocessing
├── Extract_skeleton.py             # Extract skeleton landmarks from video frames
├── Labeling.ipynb                  # Labeling for fall detection
├── Merging_Data.py                 # Merge multiple datasets
│ 
│   # Model Training
├── Fall_Detection.ipynb            # Notebook for training and evaluation
└── [...]
```

<br><br>
## 💡Install Dependency

```bash
pip install -r requirements.txt
```
