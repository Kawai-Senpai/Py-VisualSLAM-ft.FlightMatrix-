# 🛰️ Visual SLAM with FlightMatrix 🚀

```
                           *     .--.
                                / /  `
               +               | |
                      '         \ \__,
                  *          +   '--'  *
                      +   /\
         +              .'  '.   *
                *      /======\      +
                      ;:.  _   ;
                      |:. (_)  |
                      |:.  _   |
            +         |:. (_)  |          *
                      ;:.      ;
                    .' \:.    / `.
                   / .-'':._.'`-. \
                   |/    /||\    \|
             jgs _..--"""````"""--.._
           _.-'``                    ``'-._
         -'                                '-
```

Welcome to the **Visual SLAM** (Simultaneous Localization and Mapping) implementation for the **FlightMatrix** simulation environment! This Python-based system performs real-time 3D pose estimation using **Visual Odometry**, integrated directly with FlightMatrix via the [FlightMatrixBridge API](https://pypi.org/project/flightmatrixbridge/). Whether you're testing with the **FlightMatrix** simulation or the **KITTI dataset**, this project allows you to visualize and analyze camera trajectories, pose estimations, and 3D point cloud data.

---

## 📑 Table of Contents

1. [Features](#features)
2. [Installation](#installation)
3. [Usage](#usage)
4. [Documentation: VisualOdometry Class](#documentation-visualodometry-class)
5. [Key Components](#key-components)
6. [Dataset](#dataset)
7. [References](#references)

---

### ✨ Features

- **ORB Feature Detection & Matching**: Detects and matches key points using the ORB algorithm with FLANN-based matching for fast and accurate results.
- **Real-time 3D Pose Estimation**: Computes essential matrices and decomposes transformations to estimate pose.
- **FlightMatrix Integration**: Seamless integration with FlightMatrix through the FlightMatrixBridge API.
- **Trajectory Visualization**: Visualizes the camera's path and projections for better analysis and debugging.
- **KITTI Dataset Compatibility**: Supports benchmarking against the **KITTI visual odometry dataset**.

---

### ⚙️ Installation

#### 1. Clone the Repository:

```bash
git clone https://github.com/Kawai-Senpai/Py-VisualSLAM_ft.FlightMatrix
cd Py-VisualSLAM_ft.FlightMatrix
```

#### 2. Install Dependencies:

Install Python 3.x dependencies and packages by running:

```bash
pip install -r requirements.txt
```

#### 3. Install FlightMatrixBridge:

To enable communication with **FlightMatrix**, install the required **FlightMatrixBridge** package:

```bash
pip install flightmatrixbridge
```

#### 4. (Optional) Customize Calibration:

If needed, modify paths and settings in `MakeCalibFile.py` to suit your environment.

---

### 🏃‍♂️ Usage

#### 1. Generate Calibration Data:

Generate the required calibration files for the **FlightMatrix** simulator:

```bash
python MakeCalibFile.py
```

#### 2. Run with FlightMatrix:

Execute the following script to run **Visual SLAM** in the **FlightMatrix** simulation:

```bash
python FlightMatrix_Odometry.py
```

#### 3. Run with KITTI Dataset:

For testing with the **KITTI dataset**, run:

```bash
python KITTI_Dataset_Odometry.py
```

#### 4. Logging:

Pose estimations and error analysis are saved in the `FlightMatrixLog.log` file for further inspection.

---

### 📚 Documentation: VisualOdometry Class

The `VisualOdometry` class is the heart of this system, performing monocular visual odometry using **ORB features**, **FLANN-based matching**, and **RANSAC-based essential matrix estimation**. 

Here’s how you can use the `VisualOdometry` class for your SLAM pipeline:

#### Key Class Components

1. **Import Required Modules**:

```python
import numpy as np      # Numerical operations
import cv2              # OpenCV for image processing
import torch            # PyTorch for GPU acceleration
import threading        # For multi-threading support in optimization
```

2. **Initialize the VisualOdometry Class**:

```python
vo = VisualOdometry(
    init_pose = np.eye(4),            # Initial pose (identity matrix)
    camera_calib_file = 'calib.txt',  # Path to the calibration file
    FLANN_INDEX_LSH = 6,              # FLANN index for ORB matching
    max_features = 3000,              # Max features per frame
    bundle_adjustment_epochs = 500,   # Number of epochs for bundle adjustment
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu'), # Use GPU if available
    sharpening = False,               # Toggle image sharpening
    findEssentialMat_method=cv2.RANSAC # RANSAC for essential matrix estimation
)
```

3. **Using the Class**:

- **Update Visual Odometry**:

For each frame, simply use `vo.update(frame)` to update the pose:

```python
for i in range(max_frame):
    # Read the current frame
    frame = cv2.imread(f"{data_dir}/{i:06d}.png")

    # Update visual odometry with the current frame
    vo.update(frame)

    # Retrieve pose data
    estimated_poses = vo.estimated_poses
    img_matches = vo.display_frame
    points = vo.points_3d
    pixels = vo.observations

    # Draw trajectory if poses are available
    if estimated_poses:
        path = [(pose[0, 3], pose[2, 3]) for pose in estimated_poses]
        rotation = estimated_poses[-1][:3, :3]

        # Draw trajectory
        traj_img = draw_trajectory(path, rotation, points, pixels, frame, actual_poses, img_size, draw_scale)
        cv2.imshow("Trajectory", traj_img)

    # Show matches image
    if img_matches is not None:
        cv2.imshow("Matches", img_matches)

    # Exit loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
```

---

### 🛠️ Key Components

- **FlightMatrix_Odometry.py**: The main script to run SLAM in **FlightMatrix** simulation.
- **KITTI_Dataset_Odometry.py**: Handles KITTI dataset sequences for visual odometry testing.
- **MakeCalibFile.py**: Generates calibration files for the camera.
- **Display.py**: Contains functions for visualizing trajectories and poses.

---

### 📂 Dataset

This project is compatible with the [KITTI Visual Odometry Dataset](http://www.cvlibs.net/datasets/kitti/eval_odometry.php). Download the sequences and place them in the `Data/` folder to use with the **KITTI_Dataset_Odometry.py** script.

---

### 📚 References

- [FlightMatrixBridge API](https://pypi.org/project/flightmatrixbridge/) - Python bridge for **FlightMatrix** simulation.
- [OpenCV ORB](https://docs.opencv.org/master/db/d95/classcv_1_1ORB.html) - ORB feature detection and description.
- [FLANN Matcher](https://docs.opencv.org/3.4/dc/de2/classcv_1_1FlannBasedMatcher.html) - FLANN matcher for fast feature matching.
- [KITTI Dataset](http://www.cvlibs.net/datasets/kitti/) - Real-world dataset for visual odometry benchmarking.

---

### 🎉 Thank you for checking out the project! 🚀

Feel free to contribute or explore the code to further enhance **Visual SLAM** for FlightMatrix or other applications. For tutorials, check out our [YouTube Series](https://www.youtube.com/playlist?list=PL9197fpIl1JdTrQcSaCpQAnFXo6Ejx_LT).

