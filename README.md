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

---

### Visual Odometry Class Documentation

This class performs **Visual Odometry** using **ORB** features and **FLANN-based** matching. It includes functionality for **bundle adjustment**, camera calibration loading, and essential matrix computation using RANSAC.

#### **Class Attributes:**

- `bundle_adjustment_learning_rate (float)`: Learning rate for bundle adjustment, influencing convergence speed during optimization.
- `device (torch.device)`: The device on which the computations are performed (CPU or GPU). Automatically set to GPU if available, otherwise defaults to CPU.
- `bundle_adjustment_steps (list)`: A list of steps for bundle adjustment to control the number of iterations or refinement processes.
- `bundle_adjustment_epochs (int)`: The number of epochs for bundle adjustment.
- `bundle_adjustment_loss_tolerance (float)`: Tolerance for early stopping in bundle adjustment based on loss convergence.
- `bundle_adjustment_threads (dict)`: A dictionary storing bundle adjustment threads, used to parallelize adjustments.
- `lock (threading.Lock)`: A threading lock for synchronization during bundle adjustment, preventing race conditions.
- `estimated_poses (list)`: A list of estimated camera poses across frames, representing the camera trajectory.
- `points_3d (list)`: A list of 3D world points reconstructed from the 2D features.
- `observations (list)`: A list of 2D observations corresponding to keypoints detected in frames.
- `K (numpy.ndarray)`: Camera intrinsic matrix, used to convert between pixel and world coordinates.
- `P (numpy.ndarray)`: Projection matrix used for mapping 3D world points to 2D image coordinates.
- `orb (cv2.ORB)`: ORB feature detector and descriptor used to identify and describe keypoints in each frame.
- `flann (cv2.FlannBasedMatcher)`: FLANN-based matcher used for matching keypoints between consecutive frames.
- `ratio_test_threshold (float)`: Threshold for ratio test during FLANN matching, used to filter out unreliable matches.
- `knn_match_num (int)`: The number of nearest neighbors to find during FLANN matching.
- `prev_img (numpy.ndarray)`: The previous image frame for tracking keypoints and computing relative motion.
- `prev_keypoints (list)`: List of keypoints in the previous frame.
- `prev_descriptors (numpy.ndarray)`: Descriptors of the keypoints in the previous frame.
- `display_frame (numpy.ndarray)`: The current frame for displaying keypoints, useful for visualization.

#### **Additional Attributes:**

- `image_sharpen_kernel (numpy.ndarray)`: Kernel used for sharpening the image, which enhances the clarity of features in the frame.
- `sharpening (bool)`: Flag to enable or disable image sharpening before feature detection.
- `findEssentialMat_method (int)`: The method for finding the essential matrix. Common options include `cv2.RANSAC` or `cv2.LMEDS`.
- `findEssentialMat_prob (float)`: Probability parameter for RANSAC in finding the essential matrix.
- `findEssentialMat_threshold (float)`: Threshold parameter for RANSAC, controlling the maximum reprojection error allowed for a match to be considered in the model.

---

#### **Initialization:**

```python
vo = VisualOdometry(
    init_pose=np.eye(4),                        # Initial pose (identity matrix for the first frame)
    camera_calib_file='calib.txt',               # Path to the camera calibration file
    FLANN_INDEX_LSH=6,                           # FLANN index for ORB feature matching
    table_number=6,                              # FLANN table number for ORB feature matching
    key_size=12,                                 # FLANN key size for ORB feature matching
    multi_probe_level=1,                         # Multi-probe level for FLANN matching
    ratio_test_threshold=0.75,                   # Ratio test threshold for feature matching
    knn_match_num=2,                             # Number of nearest neighbors to find in FLANN matching
    max_features=3000,                           # Maximum number of features per frame
    bundle_adjustment_steps=[2, 10],             # Bundle adjustment steps
    bundle_adjustment_epochs=500,                # Number of epochs for bundle adjustment
    bundle_adjustment_learning_rate=1e-3,       # Learning rate for bundle adjustment
    bundle_adjustment_loss_tolerance=1,         # Loss tolerance for early stopping in bundle adjustment
    device=torch.device('cuda' if torch.cuda.is_available() else 'cpu'), # Use GPU if available
    image_sharpen_kernel=np.array([              # Kernel for image sharpening
        [0, -1, 0], 
        [-1, 5, -1], 
        [0, -1, 0]
    ]),
    sharpening=False,                            # Flag to enable/disable image sharpening
    findEssentialMat_method=cv2.RANSAC,          # Method to estimate essential matrix (RANSAC)
    findEssentialMat_prob=0.999,                 # Probability for RANSAC estimation
    findEssentialMat_threshold=1.0               # Threshold for RANSAC essential matrix estimation
)
```

#### **Parameters:**

- `init_pose (numpy.ndarray)`: The initial pose (4x4 matrix) of the camera. Typically, this is set to the identity matrix for the first frame.
- `camera_calib_file (str)`: The path to the camera calibration file containing intrinsic parameters like focal length and principal point.
- `FLANN_INDEX_LSH (int)`: The FLANN index used for ORB matching, typically set to 6 for ORB descriptors.
- `max_features (int)`: The maximum number of features to be detected in each frame. A larger number means more keypoints, but at the cost of computation time.
- `bundle_adjustment_epochs (int)`: The number of epochs to run for bundle adjustment. This is the number of iterations the optimizer will try to improve the pose estimation.
- `device (torch.device)`: The device (CPU or GPU) for tensor computations. If a GPU is available, it will be used for faster computation.
- `sharpening (bool)`: If `True`, the image will be sharpened before feature extraction, enhancing feature detection.
- `findEssentialMat_method (int)`: The method used to compute the essential matrix. Default is RANSAC, which is robust to outliers.
- `findEssentialMat_prob (float)`: Probability parameter for RANSAC in essential matrix estimation.
- `findEssentialMat_threshold (float)`: Threshold for RANSAC in determining inliers for the essential matrix.

---

#### **Methods:**

- **_load_calib(filepath)**: Loads camera calibration parameters (intrinsic and extrinsic) from a file.
  
    ```python
    def _load_calib(self, filepath: str):
        # Load calibration data from a file
        pass
    ```

- **_form_transf(R, t)**: Forms a transformation matrix from a given rotation matrix `R` and translation vector `t`.
  
    ```python
    def _form_transf(self, R: np.ndarray, t: np.ndarray) -> np.ndarray:
        # Convert rotation matrix and translation vector into a transformation matrix
        pass
    ```

- **get_matches(img)**: Detects ORB keypoints and descriptors in the given image, and matches them with the previous frame using FLANN.
  
    ```python
    def get_matches(self, img: np.ndarray):
        # Detect ORB keypoints and match with previous frame
        pass
    ```

- **update_pose(q1, q2)**: Updates the camera pose based on the matches between the current and previous frames.
  
    ```python
    def update_pose(self, q1: np.ndarray, q2: np.ndarray):
        # Update the camera pose based on feature matches
        pass
    ```

- **bundle_adjustment()**: Performs bundle adjustment to optimize the camera poses and 3D points. Uses multi-threading and loss tolerance for convergence.
  
    ```python
    def bundle_adjustment(self):
        # Refine poses and points using bundle adjustment
        pass
    ```

---

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

