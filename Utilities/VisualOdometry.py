import numpy as np
import cv2
import torch
import threading 

class VisualOdometry():

    def __init__(self, init_pose = None, 
                camera_calib_file = 'calib.txt',
                FLANN_INDEX_LSH = 6,
                table_number = 6,
                key_size = 12,
                multi_probe_level = 1,
                good_filter = 0.5,
                k = 2,
                bundle_adjustment_steps = [2, 10],
                bundle_adjustment_epochs = 500,
                bundle_adjustment_learning_rate = 1e-3,
                bundle_adjustment_loss_tolerance = 1,
                device = torch.device('cuda' if torch.cuda.is_available() else 'cpu'),
                image_sharpen_kernal = np.array([
                [0, -1, 0], 
                [-1, 5, -1], 
                [0, -1, 0]]),
                sharpening = False
                ):

        self.bundle_adjustment_learning_rate = bundle_adjustment_learning_rate
        self.device = device
        self.bundle_adjustment_steps = bundle_adjustment_steps
        self.bundle_adjustment_epochs = bundle_adjustment_epochs
        self.bundle_adjustment_loss_tolerance = bundle_adjustment_loss_tolerance

        self.bundle_adjustment_threads = {}
        self.lock = threading.Lock()
        
        if init_pose is not None and init_pose.shape == (4, 4):
            self.estimated_poses = [init_pose]
        else:
            self.estimated_poses = []

        self.points_3d = []
        self.observations = []

        self.K, self.P = self._load_calib(camera_calib_file)

        self.orb = cv2.ORB_create(3000)
        index_params = dict(algorithm=FLANN_INDEX_LSH, table_number=table_number, key_size=key_size, multi_probe_level=multi_probe_level)
        search_params = dict(checks=50)
        self.flann = cv2.FlannBasedMatcher(indexParams=index_params, searchParams=search_params)

        self.good_filter = good_filter
        self.k = k

        self.prev_img = None
        self.prev_keypoints = None
        self.prev_descriptors = None

        self.display_frame = None
        self.image_sharpen_kernal = image_sharpen_kernal
        self.sharpening = sharpening

    #! Helper Functions -----------------------------------------------------
    def _load_calib(self, filepath):

        with open(filepath, 'r') as f:
            params = np.fromstring(f.readline(), dtype=np.float64, sep=' ')
            P = np.reshape(params, (3, 4))
            K = P[0:3, 0:3]
        return K, P

    def _form_transf(self, R, t):

        T = np.eye(4, dtype=np.float64)
        T[:3, :3] = R
        T[:3, 3] = t
        return T

    #! Main ORB Functions ---------------------------------------------------
    def get_matches(self, img):
        try:
            # Convert the input image to grayscale
            img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  # Shape: (H, W)
            
            # Apply sharpening filter if enabled
            if self.sharpening:
                img_gray = cv2.filter2D(img_gray, -1, self.image_sharpen_kernal)  # Shape: (H, W)
            
            # Detect ORB keypoints and compute descriptors
            keypoints, descriptors = self.orb.detectAndCompute(img_gray, None)  # keypoints: list of cv2.KeyPoint, descriptors: (N, 32)
            
            # If there are no previous descriptors or current descriptors, update previous frame data and return
            if self.prev_descriptors is None or descriptors is None:
                self.prev_img = img_gray  # Shape: (H, W)
                self.prev_keypoints = keypoints  # list of cv2.KeyPoint
                self.prev_descriptors = descriptors  # Shape: (N, 32)
                return None, None
            
            # If there are not enough descriptors, return
            if len(descriptors) < self.k or len(self.prev_descriptors) < self.k:
                print("Not enough descriptors")
                return None, None
            
            # Find matches between previous and current descriptors using FLANN-based matcher
            matches = self.flann.knnMatch(self.prev_descriptors, descriptors, k=self.k)  # matches: list of DMatch
            
            # Filter good matches based on distance ratio test
            good = [m for m, n in matches if m.distance < self.good_filter * n.distance]  # good: list of DMatch
            
            # If there are not enough good matches, return
            if len(good) < 8:
                print("Not enough good matches")
                return None, None
            
            # Draw keypoints on the current frame for display
            self.display_frame = cv2.drawKeypoints(img_gray, keypoints, None, color=(0, 255, 0))  # Shape: (H, W, 3)
            
            # Extract matched keypoints' coordinates
            q1 = np.float32([self.prev_keypoints[m.queryIdx].pt for m in good])  # Shape: (M, 2)
            q2 = np.float32([keypoints[m.trainIdx].pt for m in good])  # Shape: (M, 2)
            
            # Update previous frame data with current frame data
            self.prev_img = img_gray  # Shape: (H, W)
            self.prev_keypoints = keypoints  # list of cv2.KeyPoint
            self.prev_descriptors = descriptors  # Shape: (N, 32)
            
            # Return matched keypoints' coordinates
            return q1, q2  # Shape: (M, 2), (M, 2)
        except Exception as e:
            # Print any exception that occurs and return None
            print(e)
            return None, None

    def update_pose(self, q1, q2):
        # Compute the Essential Matrix using the matched points
        Essential, mask = cv2.findEssentialMat(q1, q2, self.K)  # Essential: (3, 3), mask: (M, 1)

        # Decompose the Essential Matrix to obtain R, t, and 3D points
        R, t, hom_Q, valid_indices = self.decomp_essential_mat(Essential, q1, q2)  # R: (3, 3), t: (3,), hom_Q: (4, N), valid_indices: (N,)

        if R is None or t is None or hom_Q is None:
            print("Failed to decompose Essential Matrix")
            return

        # Form the transformation matrix from R and t
        transf = self._form_transf(R, t)  # Shape: (4, 4)

        if transf is None:
            print("Failed to form transformation matrix")
            return

        # Update the pose with respect to the world frame
        if not self.estimated_poses:
            # If it's the first pose, invert the transformation
            self.estimated_poses.append(np.linalg.inv(transf))  # Shape: (4, 4)
        else:
            # Update the current pose based on the previous pose
            self.estimated_poses.append(self.estimated_poses[-1] @ np.linalg.inv(transf))  # Shape: (4, 4)

        # Display the estimated pose
        print(f"Estimated Pose --> x: {self.estimated_poses[-1][0, 3]}, y: {self.estimated_poses[-1][1, 3]}, z: {self.estimated_poses[-1][2, 3]}")

        # Convert the homogeneous 3D points to world coordinates
        hom_Q_world = self.estimated_poses[-1] @ hom_Q  # Shape: (4, N)
        Q = hom_Q_world[:3, :] / hom_Q_world[3, :]  # Shape: (3, N)

        # Filter out points with negative depth
        Q = Q[:, valid_indices]  # Shape: (3, M)
        q2 = q2[valid_indices]  # Shape: (M, 2)

        # Append the 3D points and observations for bundle adjustment
        self.points_3d.append(Q.T)  # Shape: (M, 3)
        self.observations.append(q2)  # Shape: (M, 2)

        # Start the bundle adjustment threads if not already running
        for step in self.bundle_adjustment_steps:
            if step not in self.bundle_adjustment_threads or not self.bundle_adjustment_threads[step].is_alive():
                thread = threading.Thread(target=self.bundle_adjustment, args=(step, self.bundle_adjustment_epochs))
                thread.start()
                self.bundle_adjustment_threads[step] = thread

    def decomp_essential_mat(self, E, q1, q2):
        try:
            # Decompose the Essential Matrix into possible rotations and translation
            R1, R2, t = cv2.decomposeEssentialMat(E)  # R1: (3, 3), R2: (3, 3), t: (3, 1)

            # Form the four possible transformation matrices
            transformations = [
                self._form_transf(R1, t.squeeze()),    # Shape: (4, 4)
                self._form_transf(R2, t.squeeze()),    # Shape: (4, 4)
                self._form_transf(R1, -t.squeeze()),   # Shape: (4, 4)
                self._form_transf(R2, -t.squeeze())    # Shape: (4, 4)
            ]

            # Extend the intrinsic matrix K to homogeneous coordinates
            K_hom = np.hstack((self.K, np.zeros((3, 1))))  # Shape: (3, 4)
            
            # Calculate projection matrices for each transformation
            projections = [K_hom @ T for T in transformations]  # List of (3, 4)

            # Initialize lists to store positive depth counts and valid indices
            positives = []
            valid_indices_list = []
            hom_Q12_list = []

            for P, T in zip(projections, transformations):
                # Triangulate points between previous and current frames
                hom_Q1 = cv2.triangulatePoints(self.P, P, q1.T, q2.T)  # Shape: (4, N)
                
                # Transform the points to the current frame
                hom_Q2 = T @ hom_Q1  # Shape: (4, N)

                # Convert homogeneous coordinates to 3D points
                Q1 = hom_Q1[:3] / (hom_Q1[3] + 1e-6)  # Shape: (3, N)
                Q2 = hom_Q2[:3] / (hom_Q2[3] + 1e-6)  # Shape: (3, N)

                # Check for points with positive depth in both frames
                valid = (Q1[2] > 0) & (Q2[2] > 0)  # Shape: (N,)
                score = valid.sum()  # Scalar

                # Store results
                positives.append(score)
                valid_indices_list.append(valid)
                hom_Q12_list.append(hom_Q2)

            # Select the transformation with the most points having positive depth
            max_idx = np.argmax(positives)
            if positives[max_idx] == 0:
                # No valid transformation found
                return None, None, None, None

            # Extract the correct rotation and translation
            R = transformations[max_idx][:3, :3]  # Shape: (3, 3)
            t = transformations[max_idx][:3, 3]  # Shape: (3,)
            hom_Q = hom_Q12_list[max_idx]  # Shape: (4, N)
            valid_indices = valid_indices_list[max_idx]  # Shape: (N,)
            return R, t, hom_Q, valid_indices

        except Exception as e:
            print("Error in decomp_essential_mat:", e)
            return None, None, None, None

    #! Main Update Function ------------------------------------------------
    def update(self, img):
        q1, q2 = self.get_matches(img)  # q1: (M, 2), q2: (M, 2)
        if q1 is not None and q2 is not None:
            self.update_pose(q1, q2)

    def shutdown(self):
        for thread in self.bundle_adjustment_threads.values():
            thread.join()

    #! Bundle Adjustment ----------------------------------------------------
    #* Using PyTorch ----------------------------------------------------------
    def project(self, points_3d, pose, K):
        """
        Projects 3D points into the image plane using the given pose and camera intrinsics.

        Parameters:
            points_3d (torch.Tensor): Tensor of shape (3, N) representing 3D points in world coordinates.
            pose (torch.Tensor): Tensor of shape (4, 4) representing the camera pose.
            K (torch.Tensor): Tensor of shape (3, 3) representing the camera intrinsic matrix.

        Returns:
            torch.Tensor: Tensor of shape (N, 2) containing the projected 2D points in pixel coordinates.
        """
        # Convert 3D points to homogeneous coordinates by adding a row of ones
        ones = torch.ones((1, points_3d.shape[1]), device=self.device)  # Shape: (1, N)
        points_3d_h = torch.vstack((points_3d, ones))  # Shape: (4, N)

        # Transform the points to the camera coordinate frame
        # Since pose is the camera pose in world coordinates, we invert it to transform points to camera frame
        cam_points_h = torch.linalg.inv(pose) @ points_3d_h  # Shape: (4, N)

        # Project the 3D points onto the 2D image plane
        img_points_h = K @ cam_points_h[:3, :]  # Shape: (3, N)

        # Convert homogeneous coordinates to 2D pixel coordinates
        img_points = (img_points_h[:2, :] / img_points_h[2, :]).T  # Shape: (N, 2)

        return img_points

    def optimisable_func(self, estimated_poses, points_3d, K, observations):
        """
        Computes the total reprojection error for bundle adjustment.

        Parameters:
            estimated_poses (list): List of torch.Tensors representing camera poses.
            points_3d (list): List of torch.Tensors representing 3D points.
            K (torch.Tensor): Camera intrinsic matrix.
            observations (list): List of torch.Tensors representing observed 2D points.

        Returns:
            torch.Tensor: Scalar tensor representing the total reprojection error.
        """
        reprojection_error = torch.tensor(0.0, device=self.device)  # Scalar

        # Loop over all frames to compute the error
        for i in range(len(observations)):
            # Project the 3D points into the image plane
            projected_points = self.project(points_3d[i].T, estimated_poses[i], K)  # Shape: (N, 2)

            # Compute the squared differences between observed and projected points
            error = (projected_points - observations[i]) ** 2  # Shape: (N, 2)

            # Sum the errors for all points in this frame
            reprojection_error += error.sum()  # Scalar

        return reprojection_error

    def bundle_adjustment(self, steps, iterations):
        """
        Performs bundle adjustment to refine camera poses and 3D points.

        Parameters:
            steps (int): Number of recent frames to include in the optimization.
            iterations (int): Number of optimization iterations.
        """
        if len(self.estimated_poses) >= steps and len(self.points_3d) >= steps and len(self.observations) >= steps:
            with self.lock:
                total_frames = len(self.estimated_poses)
                start_idx = total_frames - steps

                # Prepare variables for optimization
                poses = [torch.tensor(self.estimated_poses[i], dtype=torch.float32, device=self.device, requires_grad=True)
                        for i in range(start_idx, total_frames)]  # List of (4, 4)
                points = [torch.tensor(self.points_3d[i], dtype=torch.float32, device=self.device, requires_grad=True)
                        for i in range(start_idx, total_frames)]  # List of (M, 3)
                observations = [torch.tensor(self.observations[i], dtype=torch.float32, device=self.device)
                                for i in range(start_idx, total_frames)]  # List of (M, 2)

            K = torch.tensor(self.K, dtype=torch.float32, device=self.device)  # Shape: (3, 3)

            # Use Adam optimizer for joint optimization of poses and points
            optimizer = torch.optim.Adam(poses + points, lr=self.bundle_adjustment_learning_rate)

            for step in range(iterations):
                optimizer.zero_grad()

                # Compute the total reprojection error
                loss = self.optimisable_func(poses, points, K, observations)  # Scalar

                # Backpropagate the error
                loss.backward()
                optimizer.step()

                # Optional: Print progress every 100 iterations
                if (step + 1) % 100 == 0 or step == 0:
                    print(f'Bundle Adjustment Iteration {step + 1}/{iterations}, Loss: {loss.item()}')

                # Early stopping if loss is below threshold
                if loss.item() < self.bundle_adjustment_loss_tolerance:
                    print(f'Converged at iteration {step + 1}, Loss: {loss.item()}')
                    break

            # Update the estimated poses and 3D points with optimized values
            with self.lock:
                for i, idx in enumerate(range(start_idx, total_frames)):
                    self.estimated_poses[idx] = poses[i].detach().cpu().numpy()  # Shape: (4, 4)
                    self.points_3d[idx] = points[i].detach().cpu().numpy()  # Shape: (M, 3)
