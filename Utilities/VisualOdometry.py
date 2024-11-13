import numpy as np
import cv2
import torch
import threading 

class VisualOdometry():
    
    """
    A class to perform Visual Odometry using ORB features and FLANN-based matching.
    Attributes:
        bundle_adjustment_learning_rate (float): Learning rate for bundle adjustment.
        device (torch.device): Device to run computations on (CPU or GPU).
        bundle_adjustment_steps (list): Steps for bundle adjustment.
        bundle_adjustment_epochs (int): Number of epochs for bundle adjustment.
        bundle_adjustment_loss_tolerance (float): Loss tolerance for early stopping in bundle adjustment.
        bundle_adjustment_threads (dict): Dictionary to store bundle adjustment threads.
        lock (threading.Lock): Lock for thread synchronization.
        estimated_poses (list): List of estimated poses.
        points_3d (list): List of 3D points.
        observations (list): List of 2D observations.
        K (numpy.ndarray): Camera intrinsic matrix.
        P (numpy.ndarray): Projection matrix.
        orb (cv2.ORB): ORB feature detector and descriptor.
        flann (cv2.FlannBasedMatcher): FLANN-based matcher.
        ratio_test_threshold (float): Threshold for ratio test in FLANN matching.
        knn_match_num (int): Number of nearest neighbors to find in FLANN matching.
        prev_img (numpy.ndarray): Previous image frame.
        prev_keypoints (list): List of keypoints in the previous frame.
        prev_descriptors (numpy.ndarray): Descriptors of keypoints in the previous frame.
        display_frame (numpy.ndarray): Frame for displaying keypoints.
        image_sharpen_kernel (numpy.ndarray): Kernel for image sharpening.
        sharpening (bool): Flag to enable/disable image sharpening.
        findEssentialMat_method (int): Method for finding the Essential Matrix.
        findEssentialMat_prob (float): Probability for RANSAC in finding the Essential Matrix.
        findEssentialMat_threshold (float): Threshold for RANSAC in finding the Essential Matrix.
    Methods:
        _load_calib(filepath):
            Loads camera calibration parameters from a file.
        _form_transf(R, t):
            Forms a transformation matrix from rotation and translation.
        get_matches(img):
            Detects ORB keypoints and computes descriptors, then finds matches between the previous and current frames.
        update_pose(q1, q2):
            Updates the camera pose using matched keypoints.
        decomp_essential_mat(E, q1, q2):
            Decomposes the Essential Matrix to obtain rotation and translation.
        update(img):
            Main update function to process a new image frame.
        shutdown():
            Joins all bundle adjustment threads.
        project(points_3d, pose, K):
        optimisable_func(estimated_poses, points_3d, K, observations):
        bundle_adjustment(steps, iterations):
    """

    def __init__(self, init_pose = None, 
                camera_calib_file = 'calib.txt',
                FLANN_INDEX_LSH = 6,
                table_number = 6,
                key_size = 12,
                multi_probe_level = 1,
                ratio_test_threshold = 0.5,  
                knn_match_num = 2,            
                max_features = 3000,         
                bundle_adjustment_steps = [2, 5, 10, 15, 20, 25],
                bundle_adjustment_epochs = 1000,
                bundle_adjustment_learning_rate = 1e-3,
                bundle_adjustment_loss_tolerance = 0.001,
                device = torch.device('cuda' if torch.cuda.is_available() else 'cpu'),
                image_sharpen_kernel = np.array([ 
                [0, -1, 0], 
                [-1, 5, -1], 
                [0, -1, 0]]),
                sharpening = False,
                findEssentialMat_method=cv2.RANSAC,
                findEssentialMat_prob=0.999,
                findEssentialMat_threshold=1.0
                ):

        """
            Initializes the VisualOdometry class with the given parameters.
            Args:
                init_pose (np.ndarray, optional): Initial pose of the camera as a 4x4 transformation matrix. Defaults to None.
                camera_calib_file (str, optional): Path to the camera calibration file. Defaults to 'calib.txt'.
                FLANN_INDEX_LSH (int, optional): Algorithm index for FLANN-based matcher. Defaults to 6.
                table_number (int, optional): Table number for FLANN-based matcher. Defaults to 6.
                key_size (int, optional): Key size for FLANN-based matcher. Defaults to 12.
                multi_probe_level (int, optional): Multi-probe level for FLANN-based matcher. Defaults to 1.
                ratio_test_threshold (float, optional): Threshold for ratio test in feature matching. Defaults to 0.75.
                knn_match_num (int, optional): Number of nearest neighbors to find in KNN matching. Defaults to 2.
                max_features (int, optional): Maximum number of features to detect. Defaults to 3000.
                bundle_adjustment_steps (list, optional): Steps for bundle adjustment. Defaults to [2, 10].
                bundle_adjustment_epochs (int, optional): Number of epochs for bundle adjustment. Defaults to 500.
                bundle_adjustment_learning_rate (float, optional): Learning rate for bundle adjustment. Defaults to 1e-3.
                bundle_adjustment_loss_tolerance (float, optional): Loss tolerance for bundle adjustment. Defaults to 1.
                device (torch.device, optional): Device to run computations on (CPU or GPU). Defaults to GPU if available, otherwise CPU.
                image_sharpen_kernel (np.ndarray, optional): Kernel for image sharpening. Defaults to a 3x3 sharpening kernel.
                sharpening (bool, optional): Flag to enable or disable image sharpening. Defaults to False.
                findEssentialMat_method (int, optional): Method for finding the essential matrix. Defaults to cv2.RANSAC.
                findEssentialMat_prob (float, optional): Probability for RANSAC in finding the essential matrix. Defaults to 0.999.
                findEssentialMat_threshold (float, optional): Threshold for RANSAC in finding the essential matrix. Defaults to 1.0.
            Attributes:
                bundle_adjustment_learning_rate (float): Learning rate for bundle adjustment.
                device (torch.device): Device to run computations on (CPU or GPU).
                bundle_adjustment_steps (list): Steps for bundle adjustment.
                bundle_adjustment_epochs (int): Number of epochs for bundle adjustment.
                bundle_adjustment_loss_tolerance (float): Loss tolerance for bundle adjustment.
                bundle_adjustment_threads (dict): Dictionary to store threads for bundle adjustment.
                lock (threading.Lock): Lock for thread synchronization.
                estimated_poses (list): List of estimated poses.
                points_3d (list): List of 3D points.
                observations (list): List of observations.
                K (np.ndarray): Camera intrinsic matrix.
                P (np.ndarray): Camera projection matrix.
                orb (cv2.ORB): ORB feature detector.
                flann (cv2.FlannBasedMatcher): FLANN-based feature matcher.
                ratio_test_threshold (float): Threshold for ratio test in feature matching.
                knn_match_num (int): Number of nearest neighbors to find in KNN matching.
                prev_img (np.ndarray): Previous image frame.
                prev_keypoints (list): Keypoints from the previous image frame.
                prev_descriptors (np.ndarray): Descriptors from the previous image frame.
                display_frame (np.ndarray): Frame to display.
                image_sharpen_kernel (np.ndarray): Kernel for image sharpening.
                sharpening (bool): Flag to enable or disable image sharpening.
                findEssentialMat_method (int): Method for finding the essential matrix.
                findEssentialMat_prob (float): Probability for RANSAC in finding the essential matrix.
                findEssentialMat_threshold (float): Threshold for RANSAC in finding the essential matrix.
            """

        self.bundle_adjustment_learning_rate = bundle_adjustment_learning_rate
        self.device = device
        self.bundle_adjustment_steps = bundle_adjustment_steps
        self.bundle_adjustment_epochs = bundle_adjustment_epochs
        self.bundle_adjustment_loss_tolerance = bundle_adjustment_loss_tolerance

        self.bundle_adjustment_threads = []
        self.lock = threading.Lock()
        
        if init_pose is not None and init_pose.shape == (4, 4):
            self.estimated_poses = [init_pose]
        else:
            self.estimated_poses = []

        self.points_3d = []
        self.observations = []

        self.K, self.P = self._load_calib(camera_calib_file)

        self.orb = cv2.ORB_create(nfeatures=max_features)
        index_params = dict(algorithm=FLANN_INDEX_LSH, table_number=table_number, key_size=key_size, multi_probe_level=multi_probe_level)
        search_params = dict(checks=50)
        self.flann = cv2.FlannBasedMatcher(indexParams=index_params, searchParams=search_params)

        self.ratio_test_threshold = ratio_test_threshold
        self.knn_match_num = knn_match_num

        self.prev_img = None
        self.prev_keypoints = None
        self.prev_descriptors = None

        self.display_frame = None
        self.image_sharpen_kernel = image_sharpen_kernel
        self.sharpening = sharpening

        self.findEssentialMat_method = findEssentialMat_method
        self.findEssentialMat_prob = findEssentialMat_prob
        self.findEssentialMat_threshold = findEssentialMat_threshold

    #! Helper Functions -----------------------------------------------------
    def _load_calib(self, filepath):
        """
        Loads calibration parameters from a file.
        Args:
            filepath (str): The path to the calibration file.
        Returns:
            tuple: A tuple containing:
                - K (numpy.ndarray): The intrinsic camera matrix (3x3).
                - P (numpy.ndarray): The projection matrix (3x4).
        """

        with open(filepath, 'r') as f:
            params = np.fromstring(f.readline(), dtype=np.float64, sep=' ')
            P = np.reshape(params, (3, 4))
            K = P[0:3, 0:3]
        return K, P

    def _form_transf(self, R, t):
        """
        Forms a 4x4 transformation matrix from a rotation matrix and a translation vector.
        Parameters:
        R (numpy.ndarray): A 3x3 rotation matrix.
        t (numpy.ndarray): A 3x1 translation vector.
        Returns:
        numpy.ndarray: A 4x4 transformation matrix combining the rotation and translation.
        """

        T = np.eye(4, dtype=np.float64)
        T[:3, :3] = R
        T[:3, 3] = t
        return T

    #! Main ORB Functions ---------------------------------------------------
    def get_matches(self, img):
        """
        Detects and matches keypoints between the current and previous frames using ORB and FLANN-based matcher.
        Args:
            img (numpy.ndarray): The input image in BGR format.
        Returns:
            tuple: A tuple containing:
                - q1 (numpy.ndarray or None): Coordinates of matched keypoints in the previous frame. Shape: (M, 2).
                - q2 (numpy.ndarray or None): Coordinates of matched keypoints in the current frame. Shape: (M, 2).
                If there are not enough descriptors or good matches, returns (None, None).
        Raises:
            Exception: If an error occurs during processing.
        """
        try:
            # Convert the input image to grayscale
            img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  # Shape: (H, W)
            
            # Apply sharpening filter if enabled
            if self.sharpening:
                img_gray = cv2.filter2D(img_gray, -1, self.image_sharpen_kernel)  # Shape: (H, W)
            
            # Detect ORB keypoints and compute descriptors
            keypoints, descriptors = self.orb.detectAndCompute(img_gray, None)  # keypoints: list of cv2.KeyPoint, descriptors: (N, 32)
            
            # If there are no previous descriptors or current descriptors, update previous frame data and return
            if self.prev_descriptors is None or descriptors is None:
                self.prev_img = img_gray  # Shape: (H, W)
                self.prev_keypoints = keypoints  # list of cv2.KeyPoint
                self.prev_descriptors = descriptors  # Shape: (N, 32)
                return None, None
            
            # If there are not enough descriptors, return
            if len(descriptors) < self.knn_match_num or len(self.prev_descriptors) < self.knn_match_num:
                print("Not enough descriptors")
                return None, None
            
            # Find matches between previous and current descriptors using FLANN-based matcher
            matches = self.flann.knnMatch(self.prev_descriptors, descriptors, k=self.knn_match_num)  # matches: list of DMatch
            
            # Filter good matches based on distance ratio test
            good = [m for m, n in matches if m.distance < self.ratio_test_threshold * n.distance]  # good: list of DMatch
            
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
        """
        Updates the pose of the camera using matched feature points from two consecutive frames.
        Args:
            q1 (np.ndarray): Matched feature points from the first frame. Shape: (N, 2).
            q2 (np.ndarray): Matched feature points from the second frame. Shape: (N, 2).
        Returns:
            None
        This function performs the following steps:
        1. Computes the Essential Matrix using RANSAC to filter out outliers.
        2. Filters the matched points using the inlier mask to improve accuracy.
        3. Decomposes the Essential Matrix to obtain the rotation (R) and translation (t) matrices.
        4. Forms the transformation matrix from R and t.
        5. Updates the pose with respect to the world frame.
        6. Converts the homogeneous 3D points to world coordinates.
        7. Filters out points with negative depth.
        8. Appends the 3D points and observations for bundle adjustment.
        9. Starts the bundle adjustment threads if not already running.
        Note:
            - The function prints the estimated pose in the world frame.
            - The function handles the first pose differently by inverting the transformation matrix.
            - The function ensures that bundle adjustment threads are running for specified steps.
        """
        # Use RANSAC in findEssentialMat to compute the Essential Matrix and obtain a mask of inliers
        Essential, mask = cv2.findEssentialMat(
            q1, q2, self.K, 
            method=self.findEssentialMat_method, 
            prob=self.findEssentialMat_prob, 
            threshold=self.findEssentialMat_threshold
        )
        
        # Filter the matched points using the inlier mask to improve accuracy
        q1_inliers = q1[mask.ravel() == 1]
        q2_inliers = q2[mask.ravel() == 1]
        
        # Decompose the Essential Matrix using only the inlier points
        R, t, hom_Q, valid_mask = self.decomp_essential_mat(Essential, q1_inliers, q2_inliers)

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

        # Apply the valid mask to hom_Q before transforming
        hom_Q = hom_Q[:, valid_mask]  # Shape: (4, M)
        q2_inliers = q2_inliers[valid_mask]  # Shape: (M, 2)

        # Convert the homogeneous 3D points to world coordinates without inversion
        hom_Q_world = self.estimated_poses[-1] @ hom_Q  # Shape: (4, M)
        Q = hom_Q_world[:3, :] / hom_Q_world[3, :]  # Shape: (3, M)
        
        # **Reprojection Error Filtering Starts Here**
        # Project the 3D points back onto the image plane
        points_3d_world = Q.T  # Shape: (M, 3)
        
        # Compute projection using intrinsic matrix
        projected_points_hom = (self.K @ points_3d_world.T).T  # Shape: (M, 3)
        projected_points = (projected_points_hom[:, :2] / projected_points_hom[:, 2, np.newaxis])  # Shape: (M, 2)
        
        # Call the new filter_points method
        final_points_3d, final_observations = self.filter_points(projected_points, q2_inliers, points_3d_world)
        
        # Append the filtered 3D points and observations for bundle adjustment
        if final_points_3d.size > 0:
            self.points_3d.append(final_points_3d)  # Shape: (M_filtered, 3)
            self.observations.append(final_observations)  # Shape: (M_filtered, 2)
        else:
            print("No 3D points passed the filtering criteria.")
            return
        
        # Start the bundle adjustment threads if not already running
        for step in self.bundle_adjustment_steps:
            thread = threading.Thread(target=self.bundle_adjustment, args=(step, self.bundle_adjustment_epochs))
            thread.start()
            self.bundle_adjustment_threads.append(thread)

    def decomp_essential_mat(self, E, q1, q2):
        """
        Decompose the Essential Matrix into possible rotations and translation, and select the best transformation
        based on the number of points with positive depth.
        Parameters:
        E (numpy.ndarray): The Essential Matrix of shape (3, 3).
        q1 (numpy.ndarray): The matched points in the first image of shape (N, 2).
        q2 (numpy.ndarray): The matched points in the second image of shape (N, 2).
        Returns:
        tuple: A tuple containing:
            - R (numpy.ndarray): The rotation matrix of shape (3, 3).
            - t (numpy.ndarray): The translation vector of shape (3,).
            - hom_Q (numpy.ndarray): The homogeneous coordinates of the 3D points in the current frame of shape (4, N).
            - valid_indices (numpy.ndarray): A boolean array indicating valid points with positive depth of shape (N,).
        """
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
        """
        Updates the visual odometry with a new image.

        Parameters:
        img (numpy.ndarray): The new image frame to process.

        This method extracts feature matches between the current image and the previous image,
        and updates the pose of the camera based on these matches.

        The method first calls `get_matches` to obtain matching feature points between the 
        current image and the previous image. If matches are found (`q1` and `q2` are not None),
        it then calls `update_pose` to update the camera's pose using these matches.

        Returns:
        None
        """
        q1, q2 = self.get_matches(img)  # q1: (M, 2), q2: (M, 2)
        if q1 is not None and q2 is not None:
            self.update_pose(q1, q2)

    def shutdown(self):
        """
        Shuts down the Visual Odometry system by joining all bundle adjustment threads.

        This method iterates over all threads in the `bundle_adjustment_threads` dictionary
        and waits for each thread to complete its execution using the `join` method.
        """
        for thread in self.bundle_adjustment_threads:
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

    def filter_points(self, projected_points, observed_points, points_3d_world):
        """
        Filters 3D points based on reprojection error and depth thresholds.
        
        Parameters:
            projected_points (numpy.ndarray): Projected 2D points on the image plane. Shape: (M, 2)
            observed_points (numpy.ndarray): Observed 2D keypoints in the image. Shape: (M, 2)
            points_3d_world (numpy.ndarray): 3D points in world coordinates. Shape: (M, 3)
        
        Returns:
            tuple: Filtered 3D points and observations.
        """
        # Compute Euclidean distance between observed and projected points
        reprojection_errors = np.linalg.norm(observed_points - projected_points, axis=1)  # Shape: (M,)
        
        # Define a higher reprojection error threshold
        reproj_error_threshold = 150.0  # pixels
        
        # Create a mask for points with reprojection error below the threshold
        error_mask = reprojection_errors < reproj_error_threshold
        
        # Apply the mask to filter 3D points and observations
        filtered_points_3d = points_3d_world[error_mask]
        filtered_observations = observed_points[error_mask]
        
        # Define depth thresholds
        depth_threshold_min = 0.5  # meters
        depth_threshold_max = 50.0  # meters
        depths = filtered_points_3d[:, 2]
        depth_mask = (depths > depth_threshold_min) & (depths < depth_threshold_max)
        
        # Apply depth mask
        final_points_3d = filtered_points_3d[depth_mask]
        final_observations = filtered_observations[depth_mask]
        
        return final_points_3d, final_observations
