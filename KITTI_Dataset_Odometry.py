import cv2
import numpy as np
from Utilities.VisualOdometry import VisualOdometry
from Utilities.Display import draw_trajectory

# Data directory
main_dir = "Data/KITTI_sequence_2/" # <-- Sequence 2
#main_dir = "Data/KITTI_sequence_1/" # <-- Sequence 1

data_dir = f"{main_dir}image_l" # Image directory
max_frame = 50
draw_scale = 5  # Scaling factor for drawing
img_size = (800, 800)  # Trajectory image size

# Initialize the Visual Odometry object
vo = VisualOdometry(camera_calib_file=f"{main_dir}calib.txt")

# Load actual poses from file
actual_poses_file = f"{main_dir}poses.txt"
actual_poses = []
with open(actual_poses_file, 'r') as f:
    for line in f:
        pose = np.array(line.split(), dtype=float).reshape(3, 4)
        actual_poses.append((pose[0, 3], pose[2, 3]))

# Main processing loop
for i in range(max_frame):
    # Read the current frame
    frame = cv2.imread(f"{data_dir}/{i:06d}.png")

    # Update visual odometry with the current frame
    vo.update(frame)

    # Get data from visual odometry
    estimated_poses = vo.estimated_poses
    img_matches = vo.display_frame
    points = vo.points_3d
    pixels = vo.observations

    # Draw the trajectory if poses are available
    if estimated_poses:
        
        # Extract the trajectory path
        path = [(pose[0, 3], pose[2, 3]) for pose in estimated_poses]
        
        # Get the current rotation matrix
        rotation = estimated_poses[-1][:3, :3]

        # Draw the trajectory image with actual poses
        traj_img = draw_trajectory(path, rotation, points, pixels, frame, actual_poses, img_size, draw_scale)

        cv2.imshow("Trajectory", traj_img)

    # Display matches image
    if img_matches is not None:
        cv2.imshow("Matches", img_matches)

    # Exit loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

while True:
    # Get data from visual odometry
    estimated_poses = vo.estimated_poses
    img_matches = vo.display_frame
    points = vo.points_3d
    pixels = vo.observations

    # Draw the trajectory if poses are available
    if estimated_poses:
        
        # Extract the trajectory path
        path = [(pose[0, 3], pose[2, 3]) for pose in estimated_poses]
        
        # Get the current rotation matrix
        rotation = estimated_poses[-1][:3, :3]

        # Draw the trajectory image with actual poses
        traj_img = draw_trajectory(path, rotation, points, pixels, frame, actual_poses, img_size, draw_scale)

        cv2.imshow("Trajectory", traj_img)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Shut down visual odometry and close windows
vo.shutdown()
cv2.destroyAllWindows()
