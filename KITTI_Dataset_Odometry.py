import cv2
from Utilities.VisualOdometry import VisualOdometry
from Utilities.Display import draw_trajectory

# Data directory
#data_dir = "Data/KITTI_sequence_1/image_l" # <-- Sequence 1
data_dir = "Data/KITTI_sequence_2/image_l" # <-- Sequence 2
max_frame = 50
draw_scale = 2  # Scaling factor for drawing
img_size = (800, 800)  # Trajectory image size

# Initialize the Visual Odometry object
vo = VisualOdometry(camera_calib_file="Data/KITTI_sequence_2/calib.txt")

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

        # Draw the trajectory image
        traj_img = draw_trajectory(path, rotation, points, pixels, frame, img_size, draw_scale)

        cv2.imshow("Trajectory", traj_img)

    # Display matches image
    if img_matches is not None:
        cv2.imshow("Matches", img_matches)

    # Exit loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Shut down visual odometry and close windows
vo.shutdown()
cv2.destroyAllWindows()
