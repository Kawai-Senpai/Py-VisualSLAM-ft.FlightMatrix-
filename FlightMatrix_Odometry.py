import cv2
from Utilities.VisualOdometry import VisualOdometry
from Utilities.Display import draw_trajectory
from flightmatrix.bridge import FlightMatrixBridge
from flightmatrix.utilities import DroneController

resolution = (1226, 370)  # Set frame resolution

# Initialize the FlightMatrix bridge
bridge = FlightMatrixBridge(resolution=resolution)  # Set frame resolution

# Enable logging to file
bridge.set_write_to_file(False)

# Initialize the drone controller
drone = DroneController(bridge)

# Initialize the Visual Odometry object
vo = VisualOdometry(camera_calib_file="Data/FlightMatrixCalib.txt")

# Main processing loop
while True:
    # Retrieve the current left camera frame from the simulation
    left_frame_data = bridge.get_left_frame()
    frame = left_frame_data['frame']

    # Update visual odometry with the current frame
    vo.update(frame)

    # Get data from visual odometry
    estimated_poses = vo.estimated_poses
    img_matches = vo.display_frame
    points = vo.points_3d
    pixels = vo.observations

    #drone.move_forward(0.1)  # Move forward at speed 0.1

    # Draw the trajectory if poses are available
    if estimated_poses:
        # Extract the trajectory path
        path = [(pose[0, 3], pose[2, 3]) for pose in estimated_poses]

        # Get the current rotation matrix
        rotation = estimated_poses[-1][:3, :3]

        # Draw the trajectory image
        traj_img = draw_trajectory(path, rotation, points, pixels, frame, img_size=resolution, draw_scale=2)
        cv2.imshow("Trajectory", traj_img)

    # Display matches image
    if img_matches is not None:
        cv2.imshow("Matches", img_matches)

    # Exit loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Stop the drone
drone.stop()

# Shut down visual odometry and close windows
vo.shutdown()
cv2.destroyAllWindows()