import cv2
import numpy as np

# Function to draw trajectory on an image
def draw_trajectory(trajectory, rotation, all_points, all_pixels, frame, actual_poses=None, img_size=(800, 800), draw_scale=1):
    
    """
    Draws a visual representation of a trajectory on a blank image.
    Parameters:
    trajectory (list of list of float): The trajectory path to be drawn.
    rotation (numpy.ndarray): The rotation matrix representing the current orientation.
    all_points (list of numpy.ndarray): List of 3D points from each frame.
    all_pixels (list of numpy.ndarray): List of corresponding pixel coordinates for the 3D points.
    frame (numpy.ndarray): The current frame from which colors are sampled.
    actual_poses (list of list of float, optional): The actual trajectory path to be drawn in green. Defaults to None.
    img_size (tuple of int, optional): The size of the output image. Defaults to (800, 800).
    draw_scale (float, optional): The scale factor for drawing. Defaults to 1.
    Returns:
    numpy.ndarray: The image with the drawn trajectory.
    """
    # Create a blank image for the trajectory
    traj_img = np.zeros((img_size[1], img_size[0], 3), dtype=np.uint8)

    # Draw the actual trajectory path if provided
    if actual_poses:
        actual_array = np.array(actual_poses)
        actual_coords = (actual_array * draw_scale + np.array(img_size) // 2).astype(int)
        # Draw the actual path in green
        for coord in actual_coords:
            cv2.circle(traj_img, tuple(coord), 2, (0, 200, 0), -1)

    #random color
    # Draw the 3D points from the last frame
    counter = 0
    for points, pixels in zip(all_points, all_pixels):
        """points = all_points[-1]
        pixels = all_pixels[-1]"""

        # Convert 3D points to 2D image coordinates
        img_coords = (points[:, [0, 2]] * draw_scale + np.array(img_size) // 2).astype(int)
        # Ensure coordinates are within image bounds
        img_coords = np.clip(img_coords, [0, 0], np.array(img_size) - 1)

        # Get colors from the frame
        pixel_coords = pixels.astype(int)
        pixel_coords[:, 0] = np.clip(pixel_coords[:, 0], 0, frame.shape[1] - 1)
        pixel_coords[:, 1] = np.clip(pixel_coords[:, 1], 0, frame.shape[0] - 1)
        #colors = frame[pixel_coords[:, 1], pixel_coords[:, 0]]
        #random color with seed
        np.random.seed(counter)
        colors = np.random.randint(0, 255, 3)
        counter += 10
        if counter > 255:
            counter = 0
        # Draw points on the trajectory image
        traj_img[img_coords[:, 1], img_coords[:, 0]] = colors

    # Draw the trajectory path
    if trajectory:
        traj_array = np.array(trajectory)
        traj_coords = (traj_array * draw_scale + np.array(img_size) // 2).astype(int)
        # Draw the path
        for coord in traj_coords:
            cv2.circle(traj_img, tuple(coord), 2, (255, 255, 255), -1)

        # Draw the current position in red
        cv2.circle(traj_img, tuple(traj_coords[-1]), 2, (0, 0, 255), -1)

        # Draw the orientation as an arrow
        forward_vector = rotation @ np.array([0, 0, 1]).reshape(3, 1)
        start_point = tuple(traj_coords[-1])
        end_point = (
            int(start_point[0] + forward_vector[0][0] * 50),
            int(start_point[1] + forward_vector[2][0] * 50),
        )
        cv2.arrowedLine(traj_img, start_point, end_point, (100, 100, 100), 2)

    # Flip the image vertically so the origin is at the bottom left
    traj_img = cv2.flip(traj_img, 0)

    # Prepare text for translation and rotation
    translation_text = f"Translation: {np.round(trajectory[-1], 2)}"
    rotation_vector, _ = cv2.Rodrigues(rotation)
    rotation_text = f"Rotation: {np.round(rotation_vector.flatten(), 2)}"

    # Set font parameters for a neat appearance
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.5
    color = (255, 255, 255)
    thickness = 1

    # Get image dimensions
    height, width = traj_img.shape[:2]

    # Positions for the text in the bottom-left corner
    text_position_translation = (10, height - 30)
    text_position_rotation = (10, height - 10)

    # Add text to the image
    cv2.putText(traj_img, translation_text, text_position_translation, font, font_scale, color, thickness, cv2.LINE_AA)
    cv2.putText(traj_img, rotation_text, text_position_rotation, font, font_scale, color, thickness, cv2.LINE_AA)

    return traj_img
