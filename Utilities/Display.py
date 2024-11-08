
import cv2
import numpy as np

# Function to draw trajectory on an image
def draw_trajectory(trajectory, rotation, all_points, all_pixels, frame, img_size=(800, 800), draw_scale=1):
    # Create a blank image for the trajectory
    traj_img = np.zeros((img_size[1], img_size[0], 3), dtype=np.uint8)

    # Draw the 3D points from the last frame
    if all_points:
        points = all_points[-1]
        pixels = all_pixels[-1]
        # Filter points within a reasonable range
        valid_mask = np.all(np.abs(points) < 1e6, axis=1)
        valid_points = points[valid_mask]
        valid_pixels = pixels[valid_mask]

        # Convert 3D points to 2D image coordinates
        img_coords = (valid_points[:, [0, 2]] * draw_scale + np.array(img_size) // 2).astype(int)
        # Ensure coordinates are within image bounds
        img_coords = np.clip(img_coords, [0, 0], np.array(img_size) - 1)

        # Get colors from the frame
        pixel_coords = valid_pixels.astype(int)
        pixel_coords[:, 0] = np.clip(pixel_coords[:, 0], 0, frame.shape[1] - 1)
        pixel_coords[:, 1] = np.clip(pixel_coords[:, 1], 0, frame.shape[0] - 1)
        colors = frame[pixel_coords[:, 1], pixel_coords[:, 0]]

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

    return traj_img
