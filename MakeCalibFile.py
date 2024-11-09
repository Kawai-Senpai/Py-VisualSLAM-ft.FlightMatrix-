import numpy as np
import math

def get_float_input(prompt, default=None):
    while True:
        try:
            user_input = input(prompt)
            if user_input == "" and default is not None:
                return default
            return float(user_input)
        except ValueError:
            print("Invalid input. Please enter a numeric value.")

def rotation_matrix_from_angles(pitch, yaw, roll):
    # Convert angles from degrees to radians
    pitch = math.radians(pitch)
    yaw = math.radians(yaw)
    roll = math.radians(roll)
    
    # Calculate rotation matrices for each axis
    Rx = np.array([
        [1, 0, 0],
        [0, math.cos(pitch), -math.sin(pitch)],
        [0, math.sin(pitch), math.cos(pitch)]
    ])
    
    Ry = np.array([
        [math.cos(yaw), 0, math.sin(yaw)],
        [0, 1, 0],
        [-math.sin(yaw), 0, math.cos(yaw)]
    ])
    
    Rz = np.array([
        [math.cos(roll), -math.sin(roll), 0],
        [math.sin(roll), math.cos(roll), 0],
        [0, 0, 1]
    ])
    
    # Combined rotation matrix
    R = Rz @ Ry @ Rx
    return R

def main():
    print("Camera Calibration File Generator")

    # Get camera parameters from the user
    image_width = get_float_input("Enter the image width in pixels: ", 1226) # Default value 1226
    image_height = get_float_input("Enter the image height in pixels: ", 370) # Default value 370
    fov_deg = get_float_input("Enter the field of view (FOV) in degrees: ", 90) # Default value 90

    # Ask if the user has sensor dimensions
    use_default_sensor = input("Do you have the sensor dimensions? (y/n): ").strip().lower() != 'y'

    if use_default_sensor:
        # Calculate aspect ratio
        aspect_ratio = image_width / image_height

        # Assuming a standard full-frame sensor width (36mm)
        sensor_width_mm = 36.0
        sensor_height_mm = sensor_width_mm / aspect_ratio
        print(f"Using calculated sensor dimensions: {sensor_width_mm:.2f}mm x {sensor_height_mm:.2f}mm")
    else:
        # Get sensor dimensions from the user
        sensor_width_mm = get_float_input("Enter the sensor width in mm: ")
        sensor_height_mm = get_float_input("Enter the sensor height in mm: ")

    # Convert FOV to focal length in mm
    focal_length_mm = (0.5 * sensor_width_mm) / math.tan(math.radians(fov_deg / 2))

    # Compute focal lengths in pixels
    f_x = (focal_length_mm * image_width) / sensor_width_mm
    f_y = (focal_length_mm * image_height) / sensor_height_mm

    # Compute principal point (assuming it's at the center of the image)
    c_x = image_width / 2
    c_y = image_height / 2

    # Create the intrinsic matrix K
    K = np.array([
        [f_x, 0, c_x],
        [0, f_y, c_y],
        [0, 0, 1]
    ])

    # Create the first projection matrix P1 (assuming no initial rotation or translation)
    P1 = np.hstack((K, np.zeros((3, 1))))

    # Get rotation and translation from the user for the second projection matrix
    pitch = get_float_input("Enter the pitch angle in degrees: ", 0) # Default value 0
    yaw = get_float_input("Enter the yaw angle in degrees: ", 0) # Default value 0
    roll = get_float_input("Enter the roll angle in degrees: ", 0) # Default value 0
    tx = get_float_input("Enter the translation in x (mm): ", 0) # Default value 0
    ty = get_float_input("Enter the translation in y (mm): ", 0) # Default value 0
    tz = get_float_input("Enter the translation in z (mm): ", 0) # Default value 0

    # Compute the rotation matrix
    R = rotation_matrix_from_angles(pitch, yaw, roll)

    # Create the translation vector
    t = np.array([[tx], [ty], [tz]])

    # Create the second projection matrix P2
    P2 = np.hstack((K @ R, K @ t))

    output_file = "Data/FlightMatrixCalib.txt"

    # Flatten the projection matrices and write to the file
    with open(output_file, 'w') as f:
        np.savetxt(f, P1.reshape(1, -1), fmt='%.12e')
        np.savetxt(f, P2.reshape(1, -1), fmt='%.12e')

    print(f"Calibration file saved to {output_file}")

if __name__ == "__main__":
    main()
