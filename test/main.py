import cv2
import numpy as np
import os

def hsv_to_black_and_white(image_path, color_ranges, output_folder):
    # Load the image
    original_image = cv2.imread(image_path)
    if original_image is None:
        print("Error: Could not read the image.")
        return

    # Create a copy of the image to accumulate all centerlines
    combined_image = original_image.copy()

    # Create output folder if it doesn't exist
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    for color_name, hsv_range in color_ranges.items():
        lower_hsv, upper_hsv, line_color = hsv_range

        # Create a copy of the original image to save each color's centerline separately
        image_with_single_color_lines = original_image.copy()

        # Convert the image to the HSV color space
        hsv_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2HSV)

        # Create a mask with the HSV range (white for in-range, black for out-of-range)
        mask = cv2.inRange(hsv_image, lower_hsv, upper_hsv)

        # Invert the mask to make the objects in the range black (0), others white (255)
        black_and_white_image = cv2.bitwise_not(mask)

        # Ensure that the image is saved as a binary black/white image with only 0 and 255 values
        black_and_white_image[black_and_white_image > 0] = 255

        # Find contours (black blobs)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if len(contours) == 0:
            print(f"No {color_name} blobs detected.")
            continue

        # Find the largest contour
        largest_contour = max(contours, key=cv2.contourArea)

        # Compute the centroid of the largest contour (blob)
        M = cv2.moments(largest_contour)
        if M["m00"] == 0:
            print(f"Invalid contour for {color_name}.")
            continue

        cX = int(M["m10"] / M["m00"])  # X-coordinate of the centroid
        cY = int(M["m01"] / M["m00"])  # Y-coordinate of the centroid

        # Get image dimensions
        height, width = black_and_white_image.shape

        # Compute percentage of width and height
        perc_x = (cX / width) * 100
        perc_y = (cY / height) * 100

        print(f"Center of the {color_name} blob: ({perc_x:.2f}%, {perc_y:.2f}%)")

        # Draw colored lines at the centroid on the individual image for this color
        line_width = int(width * 0.01)  # 1% of image width
        cv2.line(image_with_single_color_lines, (0, cY), (width, cY), line_color, line_width)  # Horizontal line
        cv2.line(image_with_single_color_lines, (cX, 0), (cX, height), line_color, line_width)  # Vertical line

        # Draw colored lines at the centroid on the combined image
        cv2.line(combined_image, (0, cY), (width, cY), line_color, line_width)  # Horizontal line
        cv2.line(combined_image, (cX, 0), (cX, height), line_color, line_width)  # Vertical line

        # Define output paths
        output_image_path = os.path.join(output_folder, f"processed_image_{color_name}.png")
        bw_image_path = os.path.join(output_folder, f"black_and_white_image_{color_name}.png")

        # Save the output image with lines for the current color
        cv2.imwrite(output_image_path, image_with_single_color_lines)
        print(f"Processed image with {color_name} lines saved as {output_image_path}")

        # Save the black and white image
        cv2.imwrite(bw_image_path, black_and_white_image)
        print(f"Black and white image saved as {bw_image_path}")

    # Save the combined image with all centerlines
    combined_image_path = os.path.join(output_folder, "combined_centerlines_image.png")
    cv2.imwrite(combined_image_path, combined_image)
    print(f"Combined image with all centerlines saved as {combined_image_path}")

if __name__ == "__main__":
    # Example image path
    input_image_path = 'IMG_1429.jpg'  # Use the image you uploaded
    output_folder = "Output"  # Folder to save the images

    # Define the HSV ranges for yellow, red, and blue (approximate)
    color_ranges = {
        "yellow": (np.array([20, 100, 100]), np.array([30, 255, 255]), (0, 255, 255)),  # Yellow in BGR
        "red": (np.array([0, 100, 100]), np.array([10, 255, 255]), (0, 0, 255)),  # Red in BGR
        "blue": (np.array([100, 150, 0]), np.array([140, 255, 255]), (255, 0, 0))  # Blue in BGR
    }

    hsv_to_black_and_white(input_image_path, color_ranges, output_folder)
