import cv2
import numpy as np
import os

def hsv_to_black_and_white(image_path, color_ranges, output_folder, max_contours=5, confidence_threshold=60, circle_radius_percentage=2):
    # Load the image
    original_image = cv2.imread(image_path)
    if original_image is None:
        print("Error: Could not read the image.")
        return

    # Create output folder if it doesn't exist
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # Create copies of the original image for various outputs
    combined_image = np.copy(original_image)
    multi_combined_image = np.copy(original_image)
    circle_image = np.copy(original_image)  # Image for circles around centroids

    for color_name, hsv_range in color_ranges.items():
        lower_hsv, upper_hsv, line_color = hsv_range

        # Convert the image to the HSV color space
        hsv_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2HSV)

        # Create a mask with the HSV range (white for in-range, black for out-of-range)
        mask = cv2.inRange(hsv_image, lower_hsv, upper_hsv)

        # Invert the mask to make the objects in the range black (0), others white (255)
        black_and_white_image = cv2.bitwise_not(mask)
        black_and_white_image[black_and_white_image > 0] = 255

        # Find contours (black blobs)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if len(contours) == 0:
            print(f"No {color_name} blobs detected.")
            continue

        # Sort contours by size (area)
        sorted_contours = sorted(contours, key=cv2.contourArea, reverse=True)
        largest_contour_area = cv2.contourArea(sorted_contours[0]) if sorted_contours else 0

        # For combined image, add centerlines for the most confident contours
        for i, contour in enumerate(sorted_contours[:1]):  # Only the largest contour
            if len(contour) == 0:
                continue

            # Compute the centroid of the contour
            M = cv2.moments(contour)
            if M["m00"] == 0:
                continue

            cX = int(M["m10"] / M["m00"])  # X-coordinate of the centroid
            cY = int(M["m01"] / M["m00"])  # Y-coordinate of the centroid

            # Draw colored lines at the centroid on the combined image
            line_width = int(original_image.shape[1] * 0.01)  # 1% of image width
            cv2.line(combined_image, (0, cY), (original_image.shape[1], cY), line_color, line_width)  # Horizontal line
            cv2.line(combined_image, (cX, 0), (cX, original_image.shape[0]), line_color, line_width)  # Vertical line

        # Create mini folders for each contour
        for i, contour in enumerate(sorted_contours[:max_contours]):
            if len(contour) == 0:
                continue

            # Compute the centroid of the contour
            M = cv2.moments(contour)
            if M["m00"] == 0:
                continue

            cX = int(M["m10"] / M["m00"])  # X-coordinate of the centroid
            cY = int(M["m01"] / M["m00"])  # Y-coordinate of the centroid

            # Get image dimensions
            height, width = black_and_white_image.shape

            # Compute percentage of width and height
            perc_x = (cX / width) * 100
            perc_y = (cY / height) * 100

            # Compute contour confidence
            contour_area = cv2.contourArea(contour)
            confidence = (contour_area / largest_contour_area) * 100
            confidence = min(100, confidence)  # Cap the confidence at 100%

            print(f"Center of the {color_name} blob {i + 1}: ({perc_x:.2f}%, {perc_y:.2f}%)")
            print(f"Confidence for contour {i + 1}: {confidence:.2f}%")

            # Draw colored lines at the centroid on the individual image for this contour
            contour_image = np.copy(original_image)  # Copy to draw contour lines on
            line_width = int(width * 0.01)  # 1% of image width
            cv2.line(contour_image, (0, cY), (width, cY), line_color, line_width)  # Horizontal line
            cv2.line(contour_image, (cX, 0), (cX, height), line_color, line_width)  # Vertical line

            # Create mini folder for contour
            contour_folder = os.path.join(output_folder, f"Contour_{i + 1}")
            if not os.path.exists(contour_folder):
                os.makedirs(contour_folder)

            # Define output paths
            output_image_path = os.path.join(contour_folder, f"processed_image_{color_name}.png")
            bw_image_path = os.path.join(contour_folder, f"black_and_white_image_{color_name}.png")

            # Save the output image with lines for the current color
            cv2.imwrite(output_image_path, contour_image)
            print(f"Processed image with {color_name} lines for contour {i + 1} saved as {output_image_path}")

            # Save the black and white image
            cv2.imwrite(bw_image_path, black_and_white_image)
            print(f"Black and white image for contour {i + 1} saved as {bw_image_path}")

            # For high-confidence contours, add their centerlines to the multi_combined_image
            if confidence > confidence_threshold:
                cv2.line(multi_combined_image, (0, cY), (original_image.shape[1], cY), line_color, line_width)  # Horizontal line
                cv2.line(multi_combined_image, (cX, 0), (cX, original_image.shape[0]), line_color, line_width)  # Vertical line

                # Draw circles around high-confidence centroids
                radius = int(min(original_image.shape[:2]) * circle_radius_percentage / 100)  # Radius in pixels
                cv2.circle(circle_image, (cX, cY), radius, line_color, 5)  # Draw circle with thickness of 2

    # Save the combined image with centerlines of the most confident contours
    combined_image_path = os.path.join(output_folder, "combined_centerlines_image.png")
    cv2.imwrite(combined_image_path, combined_image)
    print(f"Combined image with centerlines of the most confident contours saved as {combined_image_path}")

    # Save the multi-combined image with centerlines of high-confidence contours
    multi_combined_image_path = os.path.join(output_folder, "multi_combined_centerlines_image.png")
    cv2.imwrite(multi_combined_image_path, multi_combined_image)
    print(f"Multi-combined image with high-confidence contours saved as {multi_combined_image_path}")

    # Save the circle image with circles around high-confidence centroids
    circle_image_path = os.path.join(output_folder, "circles_around_centroids_image.png")
    cv2.imwrite(circle_image_path, circle_image)
    print(f"Image with circles around centroids saved as {circle_image_path}")

if __name__ == "__main__":
    # Example image path
    input_image_path = 'IMG_1432.jpg'  # Use the image you uploaded
    output_folder = "Output"  # Folder to save the images

    # Define the HSV ranges for yellow, red, and blue (wider range for blue)
    color_ranges = {
        "yellow": (np.array([20, 100, 100]), np.array([30, 255, 255]), (0, 255, 255)),  # Yellow in BGR
        "red": (np.array([0, 100, 100]), np.array([10, 255, 255]), (0, 0, 255)),  # Red in BGR
        "blue": (np.array([100, 50, 50]), np.array([140, 255, 255]), (255, 0, 0))  # Wider range for Blue in BGR
    }

    hsv_to_black_and_white(input_image_path, color_ranges, output_folder)
