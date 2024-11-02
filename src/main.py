import cv2
import numpy as np
import matplotlib.pyplot as plt
import os

# Ensure the output directory exists
os.makedirs('output_frames', exist_ok=True)

# Function to apply Gaussian splatting to an image
def gaussian_splatting(image, splat_size=5, splat_intensity=50):
    # Create an empty canvas to draw the splats
    splatted_image = np.zeros_like(image)

    # Get the height and width of the image
    height, width, _ = image.shape

    # Loop over each pixel in the image (sampling some pixels instead of all can improve speed)
    for y in range(0, height, splat_size):
        for x in range(0, width, splat_size):
            # Get the pixel color
            color = image[y, x]

            # Create a Gaussian splat around the point
            y_min = max(0, y - splat_size)
            y_max = min(height, y + splat_size)
            x_min = max(0, x - splat_size)
            x_max = min(width, x + splat_size)

            # Create a grid to draw the Gaussian
            y_grid, x_grid = np.meshgrid(np.arange(y_min, y_max), np.arange(x_min, x_max), indexing='ij')
            y_diff = y_grid - y
            x_diff = x_grid - x

            # Gaussian formula
            gaussian = np.exp(-(x_diff**2 + y_diff**2) / (2 * splat_intensity**2))
            gaussian = gaussian[:, :, None]  # Add an extra dimension for broadcasting

            # Apply the Gaussian splat to the image
            splatted_image[y_min:y_max, x_min:x_max] += (gaussian * color).astype(np.uint8)

    return splatted_image

# Path to the video file
input_video_path = 'your_video.mp4'
cap = cv2.VideoCapture(input_video_path)

# Check if video opened successfully
if not cap.isOpened():
    print("Error: Could not open video file.")
    exit()

# Get video properties for saving the output
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = int(cap.get(cv2.CAP_PROP_FPS))
output_video_path = 'gaussian_splat_output.mp4'

# Create VideoWriter to save output video
out = cv2.VideoWriter(output_video_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (frame_width, frame_height))

frame_count = 0

# Process the video frame-by-frame
while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Apply Gaussian splatting to the frame
    splatted_frame = gaussian_splatting(frame)

    # Save the splatted frame as an image
    frame_filename = f'output_frames/splatted_frame_{frame_count:04d}.png'
    cv2.imwrite(frame_filename, splatted_frame)

    # Write the processed frame to the output video
    out.write(splatted_frame)

    # Optionally visualize the splatted frame using Matplotlib
    if frame_count % 30 == 0:  # Visualize every 30th frame
        plt.figure(figsize=(10, 5))
        plt.subplot(1, 2, 1)
        plt.title('Original Frame')
        plt.imshow(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        plt.axis('off')

        plt.subplot(1, 2, 2)
        plt.title('Gaussian Splatted Frame')
        plt.imshow(cv2.cvtColor(splatted_frame, cv2.COLOR_BGR2RGB))
        plt.axis('off')

        plt.show()

    # Display the original and splatted frames (side-by-side)
    combined_frame = cv2.hconcat([frame, splatted_frame])
    cv2.imshow('Original and Gaussian Splatting', combined_frame)

    # Exit on pressing 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

    frame_count += 1

# Release resources
cap.release()
out.release()
cv2.destroyAllWindows()
