import cv2  # OpenCV for image processing
import sys  # System utilities

# 1. Load the image (Force grayscale to make it readable)
img = cv2.imread("digit_0.jpg", cv2.IMREAD_GRAYSCALE)  # Load image as grayscale

if img is None:
    print("Error: Image not found. Download it first!")  # Error message if image is missing
    sys.exit()  # Exit program if image not found

# 2. Iterate through every row
print("--- BEGIN IMAGE ARRAY ---")
for row in img:
    # Print each row as a list of numbers
    # We use formatting to make sure every number takes up 3 spaces
    row_str = " ".join([f"{pixel:3}" for pixel in row])  # Format each pixel value
    print(row_str)  # Print formatted row
print("--- END IMAGE ARRAY ---")