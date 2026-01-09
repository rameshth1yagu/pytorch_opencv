import torch
import cv2 as cv
import numpy as np
import matplotlib.pylab as plt

print(f"Torch version: {torch.__version__}")
print(f"OpenCV version: {cv.__version__}")

def download_images():
    """
    Downloads images of handwritten digits '0' and '1' from the web and saves them locally.
    """
    import requests
    urls = {
        "digit_0.jpg": "https://learnopencv.com/wp-content/uploads/2024/07/mnist_0.jpg",
        "digit_1.jpg": "https://learnopencv.com/wp-content/uploads/2024/07/mnist_1.jpg"
    }
    headers = {'User-Agent': 'Mozilla/5.0'}
    for filename, url in urls.items():
        try:
            response = requests.get(url, headers=headers)
            response.raise_for_status()
            with open(filename, "wb") as file:
                file.write(response.content)
            print(f"Downloaded {filename}")
        except Exception as e:
            print(f"Failed to download {filename}: {e}")

download_images()

# Load the images using OpenCV
def load_image_as_array() :
    digital_0_array_og = cv.imread("digit_0.jpg")
    digital_1_array_og = cv.imread("digit_1.jpg")

    digital_0_array_grey = cv.imread("digit_0.jpg", cv.IMREAD_GRAYSCALE)
    digital_1_array_grey = cv.imread("digit_1.jpg", cv.IMREAD_GRAYSCALE)

    print(f"Digital 0 array shape: {digital_0_array_og.shape}")
    print(f"Digital 0 array grey shape: {digital_0_array_grey.shape}")

    # Visualize the image
    fig, axs = plt.subplots(1,2, figsize=(10,5))
    axs[0].imshow(digital_0_array_og, cmap='gray',interpolation='none')
    axs[0].set_title("Digit 0 Image")
    axs[0].axis('off')
    axs[1].imshow(digital_1_array_og, cmap="gray", interpolation = 'none')
    axs[1].set_title("Digit 1 Image")
    axs[1].axis('off')

load_image_as_array()

plt.show()



