import torch  # PyTorch for tensor operations
import cv2 as cv  # OpenCV for image processing
import numpy as np  # NumPy for array operations
import matplotlib.pylab as plt  # Matplotlib for visualization
import time  # Time utilities

# Print library versions for reproducibility
print(f"Torch version: {torch.__version__}")
print(f"OpenCV version: {cv.__version__}")

# Function to download example digit images from the web
def download_images():
    """
    Downloads images of handwritten digits '0' and '1' from the web and saves them locally.
    """
    import requests  # For HTTP requests
    urls = {
        "digit_0.jpg": "https://learnopencv.com/wp-content/uploads/2024/07/mnist_0.jpg",
        "digit_1.jpg": "https://learnopencv.com/wp-content/uploads/2024/07/mnist_1.jpg"
    }
    headers = {'User-Agent': 'Mozilla/5.0'}
    for filename, url in urls.items():
        try:
            response = requests.get(url, headers=headers)  # Download image
            response.raise_for_status()
            with open(filename, "wb") as file:
                file.write(response.content)  # Save image to disk
            print(f"Downloaded {filename}")
        except Exception as e:
            print(f"Failed to download {filename}: {e}")

download_images()  # Download images if not present

# Load the images using OpenCV
def load_image_as_numpy(print_info=False, visualize=False) :
    # By default, OpenCV loads images in BGR format which are NumPy arrays
    print("\nLoading images as NumPy arrays using OpenCV...")
    digital_0_array_og = cv.imread("digit_0.jpg")  # Load digit 0 (color)
    digital_1_array_og = cv.imread("digit_1.jpg")  # Load digit 1 (color)

    # CRITICAL: OpenCV loads as BGR, PyTorch needs RGB. Flip the channels.
    digital_0_array_og = cv.cvtColor(digital_0_array_og, cv.COLOR_BGR2RGB)
    digital_1_array_og = cv.cvtColor(digital_1_array_og, cv.COLOR_BGR2RGB)

    digital_0_array_grey = cv.imread("us.jpeg", cv.IMREAD_GRAYSCALE)  # Load grayscale image
    digital_1_array_grey = cv.imread("digit_1.jpg", cv.IMREAD_GRAYSCALE)  # Load digit 1 grayscale

    if print_info:
        # Print shape, min/max, dtype, and type for debugging
        print(f"Digital 0 array shape: {digital_0_array_og.shape}")
        print(f"Min and Max values in Digital 0 array: {digital_0_array_og.min()}, {digital_0_array_og.max()}")
        print(f"Data Type of Digital 0 array: {digital_0_array_og.dtype}")
        print(f"Type of Digital 0 array: {type(digital_0_array_og)}")

        print(f"Digital 0 array grey shape: {digital_0_array_grey.shape}")
        print(f"Min and Max values in Digital 0 array grey: {digital_0_array_grey.min()}, {digital_0_array_grey.max()}")
        print(f"Data Type of Digital 0 array Grey: {digital_0_array_grey.dtype}")
        print(f"Type of Digital 0 array Grey: {type(digital_0_array_grey)}")

        print(f"Digital 1 array shape: {digital_1_array_og.shape}")
        print(f"Digital 1 array grey shape: {digital_1_array_grey.shape}")

    # Visualize the image
    if visualize:
        fig, axs = plt.subplots(1,2, figsize=(10,5))
        axs[0].imshow(digital_0_array_og, cmap='gray',interpolation='none')
        axs[0].set_title("Digit 0 Image")
        axs[0].axis('off')
        axs[1].imshow(digital_1_array_og, cmap="gray", interpolation = 'none')
        axs[1].set_title("Digit 1 Image")
        axs[1].axis('off')
    return digital_0_array_og, digital_0_array_grey, digital_1_array_og, digital_1_array_grey

# Convert NumPy arrays to PyTorch tensors
def numpy_to_tensor(image_0_numpy_array, image_1_numpy_array, print_info=False, visualize=False):
    """
    Converts a NumPy array to a PyTorch tensor.
    """
    print("\nConverting NumPy array to PyTorch tensor...")
    # Convert to float32 and normalize to [0, 1]
    img_0_tensor = torch.tensor(image_0_numpy_array, dtype=torch.float32) / 255.0
    img_1_tensor = torch.tensor(image_1_numpy_array, dtype=torch.float32) / 255.0

    if print_info:
        print("\nNumPy to Tensor Conversion Info...")
        print(f"Image 0 Tensor shape: {img_0_tensor.shape}")
        print(f"Min and Max values in Image 0 Tensor: {img_0_tensor.min().item()}, {img_0_tensor.max().item()}")
        print(f"Data Type of Image 0 Tensor: {img_0_tensor.dtype}")
        print(f"Type of Image 0 Tensor: {type(img_0_tensor)}")

    if visualize:
        fig, axs = plt.subplots(1,2, figsize=(10,5))
        axs[0].imshow(img_0_tensor, cmap='gray',interpolation='none')
        axs[0].set_title("Digit 0 Tensor Image")
        axs[0].axis('off')
        axs[1].imshow(img_1_tensor, cmap="gray", interpolation = 'none')
        axs[1].set_title("Digit 1 Tensor Image")
        axs[1].axis('off')

    return img_0_tensor, img_1_tensor

# Stack two tensors together for batch processing
def tensor_stack(tensor_1, tensor_2, print_info=False, visualize=False):
    """
    Stacks two PyTorch tensors along a new dimension.
    """
    print("\nStacking two tensors...")
    stacked_tensor = torch.stack((tensor_1, tensor_2))
    # Change from (2, H, W, C) to (2, C, H, W) because PyTorch uses Channel-First format and OpenCV uses Channel-Last.
    stacked_tensor = stacked_tensor.permute(0,3,1,2)

    if print_info:
        print("\nTensor Stacking Info...")
        print(f"Stacked Tensor shape: {stacked_tensor.shape}")
        print(f"Min and Max values in Stacked Tensor: {stacked_tensor.min().item()}, {stacked_tensor.max().item()}")
        print(f"Data Type of Stacked Tensor: {stacked_tensor.dtype}")
        print(f"Type of Stacked Tensor: {type(stacked_tensor)}")

    if visualize:
        fig, axs = plt.subplots(1,2, figsize=(10,5))
        # .permute(1, 2, 0) again to convert from Channel-First to Channel-Last for visualization
        axs[0].imshow(stacked_tensor[0].permute(1, 2, 0), cmap='gray',interpolation='none')
        axs[0].set_title("Stacked Tensor - Image 0")
        axs[0].axis('off')
        axs[1].imshow(stacked_tensor[1].permute(1, 2, 0), cmap="gray", interpolation = 'none')
        axs[1].set_title("Stacked Tensor - Image 1")
        axs[1].axis('off')

    return stacked_tensor

def construct_sample_tensors(print_info=False, visualize=False):
    print("\nConstructing sample tensors of various dimensions...")
    zero_d = torch.tensor(7)
    one_d = torch.tensor([0, 1, 2, 3, 4, 5])
    two_d = torch.tensor([[1,3,4], [5,6,7]], dtype=torch.float32)
    three_d = torch.tensor([[[1,2,3], [4,5,6]], [[7,8,9], [10,11,12]]])

    if print_info:
        print("\nConstructed Tensors Info...")
        print(f"Zero-D Tensor: {zero_d}, Shape: {zero_d.shape}, Dtype: {zero_d.dtype}")
        print(f"One-D Tensor: {one_d}, Shape: {one_d.shape}, Dtype: {one_d.dtype}")
        print(f"Two-D Tensor: {two_d}, Shape: {two_d.shape}, Dtype: {two_d.dtype}")
        print(f"Three-D Tensor: {three_d}, Shape: {three_d.shape}, Dtype: {three_d.dtype}")

        print("Retrieving specific elements:")

        print(f"Element from One-D Tensor at index 3: {one_d[3]} and type: {type(one_d[3])}")
        print(f"Element from One-D Tensor at index 3: {one_d[3].item()} and type: {type(one_d[3].item())}")

    if visualize:
        print("\nVisualizing Two-Dimensional Tensor...")
        plt.imshow(two_d, cmap='gray', interpolation='none')
        plt.title("Two-Dimensional Tensor Visualization")
        plt.axis('off')

def additional_methods():
    print("\nDemonstrating additional tensor methods...")
    sample_tensor = torch.tensor([[1, 2, 3], [4, 5, 6]], dtype=torch.float32)
    print(f"Original Tensor:\n{sample_tensor}")
    print(f"\nReshaped Tensor (3x2):\n{sample_tensor.reshape(3, 2)}")
    print(f"\nTransposed Tensor:\n{sample_tensor.T}")
    print(f"\nTensor converted to NumPy array:\n{sample_tensor.numpy()}")
    print(f"\nCloned Tensor:\n{sample_tensor.clone()}")
    print(f"\nDetached Tensor:\n{sample_tensor.detach()}")
    print(f"\nSqueezed Tensor (removing dimensions of size 1):\n{torch.unsqueeze(torch.tensor([[[1], [2], [3]]]), 0).squeeze()}")
    print(f"\nUnsqueezed Tensor (adding a dimension at position 0):\n{torch.unsqueeze(sample_tensor, 0)}")

def tensor_t_vs_reshape():
    print("\nDemonstrating tensor.T vs tensor.reshape(...)")
    t = torch.tensor([[1, 2, 3],
                      [4, 5, 6]])
    print(f"Original Tensor: t -> {t}")
    print(f"Transposed: t.T -> {t.T}")
    print(f"Reshaped: t.reshape(3, 2) -> {t.reshape(3, 2)}")

def numpy_to_tensor_and_back(np_array):
    tensor = torch.tensor(np_array, dtype=torch.float32)
    tensor_again_from_numpy = torch.from_numpy(np_array)
    back_to_numpy = tensor.numpy()

def arthmetic_operations(print_info=False):
    a = torch.tensor([1, 2, 3], dtype=torch.float32)
    b = torch.tensor([4, 5, 6], dtype=torch.float32)
    sum_tensor = a + b
    diff_tensor = a - b
    prod_tensor = a * b
    div_tensor = a / b

    if print_info:
        print("\nArithmetic Operations on Tensors:")
        print(f"a: {a}")
        print(f"b: {b}")
        print(f"Sum: {sum_tensor}")
        print(f"Add using tensor.add: {torch.add(a, b)}")
        print(f"Difference: {diff_tensor}")
        print(f"Difference using tensor.sub: {torch.sub(a, b)}")
        print(f"Product: {prod_tensor}")
        print(f"Product using tensor.mul: {torch.mul(a, b)}")
        print(f"Division: {div_tensor}")
        print(f"Division using tensor.div: {torch.div(a, b)}")

        print(f"Multiple scalar to tensor (a*3) : {a * 3}")

        tensor1 = torch.tensor([[1,2,3],[4,5,6]])
        tensor2 = torch.tensor([[1,2],[3,4],[5,6]])
        print(torch.mm(tensor1,tensor2))
        tensor3 = torch.mm(tensor1,tensor2)
        print(f"Matrix multiplication using torch.mm: \n{tensor3}")
        print(f"Matrix multiplication using @ operator: \n{tensor1 @ tensor2}")
        print(f"Shape and size of resulting tensor: {tensor3.shape}, {tensor3.size()}")

def broadcasting_example():
    print("\nBroadcasting Example:")
    tensor_a = torch.tensor([[1, 2, 3], [4, 5, 6]], dtype=torch.float32)
    tensor_b = torch.tensor([10, 20, 30], dtype=torch.float32)
    result = tensor_a + tensor_b
    print(f"Tensor A:\n{tensor_a}")
    print(f"Tensor B:\n{tensor_b}")
    print(f"Result of A + B (broadcasted):\n{result}")

def cpu_gpu_example():
    # 1. Define the Device (Check if Mac GPU is available)
    if torch.backends.mps.is_available():
        device = torch.device("mps")
        print("Success: Using Apple Metal (MPS) acceleration!")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
        print("Success: Using NVIDIA CUDA acceleration!")
    else:
        device = torch.device("cpu")
        print("MPS not found. Using CPU.")

    # 2. Create a Tensor (Default: It is born on the CPU)
    x = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
    print(f"Original Device: {x.device}")

    # 3. Teleport to GPU
    x_gpu = x.to(device)

    print(f"New Device:      {x_gpu.device}")
    # Output should be: device(type='mps', index=0)

    # 4. Do Math on GPU
    result = x_gpu @ x_gpu
    print(result)


BATCH_SIZE = 64
INPUT_SIZE = 4096
HIDDEN_SIZE = 4096
LOOPS = 100  # Do it 100 times!

def stress_test(device_name):
    device = torch.device(device_name)

    # Create tensors
    x = torch.randn(BATCH_SIZE, INPUT_SIZE).to(device)
    w = torch.randn(INPUT_SIZE, HIDDEN_SIZE).to(device)

    # --- WARM UP ---
    # Run once to compile the graph/kernels
    for _ in range(5):
        _ = x @ w

    # --- THE RACE (Sustained Load) ---
    start = time.time()
    for _ in range(LOOPS):
        # The math happens here
        result = x @ w
        # We DO NOT move to cpu() inside the loop.
        # We want to keep the GPU busy without interruption.

    # Force synchronization only ONCE at the very end
    if device_name == "mps":
        torch.mps.synchronize()

    end = time.time()
    return end - start

print(f"Stress Test: {LOOPS} iterations of Matrix Mult {BATCH_SIZE}x{INPUT_SIZE}")
print("-" * 40)

# 1. CPU
print("Running on CPU...")
cpu_time = stress_test("cpu")
print(f"CPU Time: {cpu_time:.4f} seconds")

# 2. GPU
if torch.backends.mps.is_available():
    print("\nRunning on MPS (GPU)...")
    gpu_time = stress_test("mps")
    print(f"GPU Time: {gpu_time:.4f} seconds")

    speedup = cpu_time / gpu_time
    print(f"\nResult: GPU is {speedup:.1f}x faster on sustained load!")

# rgb_0, grey_0, rgb_1, grey_1 = load_image_as_numpy()
# tensor_1, tensor_2 = numpy_to_tensor(rgb_0, rgb_1)
# tensor_stack(tensor_1, tensor_2)
# construct_sample_tensors()
# #additional_methods()
# #tensor_t_vs_reshape()
# arthmetic_operations()
# broadcasting_example()
# cpu_gpu_example()
#plt.show()
