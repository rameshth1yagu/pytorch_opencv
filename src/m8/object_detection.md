Here is the comprehensive content for your `m8_object_detection.md` file. It covers the theory, architecture, and code implementation details we discussed.

---

# Object Detection with Faster R-CNN

## 1. What is Object Detection?

**Object Detection** goes a step further than Image Classification. While classification answers "What is in this image?", object detection answers two questions simultaneously:

1. **What** is in the image? (Classification)
2. **Where** is it? (Localization via Bounding Boxes)

### Key Terminology

* **Bounding Box:** A rectangle defined by coordinates `(x1, y1)` (top-left) and `(x2, y2)` (bottom-right) that encloses an object.
* **Confidence Score:** A probability (0.0 to 1.0) indicating how certain the model is that the box contains the predicted object.
* **IoU (Intersection over Union):** A metric used to evaluate accuracy by measuring how much the predicted box overlaps with the ground truth box.

---

## 2. The Architecture: Faster R-CNN

We use **Faster R-CNN** (Region-based Convolutional Neural Network), a standard architecture for high-accuracy detection. It is known as a "Two-Stage Detector":

1. **Stage 1 (Region Proposal):** The model scans the image and suggests thousands of "Regions of Interest" (RoIs) where an object *might* be. It doesn't know *what* the object is yet, just that it looks like a "blob" distinct from the background.
2. **Stage 2 (Classification & Refinement):** The model looks closely at these proposed regions to:
* Classify the object (e.g., "Person", "Car").
* Refine the box coordinates to fit the object perfectly (Regression).



### The Backbone: ResNet50 + FPN

The "brain" powering Faster R-CNN is **ResNet50** combined with a **Feature Pyramid Network (FPN)**.

* **ResNet50:** Extracts features (edges, textures, shapes).
* **FPN (Feature Pyramid Network):** Creates a "pyramid" of features at different scales. This allows the model to detect **tiny objects** (using high-resolution bottom layers) and **large objects** (using semantic top layers) simultaneously.

---

## 3. Implementation Steps

### Step 1: Loading the Pre-trained Model

We use `torchvision` to load a model pre-trained on the **COCO Dataset** (80 common object categories).

```python
import torchvision

def load_model():
    # Load Faster R-CNN with ResNet50-FPN backbone
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(
        weights=torchvision.models.detection.FasterRCNN_ResNet50_FPN_Weights.DEFAULT
    )
    model.eval() # Important: Set to evaluation mode!
    return model

```

### Step 2: The COCO Labels

The model outputs integers (1, 2, 3...). We need a list to map these to human-readable names.

* **Note:** Index 0 is always reserved for the `__background__` class.

```python
COCO_INSTANCE_CATEGORY_NAMES = [
    '__background__', 'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus',
    'train', 'truck', 'boat', 'traffic light', 'fire hydrant', 'N/A', 'stop sign',
    # ... (full list of 91 classes)
]

```

### Step 3: Inference (Prediction)

The inference logic involves transforming the image, passing it to the model, and extracting the raw tensors.

```python
import torchvision.transforms as T
from PIL import Image

def get_prediction(model, img_path, threshold):
    # 1. Load and Transform
    img = Image.open(img_path).convert('RGB') 
    transform = T.Compose([T.ToTensor()]) # Convert to Tensor (0-1 range)
    img_tensor = transform(img)

    # 2. Forward Pass
    with torch.no_grad():
        pred = model([img_tensor]) # Pass as a list of images

    # 3. Extract Results
    pred_scores = pred[0]['scores'].detach().cpu().numpy()
    pred_boxes = pred[0]['boxes'].detach().cpu().numpy()
    pred_labels = pred[0]['labels'].detach().cpu().numpy()

    # 4. Filter by Confidence Threshold (e.g., 0.8)
    valid_indices = pred_scores >= threshold
    
    return pred_boxes[valid_indices], pred_labels[valid_indices]

```

### Step 4: Visualization

We use OpenCV (`cv2`) to draw the bounding boxes. A key challenge is ensuring the boxes and text look good on images of any size (Responsive Design).

```python
import cv2
import matplotlib.pyplot as plt

def draw_boxes(img_path, boxes, labels):
    img = cv2.imread(img_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) # Fix color space (BGR -> RGB)

    # Dynamic Line Thickness based on image size
    # This prevents lines from being too thin on 4K images or too thick on thumbnails
    rect_th = max(round(sum(img.shape) / 2 * 0.003), 2)

    for i, box in enumerate(boxes):
        # Draw Box
        cv2.rectangle(img, (int(box[0]), int(box[1])), (int(box[2]), int(box[3])), 
                      color=(0, 255, 0), thickness=rect_th)
        
        # Draw Text Label
        cv2.putText(img, labels[i], (int(box[0]), int(box[1]-10)), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

    plt.imshow(img)
    plt.show()

```

---

## 4. Key Takeaways vs. Semantic Segmentation

| Feature | Semantic Segmentation (Previous Module) | Object Detection (This Module) |
| --- | --- | --- |
| **Goal** | Classify every pixel. | Classify and localize objects. |
| **Output** | A colored map (Image). | List of coordinates `[x1, y1, x2, y2]`. |
| **Separation** | Cannot distinguish two overlapping cars (they merge into one blob). | Can distinguish instances ("Car 1" box vs "Car 2" box). |
| **Math** | `argmax` on pixel vectors. | Regression on coordinates. |