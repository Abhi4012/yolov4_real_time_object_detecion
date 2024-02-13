import pdf_process
import cv2
import numpy as np
import os
import matplotlib.pyplot as plt
import mplcursors

# Define paths
path = 'yolov4_darknet/cfg'
labels_path = os.path.sep.join(['yolov4_darknet/cfg', 'coco.names'])
weights_path = os.path.sep.join(['yolov4_darknet', 'yolov4.weights'])
config_path = os.path.sep.join(['yolov4_darknet/cfg', 'yolov4.cfg'])

# Read labels
LABELS = open(labels_path).read().strip().split('\n')

# Load YOLOv4 model
net = cv2.dnn.readNet(config_path, weights_path)

# Generating random colours for bounding boxes
COLORS = np.random.randint(0, 255, size=(len(LABELS), 3), dtype='uint16')

# Get layer names and output layers
ln = net.getLayerNames()
ln = [ln[i - 1] for i in net.getUnconnectedOutLayers()]

# display an image using matplotlib
def show_image(img):
    fig = plt.gcf()
    fig.set_size_inches(30, 15)
    plt.axis('off')
    plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    plt.show()

# Preprocessing images for our model
def preprocess_image(image_path):
    image = cv2.imread(image_path)
    if image is None:
        raise FileNotFoundError(f"Image not found at path: {image_path}")
    (H, W) = image.shape[:2]
    blob = cv2.dnn.blobFromImage(image, 1 / 255.0, (416, 416), swapRB=True, crop=False)
    net.setInput(blob)
    return image, (H, W)

# perform YOLO detection
def perform_yolo(image, ln, W, H, threshold=0.5, threshold_NMS=0.3):
    layer_outputs = net.forward(ln)

    # Initialize lists to store detected objects
    boxes = []
    confidences = []
    class_ids = []

    for output in layer_outputs:
        for detection in output:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]

            if confidence > threshold:
                box = detection[0:4] * np.array([W, H, W, H])
                (centerX, centerY, width, height) = box.astype('int')

                x = int(centerX - (width / 2))
                y = int(centerY - (height / 2))

                boxes.append([x, y, int(width), int(height)])
                confidences.append(float(confidence))
                class_ids.append(class_id)

    # Apply non-maximum suppression
    indices = cv2.dnn.NMSBoxes(boxes, confidences, threshold, threshold_NMS)

    return boxes, confidences, class_ids, indices

# display the detected objects with real-time hovering
def display_detected_objects(image, boxes, confidences, class_ids, indices):
    fig, ax = plt.subplots()
    ax.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

    object_info = []  # Accumulate information about all detected objects

    if len(indices) > 0:
        for i in indices.flatten():
            (x, y) = (boxes[i][0], boxes[i][1])
            (w, h) = (boxes[i][2], boxes[i][3])

            # Get the detected object region without resizing
            detected_object = image[y:y + h, x:x + w]

            # Create a rectangle patch without resizing
            color = COLORS[class_ids[i]] / 255.0
            rect = plt.Rectangle((x, y), w, h, linewidth=2, edgecolor=color, facecolor='none')
            ax.add_patch(rect)

            # Display the class name and confidence
            text = "{}: {:.4f}".format(LABELS[class_ids[i]], confidences[i])
            ax.text(x, y, text, color=color, fontsize=8, verticalalignment='top')

            # Accumulate object information
            object_info.append(text)

    ax.axis('off')

    # Use mplcursors to display details on hover for all detected objects
    cursor = mplcursors.cursor(hover=True)

    def on_hover(sel):
        sel.annotation.set_text('\n'.join(object_info))
    
    cursor.connect("add", on_hover)

    plt.show()

# Execute the YOLO detection
def detection_yolo(image_path):
    try:
        image, (H, W) = preprocess_image(image_path)
        boxes, confidences, class_ids, indices = perform_yolo(image, ln, W, H)
        display_detected_objects(image, boxes, confidences, class_ids, indices)
    except Exception as e:
        print(f"Error: {e}")

# detection for an image
detection_yolo("images/image_4.jpg")
