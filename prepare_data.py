import os

import cv2

# Read input directories structure.
DATA_DIR = os.path.join(os.path.dirname(__file__), 'ImageSegmentation')
IMAGES_DIR = os.path.join(DATA_DIR, 'Original')
LABELS_COPA_DIR = os.path.join(DATA_DIR, 'Copa')
LABELS_DISCO_DIR = os.path.join(DATA_DIR, 'Disco')

# Get images and labels.
images = sorted(os.listdir(IMAGES_DIR))
labels_copa = sorted(os.listdir(LABELS_COPA_DIR))
labels_disco = sorted(os.listdir(LABELS_DISCO_DIR))
n_images = len(images)
print("Number of images: {}".format(n_images))

# Filter labels if they have bordes in their name.
labels_copa = [label for label in labels_copa if 'bordes' not in label]
labels_disco = [label for label in labels_disco if 'bordes' not in label]

# Create output directories structure.
folders = ["images/train", "labels/train", "images/val", "labels/val"]
for folder in folders:
    dir = os.path.join(DATA_DIR, "yolo", folder)
    if not os.path.exists(dir):
        os.makedirs(dir)

# Iterate through images and labels.
for i, image_path in enumerate(images):

    # Extract image id.
    image_id = image_path.split('/')[-1].split('.')[0].split('_')[0]
    train_or_val = "train" if i < int(0.8 * n_images) else "val"
    image_size = cv2.imread(os.path.join(IMAGES_DIR, image_path)).shape
    h, w = image_size[0], image_size[1]

    # Find corresponding labels.
    label_copa = [label for label in labels_copa if image_id in label][0]
    label_disco = [label for label in labels_disco if image_id in label][0]

    # Find contours in labels.
    label_copa = cv2.imread(os.path.join(LABELS_COPA_DIR, label_copa))
    label_copa = cv2.cvtColor(label_copa, cv2.COLOR_BGR2GRAY)
    contours_copa, _ = cv2.findContours(label_copa, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    label_disco = cv2.imread(os.path.join(LABELS_DISCO_DIR, label_disco))
    label_disco = cv2.cvtColor(label_disco, cv2.COLOR_BGR2GRAY)
    contours_disco, _ = cv2.findContours(label_disco, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    labels = []
    for contour in contours_copa:
        string = "0 "
        for point in contour:
            x, y = point[0]
            x = x / w
            y = y / h
            string += f" {x:.3f} {y:.3f}"
        labels.append(string)

    for contour in contours_disco:
        string = "1 "
        for point in contour:
            x, y = point[0]
            x = x / w
            y = y / h
            string += f" {x:.3f} {y:.3f}"
        labels.append(string)

    # Write image and labels.
    with open(os.path.join(DATA_DIR, "yolo", f"labels/{train_or_val}/{image_id}.txt"), 'w') as f:
        for label in labels:
            f.write(label + '\n')

    cv2.imwrite(os.path.join(DATA_DIR, "yolo", f"images/{train_or_val}/{image_id}.png"),
                cv2.imread(os.path.join(IMAGES_DIR, image_path)))



