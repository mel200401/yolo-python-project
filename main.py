import shutil
import os
from pathlib import Path
import pandas as pd
from sklearn.model_selection import train_test_split
from ultralytics import YOLO
import torch

# Set Up Data Paths

dataset_directory = Path("data")
train_img_directory = dataset_directory / "training_images"
test_img_directory = dataset_directory / "testing_images"
labels_directory = dataset_directory / "labels"
#labels_directory.mkdir(exist_ok=True)
labels_directory.mkdir(parents=True, exist_ok=True) 

csv_path = dataset_directory / "train_solution_bounding_boxes (1).csv"

print("directories and path are set up")

# Image Sizes
img_width = 676
img_height = 380

class_id = 0  # single class: car, pictures without car do not have annotation (need to create empty files for label)

# create empty label files for images without cars
for folder in [train_img_directory, test_img_directory]:
    for img_path in folder.iterdir():
        if img_path.suffix.lower() == ".jpg":
            (labels_directory / f"{img_path.stem}.txt").touch()

print("empty labels are created")

# have to convert the file (PASCAL VOC annotation) to YOLO readable file
df = pd.read_csv(csv_path)

for _, row in df.iterrows():
    img_name = row["image"]
    xmin, ymin, xmax, ymax = row["xmin"], row["ymin"], row["xmax"], row["ymax"]

    # clipping in case of negative or values outside of the image
    xmin = max(0, xmin)
    ymin = max(0, ymin)
    xmax = min(img_width, xmax)
    ymax = min(img_height, ymax)

    # Convert to YOLO normalized format
    x_center = ((xmin + xmax) / 2) / img_width
    y_center = ((ymin + ymax) / 2) / img_height
    width = (xmax - xmin) / img_width
    height = (ymax - ymin) / img_height

    label_path = labels_directory / f"{Path(img_name).stem}.txt"

    with label_path.open("a") as f:
        f.write(f"{class_id} {x_center} {y_center} {width} {height}\n")

print("YOLO label files created.")

# Splitting into training, evaluate and the test set
for split in ["train", "val"]:
    (dataset_directory / split / "images").mkdir(parents=True, exist_ok=True)
    (dataset_directory / split / "labels").mkdir(parents=True, exist_ok=True)

# List of training images (so we don't use the test images for the split)
train_images = [
    img for img in train_img_directory.iterdir() if img.suffix.lower() == ".jpg"
]

# 80% train, 20% val
train_imgs, val_imgs = train_test_split(train_images, test_size=0.2, random_state=42)

print("Data has been split")

# Copy training images + labels
for img_path in train_imgs:
    shutil.copy(img_path, dataset_directory / "train" / "images" / img_path.name)
    shutil.copy(
        labels_directory / f"{img_path.stem}.txt",
        dataset_directory / "train" / "labels" / f"{img_path.stem}.txt",
    )

# Copy validation images + labels
for img_path in val_imgs:
    shutil.copy(img_path, dataset_directory / "val" / "images" / img_path.name)
    shutil.copy(
        labels_directory / f"{img_path.stem}.txt",
        dataset_directory / "val" / "labels" / f"{img_path.stem}.txt",
    )

# have the test set also in same format in case we want to have some evaluation metrics
(dataset_directory / "test" / "images").mkdir(parents=True, exist_ok=True)
(dataset_directory / "test" / "labels").mkdir(parents=True, exist_ok=True)

# Copy test images + labels
for img_path in test_img_directory.iterdir():
    if img_path.suffix.lower() == ".jpg":
        shutil.copy(img_path, dataset_directory / "test" / "images" / img_path.name)
        shutil.copy(
            labels_directory / f"{img_path.stem}.txt",
            dataset_directory / "test" / "labels" / f"{img_path.stem}.txt",
        )

print("Dataset is ready for YOLO.")

# Create Yaml file for YOLO
yaml_content = f"""
path: {dataset_directory.resolve()}
train: train/images
val: val/images
test: test/images

nc: 1
names: ["car"]
"""

"""
train: {dataset_directory / "train" / "images"}
val: {dataset_directory / "val" / "images"}
test: {dataset_directory / "test" / "images"}
"""


with open(dataset_directory / "data.yaml", "w") as f:
    f.write(yaml_content)

print("data.yaml created.")


# ==========================================
# TRAINING SECTION (Optional)
# Uncomment this section if you want to retrain the model. 
# Estimated ~2 hours on MacBook MPS.
# ==========================================
"""
#import model
model = YOLO('yolov8n.pt')

if torch.backends.mps.is_available():#for ios
    device_type = 'mps'        
elif torch.cuda.is_available(): #for cuda
    device_type = '0'          
else:
    device_type = 'cpu'        

#start training
results = model.train(
    data='data/data.yaml',   
    epochs=15,               
    imgsz=640,               
    batch=8,                #adjust this if you have small VRAM
    name='car_detection', 
    device=device_type   
)

"""
# INFERENCE SECTION
# Make sure the 'best.pt' is in the specified path below.
model_path = 'runs/car_detection/weights/best.pt' 
model = YOLO(model_path)


