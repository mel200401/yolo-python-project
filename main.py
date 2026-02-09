import os
import shutil
from pathlib import Path
import pandas as pd
from sklearn.model_selection import train_test_split

# Set Up Data Paths

dataset_directory = Path("data")
train_img_directory = dataset_directory / "training_images"
test_img_directory = dataset_directory / "testing_images"
labels_directory = dataset_directory / "labels"
labels_directory.mkdir(exist_ok=True)

csv_path = dataset_directory / "train_solution_bounding_boxes (1).csv"

# Image Sizes
img_width = 676
img_height = 380

class_id = 0  # single class: car, pictures without car do not have annotation (need to create empty files for label)

# create empty label files for images without cars
for folder in [train_img_directory, test_img_directory]:
    for img_path in folder.iterdir():
        if img_path.suffix.lower() == ".jpg":
            (labels_directory / f"{img_path.stem}.txt").touch()

# have to convert the file (which has similar structure to PASCAL VOC annotation) to YOLO readable file
df = pd.read_csv(csv_path)

for _, row in df.iterrows():
    img_name = row["image"]
    xmin, ymin, xmax, ymax = row["xmin"], row["ymin"], row["xmax"], row["ymax"]

    # Convert to YOLO normalized format
    x_center = ((xmin + xmax) / 2) / img_width
    y_center = ((ymin + ymax) / 2) / img_height
    width = (xmax - xmin) / img_width
    height = (ymax - ymin) / img_height

    label_path = labels_directory / f"{Path(img_name).stem}.txt"

    with label_path.open("a") as f:
        f.write(f"{class_id} {x_center} {y_center} {width} {height}\n")

print("YOLO label files created.")
