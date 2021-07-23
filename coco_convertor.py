import glob
import fiftyone as fo
import pandas as pd
import numpy as np
from sklearn import preprocessing

from parse_csv import parse_RSNA_csv

images_patt = "/home/hossein/Desktop/dataset/RSNA_COMPETE/train_png"
train_image_level_path = "/home/hossein/Desktop/dataset/RSNA_COMPETE/train_image_level.csv"  # contains bboxes
train_study_level_path = "/home/hossein/Desktop/dataset/RSNA_COMPETE/train_study_level.csv"
eval_percent = 0.1

# Ex: your custom label format
# annotations = {
#     "/path/to/images/000001.jpg": [
#         {"bbox": ..., "label": ...},
#         ...
#     ],
#     ...
# }

train_image_df, eval_image_df = parse_RSNA_csv(train_image_level_path=train_image_level_path,
                                               train_study_level_path=train_study_level_path, eval_percent=eval_percent)

# Create dataset
dataset = fo.Dataset(name="coco_train_dataset")

# Persist the dataset on disk in order to
# be able to load it in one line in the future
dataset.persistent = True

# Add your samples to the dataset
for filepath in glob.glob(images_patt):
    sample = fo.Sample(filepath=filepath)

    # Convert detections to FiftyOne format
    detections = []
    for obj in annotations[filepath]:
        label = obj["label"]

        # Bounding box coordinates should be relative values
        # in [0, 1] in the following format:
        # [top-left-x, top-left-y, width, height]
        bounding_box = obj["bbox"]

        detections.append(
            fo.Detection(label=label, bounding_box=bounding_box)
        )

    # Store detections in a field name of your choice
    sample["ground_truth"] = fo.Detections(detections=detections)

    dataset.add_sample(sample)
