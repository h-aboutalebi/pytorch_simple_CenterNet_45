import glob
import fiftyone as fo
import os
import pandas as pd
import numpy as np
from sklearn import preprocessing

from parse_csv import Parser

csv_output = "/home/hossein/Desktop/dataset/RSNA_COMPETE/hossein_train.csv"
export_output = "/home/hossein/Desktop/dataset/RSNA_COMPETE/coco"
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

parser = Parser(img_pth=images_patt, load=True, load_path=csv_output)
train_image_df= parser.parse_RSNA_csv(train_image_level_path=train_image_level_path,
                                                      train_study_level_path=train_study_level_path,
                                                      eval_percent=eval_percent)

eval_dataset=parser.coco_covert(train_image_df,name="RSNA_train_dataset")
label_field = "ground_truth"  # for example

# Export the dataset
eval_dataset.export(
    export_dir=export_output,
    dataset_type=fo.types.COCODetectionDataset,
    label_field=label_field,
)