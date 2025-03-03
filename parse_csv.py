import pandas as pd
import numpy as np
from PIL import Image
import glob
import os
import fiftyone as fo
from sklearn import preprocessing
from alive_progress import alive_bar


class Parser:

    def __init__(self, img_pth, load, load_path=None):
        self.img_pth = img_pth
        self.load = load
        self.load_path = load_path

    def add_remainder(self, df):
        added_rows = []
        for index, row in df.iterrows():
            # print(row)
            metadata = row["label"].split("opacity")[1:]
            if (len(metadata) >= 2):
                for element in metadata[1:]:
                    new_row = row.copy()
                    new_row["x_min"] = element.split(" ")[2]
                    new_row["y_min"] = element.split(" ")[3]
                    new_row["x_max"] = element.split(" ")[4]
                    new_row["y_max"] = element.split(" ")[5]
                    added_rows.append(new_row)
        added_rows_df = pd.DataFrame(added_rows, columns=list(df.columns))
        df = pd.concat([df, added_rows_df], ignore_index=True)
        return df

    def set_image_size(self, df):
        for root, dirs, files in os.walk(self.img_pth):
            for file in files:
                # append the file name to the list
                width, height = Image.open(os.path.join(root, file)).size
                image_name = str(file.split("_")[1][:-4]) + "_image"
                df.loc[df['id'] == image_name, ["width", "height"]] = [width, height]

    def parse_RSNA_csv(self, train_image_level_path, train_study_level_path, eval_percent):
        if (self.load is False):
            train_study_df = pd.read_csv(train_study_level_path)
            train_study_df = train_study_df.rename(
                columns={'Negative for Pneumonia': 'negative', 'Typical Appearance': 'typical',
                         'Indeterminate Appearance': 'indeterminate',
                         'Atypical Appearance': 'atypical'}, inplace=False)
            train_study_df['label_y'] = 'negative'
            train_study_df.loc[train_study_df['typical'] == 1, 'label_y'] = 'typical'
            train_study_df.loc[train_study_df['indeterminate'] == 1, 'label_y'] = 'indeterminate'
            train_study_df.loc[train_study_df['atypical'] == 1, 'label_y'] = 'atypical'
            train_study_df['StudyInstanceUID'] = train_study_df['id'].apply(lambda x: x.replace('_study', ''))
            del train_study_df['id']
            # print(train_study_df.head(6))

            train_image_df = pd.read_csv(train_image_level_path)
            train_image_df['x_min'] = train_image_df.label.apply(lambda x: float(x.split()[2]))
            train_image_df['y_min'] = train_image_df.label.apply(lambda x: float(x.split()[3]))
            train_image_df['x_max'] = train_image_df.label.apply(lambda x: float(x.split()[4]))
            train_image_df['y_max'] = train_image_df.label.apply(lambda x: float(x.split()[5]))
            # print(train_image_df.head(3).T)

            # setting the width and height of the image
            train_image_df["width"] = 0
            train_image_df["height"] = 0
            self.set_image_size(train_image_df)

            train_image_df = train_image_df.merge(train_study_df, on='StudyInstanceUID')
            # print(train_image_df.head(6).T)
            train_image_df['integer_label'] = train_image_df['label_y']
            le = preprocessing.LabelEncoder()
            le.fit(train_image_df['integer_label'])
            train_image_df['integer_label'] = le.transform(train_image_df['integer_label'])
            print("list classes integer_labels: {}".format(le.classes_))
            train_image_df = train_image_df.sample(frac=1).reset_index(drop=True)
            print(train_image_df.head(10).T)

            # saving dataframe:
            train_image_df.to_csv(self.load_path)
            print("saved csv file")
        else:
            train_image_df = pd.read_csv(self.load_path)

        number_eval = int(len(train_image_df.index) * eval_percent)
        # eval_image_df = train_image_df.iloc[:number_eval]
        # train_image_df = train_image_df.iloc[number_eval:]
        # print("len trainset: {}".format(train_image_df.shape[0]))
        # print("len evalset: {}".format(eval_image_df.shape[0]))
        train_image_df = self.add_remainder(train_image_df)
        # eval_image_df = self.add_remainder(eval_image_df)
        return train_image_df #, eval_image_df

    def coco_covert(self, df, name="RSNA_train_dataset"):
        try:
            dataset = fo.Dataset(name=name)
        except Exception as e:
            print("ERROR : " + str(e) + "\nfixing it with loading")
            dataset = fo.load_dataset(name=name)
            dataset.delete()
            return self.coco_covert(df,name=name)

        # Persist the dataset on disk in order to
        # be able to load it in one line in the future
        dataset.persistent = True

        # Add your samples to the dataset
        with alive_bar(1000) as bar:
            for index, row in df.iterrows():
                bar()
                filepath = os.path.join(self.img_pth, row["StudyInstanceUID"] + "_" + row["id"][:-6] + ".png")
                if (os.path.isfile(filepath) is False):
                    continue
                sample = fo.Sample(filepath=filepath)

                # Convert detections to FiftyOne format
                detections = []
                label = row["label_y"]

                # Bounding box coordinates should be relative values
                # in [0, 1] in the following format:
                # [top-left-x, top-left-y, width, height]
                bounding_box = self.compute_bounding_box(row["x_min"], row["y_min"], row["x_max"], row["y_max"],
                                                         row["width"], row["height"])

                detections.append(
                    fo.Detection(label=label, bounding_box=bounding_box)
                )

                # Store detections in a field name of your choice
                sample["ground_truth"] = fo.Detections(detections=detections)

                dataset.add_sample(sample)
        return dataset

    def compute_bounding_box(self, x_min, y_min, x_max, y_max, img_width, img_height):
        x_min,x_max=float(x_min),float(x_max)
        y_min,y_max=float(y_min),float(y_max)
        img_height,img_height=float(img_width),float(img_height)
        top_left_x = x_min / img_width
        top_left_y = y_max / img_height
        width = (x_max - x_min) / img_width
        height = (y_max - y_min) / img_height
        return [top_left_x, top_left_y, width, height]
