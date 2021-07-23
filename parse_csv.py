import pandas as pd
import numpy as np
from sklearn import preprocessing


def add_remainder(df):
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


def parse_RSNA_csv(train_image_level_path, train_study_level_path, eval_percent):
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
    train_image_df = train_image_df.merge(train_study_df, on='StudyInstanceUID')
    # print(train_image_df.head(6).T)
    train_image_df['integer_label'] = train_image_df['label_y']
    le = preprocessing.LabelEncoder()
    le.fit(train_image_df['integer_label'])
    train_image_df['integer_label'] = le.transform(train_image_df['integer_label'])
    print("list classes integer_labels: {}".format(le.classes_))
    train_image_df = train_image_df.sample(frac=1).reset_index(drop=True)
    print(train_image_df.head(10).T)
    number_eval = int(len(train_image_df.index) * eval_percent)
    eval_image_df = train_image_df.iloc[:number_eval]
    train_image_df = train_image_df.iloc[number_eval:]
    print("len trainset: {}".format(train_image_df.shape[0]))
    print("len evalset: {}".format(eval_image_df.shape[0]))
    train_image_df = add_remainder(train_image_df)
    eval_image_df = add_remainder(eval_image_df)
    return train_image_df, eval_image_df
