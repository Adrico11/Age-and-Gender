import numpy as np
import pandas as pd
import cv2
from sklearn.model_selection import train_test_split
from typing import List


class DataPreprocessor:

    def __init__(
            self, df_init, data_folder_path, test_size,
            img_size=(64, 64)) -> None:
        self.df_init = df_init
        self.data_folder_path = data_folder_path
        self.test_size = test_size
        self.img_size = img_size
        self.features = None
        self.target = None
        self.age_data = None
        self.gender_data = None

    def preprocess_data(self) -> np.array:

        target_list = []
        features_list = []

        for _, row in self.df_init.iterrows():
            image_id = row.image_id
            # cv2.IMREAD_UNCHANGED
            image = cv2.imread(
                self.data_folder_path+image_id, cv2.IMREAD_GRAYSCALE)
            image = cv2.resize(image, dsize=self.img_size)
            image = image.reshape((image.shape[0], image.shape[1], 1))  # 3
            # No need to divide by 255 since we do it in the datagenerator...
            features_list.append(image)
            age = row.ages
            gender = row.genders
            target_list.append([age, gender])

        self.features = np.array(features_list)
        self.target = np.array(target_list)

    def data_split(self, random_state=42) -> List:
        x_train, x_test, y_train, y_test = train_test_split(
            self.features, self.target, test_size=self.test_size,
            random_state=random_state)  # shuffle  = True ??????????????????
        y_train_age, y_train_gender = y_train[:, 0], y_train[:, 1]
        y_test_age, y_test_gender = y_test[:, 0], y_test[:, 1]

        # better as a dictionary ?????????????????
        self.age_data = [x_train, x_test, y_train_age, y_test_age]
        self.gender_data = [x_train, x_test, y_train_gender, y_test_gender]


if __name__ == '__main__':
    df_init = pd.DataFrame()  # empty dataframe...
    data_folder_path = "UTKFace/"
    test_size = 0.2
    data_preprocessor = DataPreprocessor(df_init, data_folder_path, test_size)
    data_preprocessor.preprocess_data()
    data_preprocessor.data_split()
