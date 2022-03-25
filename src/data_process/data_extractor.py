import pandas as pd
# import seaborn as sns
# import matplotlib.pyplot as plt
import os
from typing import List


class DataExtractor:

    def __init__(self, data_folder_path, max_count=500) -> None:
        self.data_folder_path = data_folder_path
        self.full_dataset = None
        self.max_count = max_count
        self.balanced_df = None

    def fetch_data(self) -> List:
        files = os.listdir(self.data_folder_path)
        ages = []
        genders = []
        for file in files:
            split_var = file.split('_')
            ages.append(int(split_var[0]))
            genders.append(int(split_var[1]))
        return files, ages, genders

    def create_df(self) -> None:
        files, ages, genders = self.fetch_data()
        self.full_dataset = pd.DataFrame(
            {"image_files": files, "ages": ages, "genders": genders})

    # def balance_dataset(self) -> None:
    #     ages_counts = self.full_dataset.ages.value_counts()
    #     # list of ages with more than max_count corresponding images
    #     ages_list = ages_counts[ages_counts > self.max_count].index.tolist()
    #     # create a new df without those ages
    #     self.balanced_df = self.full_dataset[
    #         ~self.full_dataset["ages"].isin(ages_list)].copy()

    #     for age in ages_list:
    #         age_df = self.full_dataset[
    #             self.full_dataset.ages == age].sample(self.max_count)
    #         # add the max_count files to the new df of the age in question
    #         self.balanced_df = pd.concat(
    #             [self.balanced_df, age_df], ignore_index=True, sort=False)
    #     self.balanced_df.sample(frac=1)  # shuffle the new df
    #     if self.balanced_df.ages.value_counts().max() > self.max_count:
    #         print("Dataset is still imbalanced...")


if __name__ == '__main__':
    data_folder_path = "data/UTKFace/"
    max_count = 500
    data_fetcher = DataExtractor(data_folder_path, max_count)
    data_fetcher.create_df()
    # print(data_fetcher.full_dataset.head())
    # print(data_fetcher.full_dataset.shape)

    # data_fetcher.balance_dataset()

    # print(data_fetcher.balanced_df.head())
    # print(data_fetcher.balanced_df.shape)
    # print(data_fetcher.balanced_df.ages.value_counts().max())
    # sns.set_theme()
    # sns.displot(data_fetcher.balanced_df['ages'], kde=True, bins=30)
    # plt.show()
