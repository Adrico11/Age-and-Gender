from extract_data import DataFetcher
from preprocess_data import DataPreprocessor

data_folder_path = "data/UTKFace/"
max_count = 500
test_size = 0.2


def main_preprocess():

    print("Fetching dataset")
    data_fetcher = DataFetcher(data_folder_path, max_count)
    data_fetcher.create_df()
    print("Balancing dataset")
    data_fetcher.balance_dataset()

    print("Preprocessing dataset")
    data_preprocessor = DataPreprocessor(
        data_fetcher.balanced_df, data_folder_path, test_size)
    data_preprocessor.preprocess_data()
    print(data_preprocessor.features.shape)
    print(data_preprocessor.target.shape)
    print("Spliting dataset into train/test sets")
    data_preprocessor.data_split()
    print(data_preprocessor.age_data[0].shape)
    print(data_preprocessor.gender_data[0].shape)


if __name__ == '__main__':
    main_preprocess()
