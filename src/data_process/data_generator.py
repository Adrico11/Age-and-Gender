from tensorflow.keras.utils import to_categorical
from PIL import Image
import numpy as np


def preprocess_image(img_path, im_size=(64, 64)):
    """
    Used to perform some minor preprocessing on the image
    before inputting into the network.
    """
    im = Image.open(img_path)
    im = im.resize(im_size)
    im = np.array(im) / 255.0

    return im


class DataGenerator():
    """
    Data generator for the UTKFace dataset.
    This class should be used when training our Keras multi-output model.
    """
    def __init__(
        self, df, data_folder_path,
            im_size=(64, 64), TRAIN_TEST_SPLIT=0.8):  # (198,198)

        self.df = df
        self.data_folder_path = data_folder_path
        self.im_size = im_size
        self.TRAIN_TEST_SPLIT = TRAIN_TEST_SPLIT
        self.max_age = self.df['ages'].max()
        self.idx_list = None

    def generate_split_indexes(self):
        "Return lists of train/valid/test idx"
        shuffled_idx = np.random.permutation(len(self.df))
        max_train_valid_idx = int(len(self.df) * self.TRAIN_TEST_SPLIT)
        max_train_idx = int(max_train_valid_idx * self.TRAIN_TEST_SPLIT)
        train_idx = shuffled_idx[:max_train_idx]
        valid_idx = shuffled_idx[max_train_idx:max_train_valid_idx]
        test_idx = shuffled_idx[max_train_valid_idx:]

        self.idx_list = [train_idx, valid_idx, test_idx]

    def generate_images(self, image_idx, is_training, batch_size=16):
        """
        Used to generate a batch with images when training/testing/validating
        our Keras model.
        """

        # arrays to store our batched data
        images, ages, genders = [], [], []
        while True:
            for idx in image_idx:
                person = self.df.iloc[idx]

                age = person['ages']
                gender = person['genders']
                img_path = self.data_folder_path+person['image_files']

                im = preprocess_image(img_path, self.im_size)

                ages.append(age / self.max_age)
                genders.append(to_categorical(gender, 2))
                images.append(im)

                # yielding condition
                if len(images) >= batch_size:
                    yield np.array(images), [np.array(ages), np.array(genders)]
                    images, ages, genders = [], [], []

            if not is_training:
                break

    def generate_test_images(self, image_idx, is_training, batch_size=16):
        pass


if __name__ == '__main__':
    data_folder_path = "data/UTKFace/"

    import pandas as pd
    df = pd.DataFrame()  # empty df...

    data_generator = DataGenerator(df)
    data_generator.generate_split_indexes()
    print(len(data_generator.idx_list[0]))
