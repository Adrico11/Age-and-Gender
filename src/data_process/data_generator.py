from tensorflow.keras.utils import to_categorical
from PIL import Image
import numpy as np


class UtkFaceDataGenerator():
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

    def generate_split_indexes(self):
        "Return lists of train/valid/test idx"
        shuffled_idx = np.random.permutation(len(self.df))
        max_train_valid_idx = int(len(self.df) * self.TRAIN_TEST_SPLIT)
        max_train_idx = int(max_train_valid_idx * self.TRAIN_TEST_SPLIT)
        train_idx = shuffled_idx[:max_train_idx]
        valid_idx = shuffled_idx[max_train_idx:max_train_valid_idx]
        test_idx = shuffled_idx[max_train_valid_idx:]

        return train_idx, valid_idx, test_idx

    def preprocess_image(self, img_path):
        """
        Used to perform some minor preprocessing on the image
        before inputting into the network.
        """
        im = Image.open(img_path)
        im = im.resize(self.im_size)
        im = np.array(im) / 255.0

        return im

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
                file = self.data_folder_path+person['image_files']

                im = self.preprocess_image(file)

                ages.append(age / self.max_age)
                genders.append(to_categorical(gender, 2))
                images.append(im)

                # yielding condition
                if len(images) >= batch_size:
                    yield np.array(images), [np.array(ages), np.array(genders)]
                    images, ages, genders = [], [], []

            if not is_training:
                break


if __name__ == '__main__':
    data_folder_path = "data/UTKFace/"

    import pandas as pd
    df = pd.DataFrame()  # empty df...

    data_generator = UtkFaceDataGenerator(df)
    train_idx, valid_idx, test_idx = data_generator.generate_split_indexes()
    print(len(train_idx))