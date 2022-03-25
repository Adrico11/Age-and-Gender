from src.models.model_builder import ModelBuilder
from src.models.model_trainer import ModelTrainer

from src.utils.util_funcs import make_plots

from src.data_process.data_generator import DataGenerator, preprocess_image
from src.data_process.data_extractor import DataExtractor

from keras.models import load_model
from tensorflow.keras.utils import to_categorical
import numpy as np

# # importing sys
# import sys
# # adding Folder_2 to the system path
# sys.path.insert(0, 'C:/Users/adrie/OneDrive/Bureau/Stage/Datakalab
# /age-gender-estimation-salem-sermanet/src/data_process')

# data_folder_path = "data/UTKFace/"


def create_data_lists(df, folder_name):
    images, ages, genders = [], [], []
    # ######################### Problem with max_age
    # (not the same according to the dataset...)
    max_age = 116  # df.ages.max()

    for _, row in df.iterrows():

        age = row['ages']
        gender = row['genders']
        img_path = folder_name+row['image_files']

        im = preprocess_image(img_path)

        ages.append(age / max_age)
        genders.append(to_categorical(gender, 2))
        images.append(im)

    images = np.array(images)
    ages = np.array(ages)
    genders = np.array(genders)

    return images, ages, genders


# add a case to test the model ????????????????

def run_training(train_folder_name, model_file, nb_epochs, plot):

    print("[INFO] Extracting training data...")
    data_extractor = DataExtractor(train_folder_name)
    data_extractor.create_df()
    full_dataset = data_extractor.full_dataset
    print(full_dataset.head())

    print("[INFO] Creating Data generator...")
    data_generator = DataGenerator(
        full_dataset, train_folder_name)
    data_generator.generate_split_indexes()
    train_idx, valid_idx, test_idx = data_generator.idx_list
    # self.idx_list = [train_idx, valid_idx, test_idx]

    print("[INFO] Building model...")
    model_builder = ModelBuilder()
    model_builder.assemble_full_model()
    face_model = model_builder.model

    print("[INFO] Training model...")
    model_trainer = ModelTrainer(
        face_model, data_generator, train_idx,
        valid_idx, model_file, nb_epochs)
    model_trainer.compile_model()
    model_trainer.train_model()
    print("[INFO] Model trained !")

    if plot:
        history = model_trainer.history
        make_plots(history)


def run_prediction(pred_folder_name, model_file_name, nb_pred):

    try:
        print("[INFO] Looking for trained model...")
        loaded_model = load_model(model_file_name)
    except FileNotFoundError:
        print("[ERROR] No previously trained model...")
        print("[ERROR] Train a new model first !")
        # Exit()

    print("[INFO] Found a trained model...")
    print("[INFO] Extracing prediction data...")
    data_extractor = DataExtractor(pred_folder_name)
    data_extractor.create_df()
    full_dataset = data_extractor.full_dataset
    pred_dataset = full_dataset.sample(min(len(full_dataset.index), nb_pred))
    # pred_files = pred_dataset["image_files"]
    # x_pred = []

    images, ages, genders = create_data_lists(pred_dataset, pred_folder_name)

    print("[INFO] Making predictions...")
    age_pred, gender_pred = loaded_model.predict(images)
    # weird ages pred sometimes negatives...
    print(age_pred, ages)
    gender_pred = [list(x).index(max(x)) for x in gender_pred]

    # reverse from create_data_list func
    genders = [list(x).index(max(x)) for x in genders]
    print(gender_pred, genders)

# if __name__ == '__main__':
#     main("data/UTKFace/", "age_gender_model.h5", 5)
