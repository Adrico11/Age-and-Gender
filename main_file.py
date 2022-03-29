from src.models.model_builder import ModelBuilder
from src.models.model_trainer import ModelTrainer

from src.utils.util_funcs import make_plots

from src.data_process.data_generator import DataGenerator, preprocess_image
from src.data_process.data_extractor import DataExtractor

from keras.models import load_model
from tensorflow.keras.utils import to_categorical
import numpy as np
import cv2

# # importing sys
# import sys
# # adding Folder_2 to the system path
# sys.path.insert(0, 'C:/Users/adrie/OneDrive/Bureau/Stage/Datakalab
# /age-gender-estimation-salem-sermanet/src/data_process')

# data_folder_path = "data/UTKFace/"


def make_prediction(
        loaded_model, pred_sample, pred_folder_name, im_size=(198, 198)):

    pred_row = pred_sample.iloc[0]
    age_true = pred_row['ages']
    gender_true = pred_row['genders']
    gender_true = "Male" if gender_true == 0 else "Female"
    img_path = pred_folder_name+pred_row['image_files']
    image_input = np.array([preprocess_image(img_path, im_size)])

    print("[INFO] Making predictions...")
    age_pred, gender_pred = loaded_model.predict(image_input)
    return age_pred, age_true, gender_pred, gender_true, img_path


def prediction_post_procecss(age_pred, max_age, gender_pred):
    # Age post_process
    age_pred = max(int(age_pred*max_age), 0)  # avoid having negative ages...

    # Gender post_process
    gender_pred = [list(x).index(max(x)) for x in gender_pred]
    gender_pred = "Male" if int(gender_pred[0]) == 0 else "Female"

    return age_pred, gender_pred


def predict_preprocess(img_path, age_true, gender_true, max_age):
    images, ages, genders = [], [], []
    im = preprocess_image(img_path)

    ages.append(age_true / max_age)
    genders.append(to_categorical(gender_true, 2))
    images.append(im)

    images = np.array(images)
    ages = np.array(ages)
    genders = np.array(genders)

    return images, ages, genders


# def create_data_lists(df, folder_name, max_age):
#     images, ages, genders = [], [], []
#     # ######################### Problem with max_age
#     # (not the same according to the dataset...)
#     # max_age = 116  # df.ages.max()

#     for _, row in df.iterrows():

#         age = row['ages']
#         gender = row['genders']
#         img_path = folder_name+row['image_files']
#         print(img_path)
#         im = preprocess_image(img_path)

#         ages.append(age / max_age)
#         genders.append(to_categorical(gender, 2))
#         images.append(im)

#     images = np.array(images)
#     ages = np.array(ages)
#     genders = np.array(genders)

#     return images, ages, genders


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

    # predictions = model_trainer.test_model()
    # age_pred, gender_pred = post_process(predictions)

    if plot:
        history = model_trainer.history
        make_plots(history)


def run_prediction(pred_folder_name, model_file_name):  # nb_pred):
    # ################ ONLY WORKS FOR 1 SINGLE PREDICTION ################

    # max age in the original dataset on which the model was trained on
    max_age = 116

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
    full_pred_dataset = data_extractor.full_dataset
    pred_sample = full_pred_dataset.sample(1)

    age_pred, age_true, gender_pred, gender_true, img_path = make_prediction(
        loaded_model, pred_sample, pred_folder_name, (198, 198))

    age_pred_post, gender_pred_post = prediction_post_procecss(
        age_pred, max_age, gender_pred)

    print(
        f"Predicted age: {age_pred_post} (Actual age: {age_true})")
    print(
        f"Predicted gender: {gender_pred_post} (Actual gender: {gender_true})")

    # First approach (press Ctrl+C to quit)
    # We could also save an image with the predicted/true values on it
    # (use PIL or Tkinter)
    img = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)
    cv2.imshow("Prediction image", img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
