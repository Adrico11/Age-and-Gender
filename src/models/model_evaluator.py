from PIL import Image
import numpy as np

from sklearn.metrics import classification_report, ConfusionMatrixDisplay
from sklearn.metrics import mean_squared_error, mean_absolute_error
import matplotlib.pyplot as plt


def preprocess_image(img_path, im_size=(64, 64)):
    """
    Used to perform some minor preprocessing on the image
    before inputting into the network.
    """
    im = Image.open(img_path)
    im = im.resize(im_size)
    im = np.array(im) / 255.0

    return im


class ModelEvaluator:

    # (198, 198)
    def __init__(
        self, trained_model, df, max_age,
            test_idx, data_folder_path, im_size=(64, 64)) -> None:
        self.trained_model = trained_model
        # self.data_generator = data_generator
        self.df = df
        self.test_idx = test_idx
        self.max_age = max_age
        self.data_folder_path = data_folder_path
        # self.batch_size = self.valid_batch_size = batch_size
        self.im_size = im_size
        self.input_images = None

    # def test_model(self) -> None:
    #     test_gen = self.data_generator.generate_test_images(
    #         self.test_idx, is_testing=True, batch_size=self.batch_size)
    #     age_pred, gender_pred = self.model.predict(test_gen)
    #     return age_pred, gender_pred

    # def post_process(predictions):
    #     pass

    def generate_test_images(self):
        images_list, age_true_list, gender_true_list = [], [], []
        for idx in self.test_idx:

            person = self.df.iloc[idx]
            age_true = person['ages']
            gender_true = person['genders']
            # gender_true = "Male" if gender_true == 0 else "Female"

            img_path = self.data_folder_path+person['image_files']
            im = preprocess_image(img_path, self.im_size)

            images_list.append(im)
            age_true_list.append(age_true)
            gender_true_list.append(gender_true)

        self.input_images = np.array(images_list)
        self.ages_true = age_true_list
        self.gender_true = gender_true_list

    def make_test_predictions(self):
        predictions = self.trained_model.predict(
            self.input_images)

        ###############
        age_predictions = predictions[0].tolist()
        gender_predictions = predictions[1].tolist()
        ###############

        # print(age_predictions)
        # print(gender_predictions)
        # age_pred_list = [
        #     max(int(x[0]*self.max_age), 0) for x in age_predictions
        #     ]
        age_pred_list = [max(x[0], 0) for x in age_predictions]
        gender_pred_list = [x.index(max(x)) for x in gender_predictions]

        # print(age_pred_list)
        # print(gender_pred_list)

        self.age_pred = age_pred_list
        self.gender_pred = gender_pred_list

    def eval_model(self):

        # age evaluation
        print("[INFO] Evaluating age performance...")
        mse = self.max_age * mean_squared_error(
            self.gender_true, self.gender_pred)
        mae = self.max_age * mean_absolute_error(
            self.gender_true, self.gender_pred)
        print(f"Age mse : {mse} \nAge mae : {mae}")

        # gender evaluation
        print("[INFO] Evaluating gender performance...")
        target_names = ["Male", "Female"]
        report = classification_report(
            self.gender_true, self.gender_pred, target_names=target_names)
        print("Gender classification report : ")
        print(report)
        ConfusionMatrixDisplay.from_predictions(
            self.gender_true, self.gender_pred, display_labels=target_names,
            xticks_rotation="vertical")
        plt.tight_layout()
        plt.savefig("Gender_confusion_matrix.png")

    # def make_prediction(
    #     loaded_model, pred_sample, pred_folder_name, im_size=(198, 198)):
    #     pred_row = pred_sample.iloc[0]
    #     age_true = pred_row['ages']
    #     gender_true = pred_row['genders']
    #     gender_true = "Male" if gender_true == 0 else "Female"
    #     img_path = pred_folder_name+pred_row['image_files']
    #     image_input = np.array([preprocess_image(img_path, im_size)])

    #     print("[INFO] Making predictions...")
    #     age_pred, gender_pred = loaded_model.predict(image_input)
    #     return age_pred, age_true, gender_pred, gender_true, img_path

    # def prediction_post_procecss(age_pred, max_age, gender_pred):
    #     # Age post_process
    #     age_pred = max(int(age_pred*max_age), 0)
    # # avoid having negative ages...

    #     # Gender post_process
    #     gender_pred = [list(x).index(max(x)) for x in gender_pred]
    #     gender_pred = "Male" if int(gender_pred[0]) == 0 else "Female"

    #     return age_pred, gender_pred
