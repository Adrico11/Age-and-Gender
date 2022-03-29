import argparse

from main_file import run_training, run_prediction

# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()

ap.add_argument(
    "-r", "--run", choices=['train', 'predict'],
    required=True, help="choose : train model or make prediction"
)

# add the input file path to the images directory
# "data/UTKFace_train/"
ap.add_argument(
    "-td", "--train_dataset", default="predict_data/UTKFace_predict/", 
    help="path to train dataset (i.e., directory of images)")

# add the input file path to the images directory
ap.add_argument(
    "-pd", "--predict_dataset", default="predict_data/UTKFace_predict/", 
    help="path to predict dataset (i.e., directory of images)")

# add the output file path to save the model
ap.add_argument(
    "-m", "--model",  default="trained_models/age_gender_model.h5",
    help="path to output model")

# add the number of epochs to train the model on
ap.add_argument(
    "-e", "--epochs", default=3,
    help="number of training epochs")

# # add the number of images to predict on
# ap.add_argument(
#     "-nb", "--number", default=1,
#     help="number of samples to predict")

# # add the output file path to save the output
# ap.add_argument(
#     "-l", "--categorybin", required=True,
#     help="path to output category label binarizer")


# add the output file path to save the model
ap.add_argument(
    "-p", "--plot", type=str, default=False,
    help="bool to generate plots")

args = vars(ap.parse_args())

if __name__ == '__main__':

    dataset_dict = {
        "train": "data/UTKFace_train/",
        "predict": "predict_data/UTKFace_predict/"
        }

    run = args["run"]

    if run == "train":
        train_model_file_name = "trained_models/age_gender_model.h5"
        train_folder_name = args["train_dataset"]
        nb_epochs = int(args["epochs"])
        print("NUMBER OF EPOCHS")
        print(nb_epochs)
        plot = args["plot"]

        run_training(train_folder_name, train_model_file_name, nb_epochs, plot)

    elif run == "predict":
        predict_model_file_name = "trained_models/age_gender_final_model.h5"
        pred_folder_name = args["predict_dataset"]

        run_prediction(pred_folder_name, predict_model_file_name)

    else:
        print("[ERROR] Invalid input : choose train or predict")
