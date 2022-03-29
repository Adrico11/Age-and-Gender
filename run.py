import argparse

from main_file import run_training, run_prediction

# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()

ap.add_argument(
    "-r", "--run", choices=['train', 'predict'],
    required=True, help="choose : train model or make prediction"
)

# add the input file path to the images directory
ap.add_argument(
    "-d", "--dataset", default="data/UTKFace_train/",
    help="path to input dataset (i.e., directory of images)")

# # add the output file path to save the model
# ap.add_argument(
#     "-m", "--model",  default="trained_models/age_gender_model.h5",
#     help="path to output model")

# add the number of epochs to train the model on
ap.add_argument(
    "-e", "--epochs", default=10,
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

    #  data/UTKFace/
    # 'age_gender_model.h5'

    dataset_dict = {
        "train": "data/UTKFace_train/", "predict": "data/UTKFace_test/"
        }

    model_file_name = "trained_models/age_gender_model_50.h5"  # args["model"]

    run = args["run"]

    if run == "train":  # or test

        train_folder_name = args["dataset"]
        nb_epochs = int(args["epochs"])
        plot = args["plot"]

        run_training(train_folder_name, model_file_name, nb_epochs, plot)

    elif run == "predict":
        if args["dataset"] != dataset_dict["train"]:
            pred_folder_name = args["dataset"]
        else:
            pred_folder_name = dataset_dict["predict"]
        # nb_pred = int(args["number"])

        run_prediction(pred_folder_name, model_file_name)  # nb_pred)

    else:
        print("[ERROR] Invalid input : choose train or predict")
