import argparse

from main_model import main

# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()

# add the input file path to the images directory
ap.add_argument(
    "-d", "--dataset", required=True,
    help="path to input dataset (i.e., directory of images)")

# add the output file path to save the model
ap.add_argument(
    "-m", "--model", required=True,
    help="path to output model")

# add the output file path to save the model
ap.add_argument(
    "-e", "--epochs",
    help="number of training epochs")

# # add the output file path to save the model
# ap.add_argument(
#     "-l", "--categorybin", required=True,
#     help="path to output category label binarizer")

# # add the output file path to save the model
# ap.add_argument(
#     "-c", "--colorbin", required=True, 
#     help="path to output color label binarizer")

# add the output file path to save the model
ap.add_argument(
    "-p", "--plot", type=str, default="output",
    help="base filename for generated plots")

args = vars(ap.parse_args())

if __name__ == '__main__':

    #  data/UTKFace/
    # 'age_gender_model.h5'

    data_folder_name = args["dataset"]
    model_file_name = args["model"]
    nb_epochs = int(args["epochs"])
    main(data_folder_name, model_file_name, nb_epochs)
