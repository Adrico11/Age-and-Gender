import matplotlib.pyplot as plt

from src.models.model_builder import ModelBuilder
from src.models.model_trainer import ModelTrainer

from src.data_process.data_generator import DataGenerator
from src.data_process.data_extractor import DataExtractor

# # importing sys
# import sys
# # adding Folder_2 to the system path
# sys.path.insert(0, 'C:/Users/adrie/OneDrive/Bureau/Stage/Datakalab
# /age-gender-estimation-salem-sermanet/src/data_process')

# data_folder_path = "data/UTKFace/"


def main(data_folder_path, model_file, nb_epochs):

    print("Fetching dataset")
    data_extractor = DataExtractor(data_folder_path)
    data_extractor.create_df()
    full_dataset = data_extractor.full_dataset
    print(full_dataset.head())

    print("Creating Data generator")
    data_generator = DataGenerator(
        full_dataset, data_folder_path)
    data_generator.generate_split_indexes()
    train_idx, valid_idx, test_idx = data_generator.idx_list
    # self.idx_list = [train_idx, valid_idx, test_idx]

    print("Building model")
    model_builder = ModelBuilder()
    model_builder.assemble_full_model()
    face_model = model_builder.model

    print("Training model")
    model_trainer = ModelTrainer(
        face_model, data_generator, train_idx,
        valid_idx, model_file, nb_epochs)
    model_trainer.compile_model()
    model_trainer.train_model()
    history = model_trainer.history

    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.show()


if __name__ == '__main__':
    main("data/UTKFace/", "age_gender_model.h5", 5)
