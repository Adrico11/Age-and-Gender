from tensorflow.keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint, EarlyStopping


class ModelTrainer:

    def __init__(
        self, model, data_generator, train_idx,
            valid_idx, model_file, nb_epochs,
            init_lr=1e-4, batch_size=32) -> None:
        self.model = model
        self.data_generator = data_generator
        self.train_idx = train_idx
        self.valid_idx = valid_idx
        self.model_file = model_file
        self.init_lr = init_lr
        self.nb_epochs = nb_epochs
        self.batch_size = self.valid_batch_size = batch_size

    def create_optimizer(self) -> None:
        opti = Adam(
            learning_rate=self.init_lr, decay=self.init_lr / self.nb_epochs)
        return opti

    def compile_model(self) -> None:
        opti = self.create_optimizer()
        self.model.compile(
            optimizer=opti,
            loss={
                'age_output': 'mse', 'gender_output': 'binary_crossentropy'},
            loss_weights={
                'age_output': 4.,
                'gender_output': 0.1},
            metrics={
                'age_output': 'mae',
                'gender_output': 'accuracy'})

    def train_model(self) -> None:

        checkpointer = ModelCheckpoint(
            self.model_file, monitor='val_loss', verbose=0,
            save_best_only=True, save_weights_only=False,
            mode='auto', save_freq='epoch')
        early_stopper = EarlyStopping(
            patience=self.nb_epochs//2, monitor='val_loss',
            restore_best_weights=True, verbose=0)
        callback_list = [checkpointer, early_stopper]

        train_gen = self.data_generator.generate_images(
            self.train_idx, is_training=True, batch_size=self.batch_size)
        valid_gen = self.data_generator.generate_images(
            self.valid_idx, is_training=True, batch_size=self.valid_batch_size)

        # shuffle = True ???
        history = self.model.fit(
            train_gen, steps_per_epoch=len(self.train_idx)//self.batch_size,
            epochs=self.nb_epochs, callbacks=callback_list,
            validation_data=valid_gen, verbose=2,
            validation_steps=len(self.valid_idx)//self.valid_batch_size)
        self.history = history


if __name__ == '__main__':

    init_lr = 1e-4
    nb_epochs = 100
    batch_size = 32
    model_file = 'age_gender_model.h5'
