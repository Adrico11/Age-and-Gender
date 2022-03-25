
class ModelEvaluator:

    def __init__(
        self, model, data_generator,
            test_idx, batch_size=32) -> None:
        self.model = model
        self.data_generator = data_generator
        self.test_idx = test_idx
        self.batch_size = self.valid_batch_size = batch_size

    def test_model(self) -> None:
        test_gen = self.data_generator.generate_test_images(
            self.test_idx, is_training=True, batch_size=self.batch_size)
        age_pred, gender_pred = self.model.predict_generator(test_gen)
        return age_pred, gender_pred

    def post_process(predictions):
        pass