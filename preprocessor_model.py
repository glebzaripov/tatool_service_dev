import os

from deeppavlov.models.preprocessors.bert_preprocessor import BertPreprocessor

class PreprocessorModelConfig:
    def __init__(self, **config_params):
        self.id = None
        self.base_path = None
        self.config_file = None
        self.model = None
        self.vocabulary = None
        self.max_seq_length = None

        self.__dict__.update(config_params)

class PreprocessorModel:
    def __init__(self, path, config):
        self.config = config
        self.base_path = os.path.join(path, self.config.base_path)
        self.bert_preprocessor = BertPreprocessor(
            vocab_file=os.path.join(self.base_path, self.config.vocabulary),
            do_lower_case=False,
            max_seq_length=self.config.max_seq_length)

    def get_config_file(self):
        return os.path.join(self.base_path, self.config.config_file)

    def get_model_checkpoint(self):
        return os.path.join(self.base_path, self.config.model)

    def call(self, batch):
        return self.bert_preprocessor(batch)
