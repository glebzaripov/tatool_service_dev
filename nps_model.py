import os
import numpy
import pandas as pd

from deeppavlov.core.data.simple_vocab import SimpleVocabulary
from deeppavlov.models.classifiers.proba2labels import Proba2Labels
from deeppavlov.models.bert.bert_classifier import BertClassifierModel
from text_spliter import Splitter
from itertools import chain


class ClassifierModelConfig:
    def __init__(self, **config_params):
        self.id = None
        self.base_path = None
        self.preprocessor_model_id = None
        self.model = None
        self.vocabulary = None
        self.confident_threshold = 0.4
        self.max_text_length = 70
        self.multilabel = True

        self.__dict__.update(config_params)


class NpsModel:
    def __init__(self, path, config, preprocessor):
        base_path = os.path.join(path, config.base_path)

        self.config = config
        self.bert_preprocessor = preprocessor
        self.vocabulary = SimpleVocabulary(save_path=os.path.join(base_path, config.vocabulary))
        self.vocabulary.load()

        self.prob2labels = Proba2Labels(confident_threshold=config.confident_threshold)

        #rubert_path = os.path.join(path, r'models/rubert_cased_L-12_H-768_A-12_v2')
        #self.bert_preprocessor = BertPreprocessor(
        #    vocab_file=os.path.join(rubert_path, r'vocab.txt'),
        #    do_lower_case=False,
        #    max_seq_length=64)

        self.bert_classifier = BertClassifierModel(
            n_classes=self.vocabulary.len,
            return_probas=True,
            one_hot_labels=True,
            bert_config_file=preprocessor.get_config_file(), #os.path.join(rubert_path, r'bert_config.json'),
            pretrained_bert=preprocessor.get_model_checkpoint(), #os.path.join(rubert_path, r'bert_model.ckpt.data-00000-of-00001'),
            save_path=os.path.join(base_path, config.model), #'models/categories/model'),
            load_path=os.path.join(base_path, config.model), #'models/categories/model'),
            keep_prob=0.5,
            learning_rate=1e-05,
            learning_rate_drop_patience=5,
            learning_rate_drop_div=2.0,
            multilabel=config.multilabel
        )

    def classify(self, data, categories_column, batch_size=10000):
        #res = pd.DataFrame()
        #res['id'] = data['id']
        #res[categories_column] = [[categories_column + '_test'] for _ in range(len(data))]
        #return res
        # ~temp

        res = []
        for i in range(0, len(data), batch_size):
            data_batch = data[i:i + batch_size]

            split_comments, split_count = self.__split_comments(data_batch['nps_comments'], self.config.max_text_length)
            bert_encoded_comments = self.bert_preprocessor.call(split_comments)
            split_probabilities = self.bert_classifier(bert_encoded_comments)
            probabilities = self.__join_probabilities(split_probabilities, split_count, self.config.multilabel)

            categories = self.vocabulary(self.prob2labels(probabilities))
            result_batch = NpsModel.__create_result_batch(data_batch, categories, categories_column)

            if len(res) == 0:
                res = result_batch
            else:
                res = res.append(result_batch)

        return res

    def __split_comments(self, comments, limit):
        #comments_frame = pd.DataFrame()
        #comments_frame['parts'] =
        #comments_frame = comments_frame.reset_index()
        parts = comments.map(lambda c: Splitter.split(c, limit))

        split_comments = pd.Series(chain.from_iterable(parts.tolist()))
        split_count = parts.str.len()

        return split_comments, split_count

    def __join_probabilities(self, probabilities, split_count, multilabel):
        source_count = len(split_count)
        cum_count = split_count.cumsum()

        joined_probabilities = numpy.ndarray(shape=(source_count, probabilities.shape[1]))
        for index, (last_index, count) in enumerate(zip(cum_count, split_count)):
            joined_probabilities[index, :] = probabilities[last_index-count:last_index, :].max(axis=0)
            if not multilabel:
                max_index = joined_probabilities[index, :].argmax()
                max_value = joined_probabilities[index, max_index]
                joined_probabilities[index, :] = 0
                joined_probabilities[index, max_index] = max_value

        return joined_probabilities

    @staticmethod
    def __create_result_batch(data_batch, categories, categories_column):
        result = pd.DataFrame()
        result['id'] = data_batch['id']
        result[categories_column] = list(map(NpsModel.process_blank_category, categories))

        #if 'nps_comments' in result_fields:
        #    result['nps_comments'] = data_batch['nps_comments']

        return result

    @staticmethod
    def process_blank_category(categories):
        if len(categories) == 0:
            return ['blank']

        if len(categories) == 1:
            return categories

        return [c for c in categories if c != 'blank']
