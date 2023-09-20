import json
import math
import os
import re
import time
import socket
import taxonomy_sync

from flask import Flask, request, jsonify
from pandas.io.json import json_normalize
from flask_httpauth import HTTPBasicAuth, HTTPTokenAuth
from timeloop import Timeloop
from datetime import timedelta
from es_logger import EsLogger, EventType
from medalia_integration import MedaliaInboundApi
from taxonomy_sync import get_taxon_code
from config import load_config

from preprocessor_model import *
from nps_model import *
from user import User

import torch
from new_model.model import ReplyModel

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
pd.set_option('display.max_colwidth', -1)

class Server:
    def __init__(self, path):
        self.config = load_config()

        authentication = self.config['authentication']
        self.users = [User(uid=user['uid'], name=user['username'], password=user['password']) for user in authentication]

        self.token_expiration_time = 86400
        self.logger = EsLogger(self.config['elastic_logging'])

        environments = self.config['environments']
        medalia_env = environments['medalia_api']
        self.medalia = MedaliaInboundApi(self.config['medalia_api'][medalia_env], self.logger, None)

        taxonomy_sync.load_and_sync_taxonomies(self.config, self.logger)
        taxonomy_sync.send_updated_taxons(self.config, self.logger)

        self.preprocessor_models = {}
        self.models = {}
        self.model_column_prefix = 'labels_'

        for config_params in self.config['preprocessor_models']:
            config = PreprocessorModelConfig(**config_params)
            self.preprocessor_models[config.id] = PreprocessorModel(path, config)

        for config_params in self.config['classifier_models']:
            config = ClassifierModelConfig(**config_params)
            preprocessor_model = self.preprocessor_models[config.preprocessor_model_id]
            self.models[config.id] = NpsModel(path, config, preprocessor_model)

        self.load_new_model()

        self.logger.log_event('Service started on %s' % socket.gethostname())
        self.rectification = re.compile(r'[|qwertyuiopasdfghjklzxcvbnm@#$%^&*~`«»<>=\-\+\[\]\{\}\d]')

    def load_new_model(self):
        config = list(filter(lambda x: x['id'] == 'categories1',
                             self.config['classifier_models_new']))[0]
        self.models['categories1_new'] = ReplyModel().load_from_checkpoint(
            f"./new_model/{config.get('ckpt_name')}",
            map_location=torch.device('cpu'),
            strict=False
        )
        self.models['categories1_new'].eval()
        self.models['categories1_new'].freeze()

        if self.models.get('categories1') is None:
            self.models['categories1'] = self.models['categories1_new']

    def join(self):
        self.medalia.join()
        self.logger.join()

    def generate_token(self, user):
        token = user.generate_auth_token(self.token_expiration_time)
        body = json.dumps(
            {'access_token': token.decode('ascii'), 'expires_in': self.token_expiration_time, 'token_type': 'bearer'})
        return self.__json_response(body)

    def verify_password(self, username_or_token, password):
        # try to authenticate with username/password
        user = next((u for u in self.users if u.name == username_or_token), None)
        if user and user.verify_password(password):
            return user

    def verify_token(self, token):
        user = User.verify_auth_token(token, self.users)
        return user

    def test_model(self):
        path = '.'
        data = pd.read_csv(os.path.join(path, r'test.csv'))
        result = self.models['categories1'].classify(data, [])
        result.to_csv(os.path.join(path, 'sample.csv'))

    def test_classify(self):
        started = time.asctime(time.localtime(time.time()))
        time.sleep(2)
        return {'started': started, 'ended': time.asctime(time.localtime(time.time()))}

    def execute_request(self, handler):
        try:
            response = handler()
            message = 'Request %s\r\n%s%s\r\n\r\n%s' % \
                      (request.url, request.headers, request.json, response)
            self.logger.log_event(message, EventType.info())
            return response
        except BaseException as ex:
            message = 'Failed to execute request %s\r\n%s%s\r\n\r\n%s' % \
                      (request.url, request.headers, request.json, ex)
            self.logger.log_event(message, EventType.error())
            raise ex

    def classify(self):
        nps_comments = json_normalize(request.json)
        self.__rectify_comments(nps_comments)
        grouped_comments = self.__split_by_models(nps_comments)

        result_by_models = {}
        for model_id, comments in grouped_comments.items():
            if model_id in self.models:
                model_categories_column = self.__get_model_column(model_id)
                if model_id == 'categories1':
                    result = self.models['categories1_new'].classify(comments, model_categories_column)
                else:
                    result = self.models[model_id].classify(comments, model_categories_column)
                result_by_models[model_id] = result

        classification_result = self.__merge_results(nps_comments, result_by_models)
        response_json = classification_result.to_json(orient='records', force_ascii=False)#.encode('utf-8')

        if request.args.get('reply_to_medalia_api') == '1':
            self.medalia.queue_data(response_json)
            if request.args.get('medalia_output') == '1':
                medalia_resp = self.medalia.import_data_batch()
                medalia_json = {'http_code': medalia_resp.status_code, 'http_body': medalia_resp.text}
                return self.__json_response(json.dumps(medalia_json))

        # response = json.dumps(response_json, ensure_ascii=False).encode('utf-8')
        return self.__json_response(response_json)

    def __json_response(self, body):
        return body, 200, {'Content-Type': 'application/json; charset=utf-8'}

    def __get_model_column(self, model_id):
        return self.model_column_prefix + model_id

    def __rectify_comments(self, comments):
        for index, raw in comments.iterrows():
            raw['nps_comments'] = self.__rectify_comment(raw['nps_comments'])

    def __rectify_comment(self, text):
        text = self.rectification.sub(' ', text)
        text = ' '.join(text.split())
        return text

    def __split_by_models(self, nps_comments):
        ids_2d_list = nps_comments['model_ids'].tolist()
        model_ids = set(chain.from_iterable(ids_2d_list))

        comments_by_model = {}
        for model_id in model_ids:
            inclusions = [model_id in ids for ids in ids_2d_list]
            comments = nps_comments.loc[inclusions]
            comments_by_model[model_id] = comments

        return comments_by_model

    def __merge_results(self, nps_comments, result_by_models):
        result = nps_comments.filter(items=['id', 'nps_comments']).reset_index()
        #result['labels'] = [{} for _ in range(nps_comments.shape[0])]

        for model_id, model_result in result_by_models.items():
            result = result.merge(model_result, how='outer', on='id')

        model_ids = result_by_models.keys()
        result = result.apply(lambda row: self.__compose_model_columns(row, model_ids), axis=1)
        result = result.drop(columns=[self.__get_model_column(model_id) for model_id in model_ids])

        return result

    def __compose_model_columns(self, row, model_ids):
        for model_id in model_ids:
            model_categories = row[self.__get_model_column(model_id)]
            # skip nan values
            if isinstance(model_categories, list):
                #categories = row['labels']
                #row['labels'] = categories
                category_codes = [get_taxon_code(model_id, category) for category in model_categories]

                row[model_id] = "|".join(model_categories)
                row[model_id + '_tagid'] = "|".join(category_codes)

        return row


tl = Timeloop()
app = Flask(__name__)
basic_auth = HTTPBasicAuth()
token_auth = HTTPTokenAuth()
server = Server('.')

@tl.job(interval=timedelta(seconds=60))
def send_updated_taxons_job():
    taxonomy_sync.send_updated_taxons(server.config, server.logger)

@app.route('/health', methods=['GET'])
def health():
    return "OK"

@app.route('/classification', methods=['POST'])
@token_auth.login_required
def classify():
    return server.execute_request(server.classify)


@app.route('/login', methods=['POST'])
@basic_auth.login_required
def login():
    return server.execute_request(lambda: server.generate_token(basic_auth.current_user()))


@basic_auth.verify_password
def verify_password(username, password):
    return server.verify_password(username, password)


@token_auth.verify_token
def verify_token(token):
    return server.verify_token(token)


if __name__ == '__main__':
    tl.start()

    try:
        app.run('0.0.0.0', 5000)
    except KeyboardInterrupt:
        pass

    tl.stop()
    server.join()
