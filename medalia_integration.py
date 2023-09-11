import json
import es_logger
import threading
import itertools
from oauthlib.oauth2 import BackendApplicationClient, TokenExpiredError, MissingTokenError
from requests_oauthlib import OAuth2Session
from config import load_config
from es_logger import EsLogger

class MedaliaInboundApi:
    def __init__(self, config, logger, token):
        self.config = config
        self.logger = logger
        self.client_lock = threading.RLock()
        self.batch_lock = threading.RLock()
        self.batch = []
        self.client = BackendApplicationClient(client_id=config['client_id'])
        self.verify_cert = None

        self.oauth = self.__create_oauth(token)

        self.batch_period = 10
        self.batch_stopped = threading.Event()
        self.batch_thread = threading.Thread(None, self._batch_thread_func)
        self.batch_thread.start()

    def __auth(self):
        token = self.oauth.fetch_token(token_url=self.config['token_url'],
                                       client_id=self.config['client_id'],
                                       client_secret=self.config['client_secret'],
                                       verify=self.verify_cert)
        return token

    def __create_oauth(self, token):
        return OAuth2Session(client=self.client,
                             scope=self.config['scope'],
                             token=token)

    def __format_response(self, response):
        return 'HTTP %d: %s' % (response.status_code, response.text) if response is not None else 'None'

    def __post_data(self, data):
        res = self.oauth.post(self.config['import_url'],
                              headers={'Content-Type': 'application/json; charset=utf-8'},
                              data=data.encode('utf-8'),
                              verify=self.verify_cert)

        if res is None or res.status_code == 401:
            description = 'HTTP response: ' + self.__format_response(res)
            raise TokenExpiredError(description=description)

        if self.logger:
            self.logger.log_event('Response from Medalia: %d %s' % (res.status_code, res.text))

        return res

    def _batch_thread_func(self):
        self.logger.log_event('Medalia inbound API batch thread started')
        while not self.batch_stopped.wait(self.batch_period):
            try:
                self.import_data_batch()
            except Exception as e:
                self.logger.log_event('Medalia inbound API error: %s' % e, es_logger.EventType.error())

    def join(self, timeout=None):
        self.batch_stopped.set()
        self.batch_thread.join(timeout)

    def queue_data(self, data):
        with self.batch_lock:
            self.batch.append(data)

    def import_data_batch(self):
        batch = []
        with self.batch_lock:
            if len(self.batch) <= 0:
                return None
            batch.extend(self.batch)
            self.batch = []

        items = list(itertools.chain.from_iterable([json.loads(item) for item in batch]))
        batch_json = json.dumps(items, ensure_ascii=False)
        self.logger.log_event('Sending batch data to Medaila, batch size %d' % len(items))
        return self.import_data(batch_json)

    def import_data(self, data, retry_count=2):
        try:
            return self.__post_data(data)
        except TokenExpiredError as e:
            if retry_count > 0:
                with self.client_lock:
                    token = self.__auth()
                    if self.logger:
                        self.logger.log_event('Got Medalia token: %s' % (json.dumps(token)))
                    self.oauth = self.__create_oauth(token)
                return self.import_data(data, retry_count - 1)
            else:
                if self.logger:
                    self.logger.log_event('Failed to get Medalia token after set of retries', es_logger.EventType.error())
                return 'Failed to import data to Medalia: %s' % e.description
