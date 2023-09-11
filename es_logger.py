import threading
import json
import elasticsearch
import elasticsearch.helpers

from datetime import datetime

class EventType(object):
    @staticmethod
    def info():
        return 'info'

    @staticmethod
    def error():
        return 'error'

class EsLogger:
    def __init__(self, config):
        self._es = elasticsearch.Elasticsearch([{'host': config['host'], 'port': config['port']}],
                                               http_auth = (config['user'], config['password']),
                                               http_compress = False,
                                               scheme="https")
        self.bulk_period = config['bulk_period']
        self.records_lock = threading.RLock()
        self.records = list()
        self.enabled = config['enabled']

        self._create_indexes(config['index_file'])

        if self.bulk_period > 0:
            self.send_stopped = threading.Event()
            self.send_thread = threading.Thread(None, self._send_thread_func)
            self.send_thread.start()

    def stop(self):
        self.send_stopped.set()

    def join(self, timeout = None):
        self.send_stopped.set()
        self.send_thread.join(timeout)

    def log_event(self, message, event_type = EventType.info()):
        print(message)
        if not self.enabled:
            return

        return self._log_record('logs-npsta-service', {
            'message': message,
            'details': {},
            'tags': [event_type]
        })

    @staticmethod
    def _add_common_fields(record):
        record['@timestamp'] = datetime.utcnow().isoformat()[:-3] + 'Z'
        return record

    def _log_record(self, index, record):
        record = EsLogger._add_common_fields(record)
        if self.bulk_period > 0:
            with self.records_lock:
                record['_index'] = index
                record['_op_type'] = 'create'
                self.records.append(record)
        else:
            self._send_record(index, record)

    def _send_record(self, index, body):
        try:
            self._es.index(index = index, body = body)
        except Exception as ex:
            print('Error in indexing data %s: ' % str(ex))

    def _send_thread_func(self):
        while not self.send_stopped.wait(self.bulk_period):
            with self.records_lock:
                if len(self.records) > 0:
                    records = list(self.records)
                    self.records.clear()
                else:
                    continue

            try:
                result = elasticsearch.helpers.bulk(self._es, records)
                #print(result)
            except elasticsearch.helpers.BulkIndexError as e:
                self._log_error(e)
            except ConnectionError as e:
                self._log_error(e)
                with self.records_lock:
                    self.records.extend(records)
            except Exception as e:
                self._log_error(e)

    def _log_error(self, e):
        print('Failed to send bulk records to ElasticSearch: %s' % str(e))

    def _create_indexes(self, index_filepath):
        with open(index_filepath) as json_file:
            indexes = json.load(json_file)

        for index in indexes:
            self._create_index(index['name'], index['definition'])

    def _create_index(self, index_name, body):
        try:
            if not self._es.indices.exists(index_name):
                self._es.indices.create(index = index_name, ignore = 400, body = body)
                print('Created elastic search index %s' % index_name)
        except Exception as ex:
            print(str(ex))
