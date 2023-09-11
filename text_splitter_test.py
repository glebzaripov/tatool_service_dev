import yaml
import json
import re
import taxonomy_sync
import es_logger
from text_spliter import Splitter
from medalia_integration import MedaliaInboundApi

def load_config(filename):
    with open(filename, 'r') as stream:
        try:
            return yaml.safe_load(stream)
        except yaml.YAMLError as exc:
            print(exc)
            raise exc


def save_token(token):
    print('oauth token:', token)

def test_medalia_api(config, logger):
    api_config = config['medalia_api']['test']
    api = MedaliaInboundApi(api_config, None, None)

    data1 = {
        "categories1": [
            "Alternative to cigarette"
        ],
        "id": "a0o4H00000AFEu3QAH",
        "index": 0,
        "nps_comments": "Намного лучше, чем сигареты"
    }

    data2 = {
        "categories1": [
            "Alternative to cigarette"
        ],
        "id": "a0o4H00000AFEu3QAH-2",
        "index": 1,
        "nps_comments": "Намного лучше, чем сигареты 22"
    }

    #res = api.import_data(json.dumps(data, ensure_ascii=False))

    api.queue_data(json.dumps(data1, ensure_ascii=False))
    api.queue_data(json.dumps(data2, ensure_ascii=False))
    res = api.import_data_batch()
    print(res)

def test_regex():
    src_text = "6|Нравится обслуживание , само устройство и вообще очень хорошие эмоции )|null|null"
    #text = re.sub(r'[^a-яА-Я(),!?:;]', " ", src_text)
    text = re.sub(r'[|qwertyuiopasdfghjklzxcvbnm@#$%^&*~`«»<>=\-\+\[\]\{\}\d]', " ", src_text)
    print(text)

if __name__ == '__main__':
    #r = Splitter.split('Hello world! This is some test. Hey-hey', 18)
    #print(_get_taxon_code('categories1', 'smoke & smell'))
    #exit(0)
    test_regex()
    exit(0)

    config = load_config('npsta_config.yml')
    logger = es_logger.EsLogger(config['elastic_logging'])
    test_medalia_api(config, logger)
    #taxonomy_sync.load_and_sync_taxonomies(config, logger)
