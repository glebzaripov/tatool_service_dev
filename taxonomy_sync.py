import local_db
import requests
import pathlib
import time
import os

from deeppavlov.core.data.simple_vocab import SimpleVocabulary
from datetime import datetime, timezone
from dateutil import tz
from es_logger import EsLogger, EventType

test_taxons1 = [local_db.Taxon('taxonA', 0), local_db.Taxon('taxonB', 1), local_db.Taxon('taxonC', 2)]
test_taxons2 = [local_db.Taxon('taxonA', 0), local_db.Taxon('taxonC', 1)]
test_taxons3 = [local_db.Taxon('taxonA', 0), local_db.Taxon('taxonC', 1), local_db.Taxon('taxonD', 3)]

def get_timestamp_ms():
    return int(time.time() * 1000)

def timestamp_to_iso(timestamp):
    return datetime.fromtimestamp(timestamp / 1000.0, tz.tzlocal()).isoformat()

def get_taxons_dict(taxons):
    taxons_dict = dict()
    for taxon in taxons:
        taxons_dict[taxon.name] = taxon
    return taxons_dict

def load_taxons(taxonomy_path):
    vocabulary = SimpleVocabulary(save_path=taxonomy_path)
    vocabulary.load()

    timestamp = get_timestamp_ms()
    taxons = [local_db.Taxon(name=name, code=code, active=True, creation_time=timestamp, update_time=timestamp)
              for name, code in zip(vocabulary, vocabulary.values())]
    return taxons

def sync_taxonomy(taxonomy_name, taxonomy_path, dao, logger):
    taxonomy_id = dao.get_or_add_taxonomy(taxonomy_name, taxonomy_path)
    current_taxons = load_taxons(taxonomy_path)

    taxons_to_add = []
    taxons_to_update = []

    for taxon in current_taxons:
        stored_taxon = dao.get_taxon(taxonomy_id, taxon.name)
        if stored_taxon is None:
            taxons_to_add.append(taxon)
        elif not taxon.state_equals(stored_taxon):
            taxons_to_update.append(taxon)

    stored_taxons = dao.get_taxons(taxonomy_id)
    current_taxons_dict = get_taxons_dict(current_taxons)

    for taxon in stored_taxons:
        if taxon.name not in current_taxons_dict:
            taxon.active = False
            taxon.update_time = get_timestamp_ms()
            taxons_to_update.append(taxon)

    dao.add_taxons(taxonomy_id, taxons_to_add)
    dao.update_taxons(taxons_to_update)

    logger.log_event('Detected %d new taxons, %d updated taxons' % (len(taxons_to_add), len(taxons_to_update)))

    # return taxons with timestamp
    return taxons_to_add + taxons_to_update

def load_and_sync_taxonomies(config, logger):
    local_db_path = config['local_db']['path']
    taxonomy_paths = [(model['id'], os.path.join(model['base_path'], model['vocabulary'])) for model in config['classifier_models']]
    #taxonomy_paths = [('test1', 'models/test_categories.txt')]

    sync_taxonomies(local_db_path, taxonomy_paths, logger)

def send_updated_taxons(config, logger):
    local_db_path = config['local_db']['path']

    env_type = config['environments']['kafka_api']
    if env_type in config['kafka_api']:
        kafka_config = config['kafka_api'][env_type]
        send_updated_taxons_impl(local_db_path, kafka_config, logger)
    else:
        logger.log_event('Taxonomy sync disabled: non-existing kafka env %s' % env_type)

def sync_taxonomies(db_path, taxonomy_paths, logger):
    try:
        db = local_db.Database(db_path)
        with local_db.Connection(db) as connection:
            with connection.create_taxon_dao() as dao:
                for (taxonomy_name, taxonomy_path) in taxonomy_paths:
                    try:
                        updated_taxons = sync_taxonomy(taxonomy_name, taxonomy_path, dao, logger)
                    except Exception as ex:
                        logger.log_event('Failed to get update taxons %s: %s' % (taxonomy_path, ex), EventType.error())
    except Exception as outerEx:
        logger.log_event('Failed to sync taxonomy %s: %s' % (taxonomy_path, outerEx), EventType.error())

def send_updated_taxons_impl(db_path, kafka_config, logger):
    try:
        db = local_db.Database(db_path)
        with local_db.Connection(db) as connection:
            with connection.create_taxon_dao() as dao:
                updated_taxons = dao.get_unsent_taxons()
                sent_taxons = send_taxons(updated_taxons, kafka_config, logger)
                dao.set_taxons_send_time(sent_taxons)
    except Exception as outerEx:
        logger.log_event('Failed to send updated taxons: %s' % outerEx, EventType.error())

def send_taxons(taxons, config, logger):
    try:
        sent = []
        timestamp = get_timestamp_ms()
        for taxon in taxons:
            try:
                taxonomy_name = taxon.model_id
                event = _taxon_to_event(taxonomy_name, taxon, timestamp)
                _send_taxon_to_kafka(event, config, logger)
                taxon.send_time = get_timestamp_ms()
                sent.append(taxon)
            except Exception as sendEx:
                logger.log_event('Failed to send taxon %s to ESB: %s' % (taxon.name, sendEx), EventType.error())
        return sent
    except Exception as outerEx:
        logger.log_event('Failed to send taxons to ESB: %s' % outerEx, EventType.error())

def _send_taxon_to_kafka(event, config, logger):
    url = '%s/messaging/t.%s/send' % (config['base_url'], config['taxons_topic'])
    response = requests.post(url, json=event, auth=(config['user'], config['password']))
    logger.log_event('Request to ESB %s\r\n%s\r\n\r\n%s' % (url, event, response.text))

def get_taxon_code(taxonomy_name, taxon_name):
    taxon_processed = taxon_name.lower()
    for ch in [' ', '/', '\\', ',', '.', '&', '(', ')', '"']:
        taxon_processed = taxon_processed.replace(ch, '')
    return '%s_%s' % (taxonomy_name, taxon_processed)

def _taxon_to_event(taxonomy_name, taxon, timestamp):
    event = {
        "BusinessEvent": {
            "$type": "PMI.BDDM.Common.Update",
            "EventTime": timestamp_to_iso(timestamp),
            "BusinessEntity": {
                "$type": "PMI.BDDM.Transactionaldata.ConsumerAnswerTaxon",
                "Name": taxon.name,
                "Code": get_taxon_code(taxonomy_name, taxon.name),
                "CodeSpace": "TATool",
                "Description": taxon.name,
                "Taxonomy": {
                    "Code": taxonomy_name,
                    "Name": taxonomy_name,
                },
                "Status": "Active" if taxon.active else "Inactive",
                "CreationTime": timestamp_to_iso(taxon.creation_time),
                "UpdateTime": timestamp_to_iso(taxon.update_time),
            }
        }
    }
    return event
