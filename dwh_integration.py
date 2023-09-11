from datetime import datetime, timedelta
from sqlalchemy import create_engine, Table, Column, Integer, String, MetaData, DateTime
from timeloop import Timeloop
from es_logger import EsLogger, EventType
from config import load_config
#from apscheduler.schedulers.background import BackgroundScheduler, BlockingScheduler

import local_db
import requests
import json
import time

tl = Timeloop()
integration = None

def _connect_to_redshift(redshift_url):
    return create_engine(redshift_url, pool_pre_ping=True).connect().execution_options(autocommit=True)


def _load_comments(connection, start_date, limit):
    sql = f'''SELECT 
                 id,
                 REGEXP_REPLACE(REPLACE(REPLACE((REPLACE(nps_comment, '|', '')), chr(10), ''), '\r', '' ), '\n', '' ) AS "nps_comment",
                 message_created_date,
                 COALESCE (score, first_sms_score) score
              FROM datamart_rrp_nps.v_nps_fact_data
              WHERE message_status = 'CompletedSMS' and order_of_sms = 'comment_message' and message_created_date > '{start_date}'
              UNION
              SELECT
                 id,
                 REGEXP_REPLACE(REPLACE(replace((REPLACE(nps_comment, '|', '')), chr(10), ''), '\r', '' ), '\n', '' ) AS "nps_comment",
                 answerdate as message_created_date,
                 score
              FROM u_durmaale.nps_trigger_comm
              WHERE message_created_date > '{start_date}'
              ORDER BY message_created_date DESC
              LIMIT {limit}'''
    comments = connection.execute(sql)
    return comments

class DwhIntegration:
    def __init__(self, config, logger):
        self.config = config
        self.logger = logger
        self.dwh_config = self.config['dwh']
        self.cls_config = self.dwh_config['classification']

    def process_dwh_comments(self):
        local_db_path = self.config['local_db']['path']

        try:
            with _connect_to_redshift(self.dwh_config['redshift_url']) as rs:
                with local_db.Connection(local_db.Database(local_db_path)) as local_connection:
                    with local_connection.create_dwh_sync_dao() as local_dao:
                        start_date = local_dao.get_last_lower_date()
                        if start_date is None:
                            start_date = datetime.now() + timedelta(days=-1)
                            logger.log_event('Redshift integration: No previously used start_date fetching comments since %s' % start_date)

                        comments = _load_comments(rs, start_date, self.dwh_config['batch_limit'])
                        comments = [c for c in comments]
                        logger.log_event('Redshift integration: Fetched %d new comments in Redshift since %s' % (len(comments), start_date))

                        if len(comments) > 0:
                            token = self.login()
                            if token is not None:
                                classified_comments = self.classify_comments(token, comments, self.dwh_config['classification_limit'])
                                if len(classified_comments) > 0:
                                    self.store_classified_comments(rs, comments, classified_comments)
                                    logger.log_event('Redshift integration: Stored %d classified comments' % len(classified_comments))
                                    new_start_date = comments[0].message_created_date  # datetime.now()
                                    local_dao.set_last_lower_date(new_start_date)
                                    logger.log_event('Redshift integration: New start_date=%s' % new_start_date)

        except Exception as ex:
            logger.log_event(f'Redshift integration: Failed to process comments since {start_date}: {ex}', EventType.error())

    def login(self):
        try:
            base_url = self.cls_config['url']
            response = requests.post(f'{base_url}/login', auth=(self.cls_config['user'], self.cls_config['password']), verify=False)
            logger.log_event(f'Login result: {response.text}')
            token = json.loads(response.text)['access_token']
            return token
        except Exception as e:
            logger.log_event(f'Failed to login to npsta: {e}', EventType.error())
            return None

    def _make_classification_item(self, comment):
        return {
            'id': comment['id'],
            'nps_comments': comment['nps_comment'],
            'model_ids': [self.cls_config['model']]
        }

    def _make_store_item(self, classified_comment, time):
        return {
            'hash_message_id': classified_comment['id'],
            'category_name': classified_comment[self.cls_config['model']],
            'comment_type': '',
            'loading_datetime': time
        }

    def _list_into_chunks(self, lst, n):
        return [lst[i:i + n] for i in range(0, len(lst), n)]

    def classify_comments(self, token, comments, batch_size):
        comments_to_classify = [self._make_classification_item(comment) for comment in comments]
        if len(comments_to_classify) <= 0:
            return []

        base_url = self.cls_config['url']
        classified_comments = []
        for batch in self._list_into_chunks(comments_to_classify, batch_size):
            response = requests.post(f'{base_url}/classification', json=batch,
                                     headers={'Authorization': f'Bearer {token}'}, verify=False)
            result = json.loads(response.text)
            classified_comments.extend(result)
            logger.log_event(f'Redshift integration: {len(classified_comments)} of {len(comments)} comments classified')
        return classified_comments

    def store_classified_comments(self, rs, comments, classified_comments):
        time = datetime.now()
        comments_to_store = [self._make_store_item(comment, time) for comment in classified_comments]

        metadata = MetaData()
        table = Table('nps_categorized_comments_tatool', metadata,
                      Column('hash_message_id', Integer),
                      Column('category_name', String),
                      Column('comment_type', String),
                      Column('loading_datetime', DateTime),
                      schema='datamart_rrp_nps')

        rs.execute(table.insert(), comments_to_store)

def test_classification(integration):
    token = integration.login()
    comments = [{
        'id': 'a0o4H00000AFEu3QAH',
        'nps_comment': 'Намного лучше, чем сигареты',
        'model_ids': ['categories1']
    }]
    classified = integration.classify_comments(token, comments)
    rs = _connect_to_redshift(integration.dwh_config['redshift_url'])
    integration.store_classified_comments(rs, comments, classified)

@tl.job(interval=timedelta(minutes=10))
def process_dwh_comments():
    integration.process_dwh_comments()

if __name__ == '__main__':
    print('Starting DWH integration...')
    config = load_config()
    logger = EsLogger(config['elastic_logging'])
    integration = DwhIntegration(config, logger)

    #process_dwh_comments()

    #scheduler = BlockingScheduler()
    #scheduler.add_job(process_dwh_comments, 'interval', seconds=5)
    #scheduler.start()
    #tl.stop()

    tl.start(block=True)

    logger.join()
