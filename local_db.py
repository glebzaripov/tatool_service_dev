import os
import uuid
import sqlite3
from datetime import datetime


class Taxon:
    def __init__(self, id=None, name='', code=0, active=True, creation_time=0, update_time=0, send_time=None, model_id=None):
        self.id = id
        self.name = name
        self.code = code
        self.active = active
        self.creation_time = creation_time
        self.update_time = update_time
        self.send_time = send_time
        self.model_id = model_id

    def state_equals(self, taxon):
        return self.code == taxon.code and self.active == taxon.active

    def to_tuple_without_id(self):
        return self.name, self.code, self.active, self.creation_time, self.update_time

class Taxonomy:
    def __init__(self, id=None, model_id='', file_path=''):
        self.id = id
        self.model_id = model_id
        self.file_path = file_path

class Dao:
    def __init__(self, connection):
        self._connection = connection
        self._cursor = None
        self._has_changes = False

    def execute(self, *args, **kwargs):
        self._cursor.execute(*args, **kwargs)

    def commit(self):
        self._has_changes = False
        self._connection.commit()

    def __enter__(self):
        self._cursor = self._connection.cursor()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self._has_changes:
            self.commit()
        if self._cursor:
            self._cursor.close()
            self._cursor = None

    @staticmethod
    def ensure_db_structure(cursor):
        cursor.execute('''CREATE TABLE IF NOT EXISTS taxonomy(
                          id INTEGER PRIMARY KEY AUTOINCREMENT,
                          model_id TEXT NOT NULL,
                          file_path TEXT NOT NULL,
                          creation_time INTEGER NOT NULL,
                          deletion_time INTEGER
                       );''')
        cursor.execute('''CREATE TABLE IF NOT EXISTS taxon(
                                  id INTEGER PRIMARY KEY AUTOINCREMENT,
                                  taxonomy_id INTEGER NOT NULL,
                                  name TEXT NOT NULL,
                                  code INTEGER NOT NULL,
                                  active INTEGER NOT NULL,
                                  creation_time INTEGER NOT NULL,
                                  update_time INTEGER NOT NULL,
                                  send_time INTEGER DEFAULT NULL
                               );''')
        cursor.execute('''CREATE TABLE IF NOT EXISTS dwh_sync_state(
                                  id INTEGER PRIMARY KEY AUTOINCREMENT,
                                  last_lower_date INTEGER DEFAULT NULL
                               );''')

        #try:
        #    cursor.execute('ALTER TABLE taxon ADD COLUMN send_time INTEGER default NULL')
        #except sqlite3.Error as e:
        #    print('send_time have not added: ' + e)

        return False

class TaxonDao(Dao):
    def __init__(self, connection):
        super().__init__(connection)

    def add_taxonomy(self, model_id, file_path):
        self._cursor.execute('INSERT INTO taxonomy(model_id, file_path, creation_time) VALUES(?, ?, CURRENT_TIMESTAMP)', (model_id, file_path,))
        self._has_changes = True
        return self._cursor.lastrowid

    def get_taxonomy(self, model_id):
        self._cursor.execute('SELECT id, model_id, file_path FROM taxonomy WHERE model_id = ?', (model_id,))
        result = self._cursor.fetchone()
        return Taxonomy(*result) if result else None

    def get_or_add_taxonomy(self, model_id, file_path):
        taxonomy = self.get_taxonomy(model_id)
        if taxonomy is None:
            taxonomy = Taxonomy(id=self.add_taxonomy(model_id, file_path))
        elif taxonomy.file_path != file_path:
            self._cursor.execute('UPDATE taxonomy SET file_path = ? WHERE id = ?', (taxonomy.file_path, taxonomy.id,))
            self._has_changes = True

        return taxonomy.id

    def get_taxonomy_paths(self):
        self._cursor.execute('SELECT file_path FROM taxonomy')
        result = self._cursor.fetchall()
        return [t for (t,) in result]

    def get_taxon(self, taxonomy_id, name):
        self._cursor.execute('SELECT t.id, t.name, t.code, t.active, t.creation_time, t.update_time, t.send_time FROM taxon t INNER JOIN taxonomy tf on t.taxonomy_id = tf.id WHERE tf.id = ? AND t.name = ?', (taxonomy_id, name))
        result = self._cursor.fetchone()
        return Taxon(*result) if result else None

    def update_taxons(self, taxons):
        self._cursor.executemany('UPDATE taxon SET code = ?, active = ?, update_time = ?, send_time = NULL WHERE id = ?', [(t.code, t.active, t.update_time, t.id) for t in taxons])
        self._has_changes = True

    def add_taxons(self, taxonomy_id, taxons):
        items = [t.to_tuple_without_id() for t in taxons]
        self._cursor.executemany('INSERT INTO taxon(taxonomy_id, name, code, active, creation_time, update_time) values (%d, ?, ?, ?, ?, ?)' % taxonomy_id, items)
        self._has_changes = True

    def get_taxon_names(self, taxonomy_id):
        self._cursor.execute('SELECT name FROM taxon WHERE taxonomy_id = ?', (taxonomy_id,))
        items = self._cursor.fetchall()
        return [item for (item,) in items]

    def get_taxons(self, taxonomy_id):
        self._cursor.execute('SELECT id, name, code, active, creation_time, update_time, send_time FROM taxon WHERE taxonomy_id = ?', (taxonomy_id,))
        return self._fetch_taxons()

    def get_unsent_taxons(self):
        self._cursor.execute('SELECT t.id, t.name, t.code, t.active, t.creation_time, t.update_time, t.send_time, tf.model_id FROM taxon t INNER JOIN taxonomy tf on t.taxonomy_id = tf.id WHERE t.send_time IS NULL')
        return self._fetch_taxons()

    def set_taxons_send_time(self, taxons):
        self._cursor.executemany('UPDATE taxon SET send_time = ? WHERE id = ?', [(t.send_time, t.id) for t in taxons])
        self._has_changes = True

    def _fetch_taxons(self):
        items = self._cursor.fetchall()
        taxons = [Taxon(*item) for item in items]
        return taxons

class DwhSyncDao(Dao):
    def __init__(self, connection):
        super().__init__(connection)

    def get_last_lower_date(self):
        self._cursor.execute('SELECT last_lower_date FROM dwh_sync_state')
        result = self._cursor.fetchone()
        return result[0] if result else None

    def set_last_lower_date(self, date):
        self._cursor.execute('UPDATE dwh_sync_state SET last_lower_date = ?', (date,))
        if self._cursor.rowcount == 0:
            self._cursor.execute('INSERT INTO dwh_sync_state(last_lower_date) values (?)', (date,))

        self._has_changes = True

class Database:
    def __init__(self, db_path, conn_timeout=20.0):
        os.makedirs(os.path.dirname(db_path), exist_ok=True)
        self._db_path = db_path
        self._conn_timeout = conn_timeout
        self._connection = None

    def connect(self, on_connected):
        if not self._connection:
            self._connection = sqlite3.connect(self._db_path, timeout=self._conn_timeout)
            self.with_cursor(on_connected)

    def disconnect(self):
        if self._connection:
            self._connection.close()
            self._connection = None

    def get_connection(self):
        return self._connection

    def with_cursor(self, on_cursor):
        cursor = self._connection.cursor()

        commit = on_cursor(cursor)
        if commit:
            self._connection.commit()

        cursor.close()
        return

    @staticmethod
    def gen_name(prefix):
        return ('%s_%s' % (prefix, str(uuid.uuid1()))).replace('-', '_')


class Connection:
    def __init__(self, db):
        self.db = db

    def __enter__(self):
        self.db.connect(Dao.ensure_db_structure)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.db.disconnect()

    def create_taxon_dao(self):
        return TaxonDao(self.db.get_connection())

    def create_dwh_sync_dao(self):
        return DwhSyncDao(self.db.get_connection())
