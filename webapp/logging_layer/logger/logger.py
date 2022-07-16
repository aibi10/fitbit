import os
from datetime import datetime

import yaml

from webapp.project_library_layer.datetime_libray import date_time
from webapp.exception_layer.logger_exception.logger_exception import AppLoggerException
from webapp.cloud_storage_layer.aws.amazon_simple_storage_service import AmazonSimpleStorageService
from webapp.data_access_layer.mongo_db.mongo_db_atlas import MongoDBOperation
from webapp.project_library_layer.initializer.initializer import Initializer
import uuid
import sys
from dateutil.parser import parse
import pandas as pd


class AppLogger:
    def __init__(self, project_id=None, log_database=None, log_collection_name=None, executed_by=None,
                 execution_id=None, socket_io=None, is_log_enable=True):
        self.log_database = log_database
        self.log_collection_name = log_collection_name
        self.executed_by = executed_by
        self.execution_id = execution_id
        self.mongo_db_object = MongoDBOperation()
        self.project_id = project_id
        self.socket_io = socket_io
        self.is_log_enable = is_log_enable

    def log(self, log_message):
        if not self.is_log_enable:
            return True
        log_writer_id = str(uuid.uuid4())
        log_data = None
        try:
            if self.socket_io is not None:
                if self.log_database == Initializer().get_training_database_name():
                    self.socket_io.emit("started_training" + str(self.project_id),
                                        {
                                            'message': "<span style='color:red'>executed_by [{}]</span>"
                                                       "<span style='color:#008cba;'> exec_id {}:</span> "
                                                       "<span style='color:green;'>{}</span> {} "
                                                       ">{}".format(self.executed_by, self.execution_id,
                                                                    date_time.get_date(),
                                                                    date_time.get_time(), log_message)}
                                        , namespace="/training_model")

                if self.log_database == Initializer().get_prediction_database_name():
                    self.socket_io.emit("prediction_started" + str(self.project_id),
                                        {
                                            'message': "<span style='color:red'>executed_by [{}]</span>"
                                                       "<span style='color:#008cba;'> exec_id {}:</span> "
                                                       "<span style='color:green;'>{}</span> {} "
                                                       ">{}".format(self.executed_by, self.execution_id,
                                                                    date_time.get_date(),
                                                                    date_time.get_time(), log_message)}
                                        , namespace="/training_model")

            file_object = None
            self.now = datetime.now()
            self.date = self.now.date()
            self.current_time = self.now.strftime("%H:%M:%S")
            log_data = {
                'log_updated_date': date_time.get_date(),
                'log_update_time': date_time.get_time(),
                'execution_id': self.execution_id,
                'message': log_message,
                'executed_by': self.executed_by,
                'project_id': self.project_id,
                'log_writer_id': log_writer_id,
                'updated_date_and_time': datetime.now()
            }
            # with open("log.txt", "a") as f:
            #    f.write("{}: {} {} >{}\n".format(self.execution_id, date_time.get_date(), date_time.get_time(),
            #                                     log_message))
            self.mongo_db_object.insert_record_in_collection(
                self.log_database, self.log_collection_name, log_data)
        except Exception as e:

            app_logger_exception = AppLoggerException(
                "Failed to log data file in module [{0}] class [{1}] method [{2}] -->log detail[{3}]"
                    .format(AppLogger.__module__.__str__(), AppLogger.__name__,
                            self.log.__name__, log_data))
            message = Exception(app_logger_exception.error_message_detail(str(e), sys))
            aws = AmazonSimpleStorageService()
            file_name = 'log_' + log_writer_id + '.txt'
            aws.write_file_content('failed_log', file_name, message.__str__())

    def get_database_name(self, training=True) -> str:
        """
        Return database name
        """
        config_path = os.path.join('config', 'params.yaml')
        with open(config_path) as yaml_file:
            config = yaml.safe_load(yaml_file)
        if training:
            database_name = config['log_database']['training_database_name']
        else:
            database_name = config['log_database']['prediction_database_name']
        return database_name

    def get_log(self, project_id, execution_id, process_type=None, data_time_value=None):
        try:
            yield "Logging loading ......"
            if process_type == "training":
                database_name = self.get_database_name()
            else:
                database_name = self.get_database_name(training=False)
            client = self.mongo_db_object.get_database_client_object()
            database = self.mongo_db_object.create_database(client, database_name)
            collection_list = database.list_collection_names()
            required_collection_list = list(filter(lambda x: not x[-1].isdigit(), collection_list))
            yield '...<br>'
            yield "Logging query is getting prepared<br>"
            is_running = True
            while is_running:
                print("Hi", date_time.get_time())
                if data_time_value is not None:
                    query = {'updated_date_and_time': {"$gt": parse(data_time_value)}, 'execution_id': execution_id,
                             'project_id': project_id}
                else:
                    query = {'execution_id': execution_id,
                             'project_id': project_id}
                # print(query)

                message = []
                time_stamp = []
                for collection in required_collection_list:
                    # print(query)
                    # print(collection)
                    yield '...'
                    result = self.mongo_db_object.get_records(database_name, collection, query)
                    for res in result:
                        log_data = "</br><span style='color:green'>executed_by [{}]</span><span style='color:#008cba;'> exec_id " \
                                   "{}:</span> <span style='color:purple;'>{}</span> {} >{}</br>".format(
                            res['executed_by'],
                            res['execution_id'],
                            res['log_updated_date'],
                            res['log_update_time'],
                            res['message'])

                        time_stamp.append(res['updated_date_and_time'])
                        message.append(log_data)
                yield '...'
                if message.__len__() > 0:
                    log_df = pd.DataFrame({"time_stamp": time_stamp, 'message': message})
                    log_df = log_df.sort_values(by=['time_stamp'])
                    if isinstance(log_df['time_stamp'].max(), datetime):
                        data_time_value = log_df['time_stamp'].max().__str__()

                    for msg in log_df['message']:
                        yield msg

                    # print(data_time_value)
                database_name_thread = Initializer().get_training_thread_database_name()
                collection_name_thread = Initializer().get_thread_status_collection_name()
                project_id = int(project_id)
                query = {'project_id': project_id, 'execution_id': execution_id}
                result = MongoDBOperation().get_record(database_name=database_name_thread,
                                                       collection_name=collection_name_thread,
                                                       query=query)
                if result is None:
                    yield 'We have not found any execution log for log id {}'.format(execution_id)
                else:
                    is_running = result['is_running']

        except Exception as e:
            app_logger_exception = AppLoggerException(
                "Failed to log data file in module [{0}] class [{1}] method [{2}]"
                    .format(AppLogger.__module__.__str__(), AppLogger.__name__,
                            self.log.__name__))
            raise Exception(app_logger_exception.error_message_detail(str(e), sys))
