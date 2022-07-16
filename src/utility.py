import importlib
import json
import yaml

from webapp.integration_layer.file_management.file_manager import FileManager
from webapp.logging_layer.logger.logger import AppLogger
import uuid
import os
import shutil



def create_directory_path(path, is_recreate=True):
    """

    :param path:
    :param is_recreate: Default it will delete the existing directory yet you can pass
    it's value to false if you do not want to remove existing directory
    :return:
    """
    try:
        if is_recreate:
            if os.path.exists(path):
                shutil.rmtree(path, ignore_errors=False)
        os.makedirs(path, exist_ok=True)
        return True
    except Exception as e:
        raise e


def clean_data_source_dir(path, logger=None, is_logging_enable=True):
    try:
        if not os.path.exists(path):
            os.mkdir(path)
        for file in os.listdir(path):
            if '.gitignore' in file:
                pass
            logger.log(f"{os.path.join(path, file)}file will be deleted.")
            os.remove(os.path.join(path, file))
            logger.log(f"{os.path.join(path, file)}file has been deleted.")
    except Exception as e:
        raise e


def download_file_from_cloud(cloud_provider, cloud_directory_path,
                             local_system_directory_file_download_path,
                             logger,
                             is_logging_enable=True):
    """
    download_training_file_from_s3_bucket(): It will download file from cloud storage to your system
    ====================================================================================================================
    :param cloud_provider: name of cloud provider amazon,google,microsoft
    :param cloud_directory_path: path of file located at cloud don't include bucket name
    :param local_system_directory_file_download_path: local system path where file has to be downloaded
    ====================================================================================================================
    :return: True if file downloaded else False
    """
    try:

        logger.is_log_enable = is_logging_enable
        file_manager = FileManager(cloud_provider=cloud_provider)
        response = file_manager.list_files(directory_full_path=cloud_directory_path)
        if not response['status']:
            return True
        is_files_downloaded = 1
        for file_name in response['files_list']:
            logger.log(f"{file_name}file will be downloaded in dir--> {local_system_directory_file_download_path}.")
            response = file_manager.download_file(directory_full_path=cloud_directory_path,
                                                  local_system_directory=local_system_directory_file_download_path,
                                                  file_name=file_name)
            is_files_downloaded = is_files_downloaded * int(response['status'])
            logger.log(f"{file_name}file has been downloaded in dir--> {local_system_directory_file_download_path}.")
        return bool(is_files_downloaded)
    except Exception as e:
        raise e


def get_logger_object_of_training(config_path: str, collection_name, execution_id=None, executed_by=None) -> AppLogger:
    config = read_params(config_path)
    database_name = config['log_database']['training_database_name']
    project_id=int(config['base']['project_id'])
    if execution_id is None:
        execution_id = str(uuid.uuid4())
    if executed_by is None:
        executed_by = "Avnish Yadav"
    logger = AppLogger(project_id=project_id, log_database=database_name, log_collection_name=collection_name,
                       execution_id=execution_id, executed_by=executed_by)
    return logger


def get_logger_object_of_prediction(config_path: str, collection_name, execution_id=None,
                                    executed_by=None) -> AppLogger:
    config = read_params(config_path)
    database_name = config['log_database']['prediction_database_name']
    project_id=int(config['base']['project_id'])
    if execution_id is None:
        execution_id = str(uuid.uuid4())
    if executed_by is None:
        executed_by = "Avnish Yadav"
    logger = AppLogger(project_id=project_id, log_database=database_name, log_collection_name=collection_name,
                       execution_id=execution_id, executed_by=executed_by)
    return logger


def read_params(config_path: str) -> dict:
    with open(config_path) as yaml_file:
        config = yaml.safe_load(yaml_file)
    return config


def values_from_schema_function(schema_path):
    try:
        with open(schema_path, 'r') as r:
            dic = json.load(r)
            r.close()

        pattern = dic['SampleFileName']
        length_of_date_stamp_in_file = dic['LengthOfDateStampInFile']
        length_of_time_stamp_in_file = dic['LengthOfTimeStampInFile']
        column_names = dic['ColName']
        number_of_columns = dic['NumberofColumns']
        return pattern, length_of_date_stamp_in_file, length_of_time_stamp_in_file, column_names, number_of_columns
    except ValueError:
        raise ValueError

    except KeyError:
        raise KeyError

    except Exception as e:
        raise e


def class_for_name(module_name, class_name):
    try:
        # load the module, will raise ImportError if module cannot be loaded
        module = importlib.import_module(module_name)
        # get the class, will raise AttributeError if class cannot be found
        class_ref = getattr(module, class_name)
        return class_ref
    except Exception as e:
        raise e
