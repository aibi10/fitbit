import os
import sys
from os import abort
from flask import render_template, redirect, url_for, jsonify, session, request, Response, stream_with_context
import threading
import json
import time
import pandas as pd
from src.utility import read_params
from webapp.data_access_layer.mongo_db.mongo_db_atlas import MongoDBOperation
from webapp.project_library_layer.initializer.initializer import Initializer

from webapp.integration_layer.file_management.file_manager import FileManager
from webapp.cloud_storage_layer.aws.amazon_simple_storage_service import AmazonSimpleStorageService
from webapp.entity_layer.registration.registration import Register
from webapp.logging_layer.logger.log_request import LogRequest
from webapp.logging_layer.logger.log_exception import LogExceptionDetail
from webapp.entity_layer.project.project import Project
from webapp.entity_layer.project.project_configuration import ProjectConfiguration
from webapp.thread_layer.train_model_thread.train_model_thread import TrainModelThread
from webapp.thread_layer.predict_from_model_thread.predict_from_model_thread import PredictFromModelThread
from webapp.data_access_layer.mongo_db.mongo_db_atlas import MongoDBOperation
import json

import uuid
from webapp.logging_layer.logger.logger import AppLogger

global process_value


class MachineLearningController:

    def __init__(self):
        self.registration_obj = Register()
        self.project_detail = Project()
        self.project_config = ProjectConfiguration()
        self.WRITE = "WRITE"
        self.READ = "READ"

    def predict_route_client(self):
        project_id = None
        try:
            log_writer = LogRequest(executed_by=None, execution_id=str(uuid.uuid4()))
            try:

                if 'email_address' in session:
                    log_writer.executed_by = session['email_address']
                    log_writer.log_start(request)
                    requested_project_data = json.loads(request.data)
                    project_id = None
                    if 'project_id' in requested_project_data:
                        project_id = int(requested_project_data['project_id'])

                    if project_id is None:
                        raise Exception('Project id required')

                    result = self.registration_obj.validate_access(session['email_address'], operation_type=self.WRITE)
                    if not result['status']:
                        log_writer.log_stop(result)
                        result.update(
                            {'message_status': 'info', 'project_id': project_id,
                             'execution_id': log_writer.execution_id})
                        return jsonify(result)

                    database_name = Initializer().get_training_thread_database_name()
                    collection_name = Initializer().get_thread_status_collection_name()
                    query = {'project_id': project_id, 'is_running': True}
                    result = MongoDBOperation().get_record(database_name=database_name, collection_name=collection_name,
                                                           query=query)
                    if result is not None:
                        execution_id = result['execution_id']
                    else:
                        execution_id = None

                    if execution_id is not None:
                        result = {'message': 'Training/prediction is in progress.', 'execution_id': execution_id,
                                  'status': True, 'message_status': 'info'}
                        log_writer.log_stop(result)
                        return jsonify(result)

                    result = {}
                    predict_from_model_obj = PredictFromModelThread(project_id=project_id,
                                                                    executed_by=log_writer.executed_by,
                                                                    execution_id=log_writer.execution_id,
                                                                    log_writer=log_writer)
                    predict_from_model_obj.start()
                    result.update(
                        {'message': 'Prediction started your execution id {0}'.format(log_writer.execution_id)})
                    result.update({'message_status': 'info', 'project_id': project_id, 'status': True,
                                   'execution_id': log_writer.execution_id})
                    return jsonify(result)
                else:
                    result = {'status': True, 'message': 'Please login to your account',
                              'execution_id': log_writer.execution_id}
                    log_writer.log_stop(result)
                    return jsonify(result)
            except Exception as e:
                result = {'status': False, 'message': str(e), 'message_status': 'info', 'project_id': project_id,
                          'execution_id': log_writer.execution_id}
                log_writer.log_stop(result)
                log_exception = LogExceptionDetail(log_writer.executed_by, log_writer.execution_id)
                log_exception.log(str(e))
                return jsonify(result)

        except Exception as e:
            return jsonify({'status': False,
                            'message': str(e)
                               , 'message_status': 'info', 'project_id': project_id})

    def train_route_client(self):
        project_id = None
        try:
            log_writer = LogRequest(executed_by=None, execution_id=str(uuid.uuid4()))

            try:
                # log_writer = LogRequest(executed_by=None, execution_id=str(uuid.uuid4()))
                if 'email_address' in session:
                    log_writer.executed_by = session['email_address']
                    log_writer.log_start(request)
                    requested_project_data = json.loads(request.data)
                    project_id = None
                    if 'project_id' in requested_project_data:
                        project_id = int(requested_project_data['project_id'])

                    if project_id is None:
                        raise Exception('Project id required')

                    result = self.registration_obj.validate_access(session['email_address'], operation_type=self.WRITE)
                    if not result['status']:
                        log_writer.log_stop(result)
                        result.update(
                            {'message_status': 'info', 'project_id': project_id,
                             'execution_id': log_writer.execution_id})
                        return jsonify(result)
                    database_name = Initializer().get_training_thread_database_name()
                    collection_name = Initializer().get_thread_status_collection_name()
                    query = {'project_id': project_id, 'is_running': True}
                    result = MongoDBOperation().get_record(database_name=database_name, collection_name=collection_name,
                                                           query=query)
                    if result is not None:
                        execution_id = result['execution_id']
                    else:
                        execution_id = None

                    if execution_id is not None:
                        result = {'message': 'Training/prediction is in progress.', 'execution_id': execution_id,
                                  'status': True, 'message_status': 'info'}
                        log_writer.log_stop(result)
                        return jsonify(result)

                    result = {}

                    train_model = TrainModelThread(project_id=project_id, executed_by=log_writer.executed_by,
                                                   execution_id=log_writer.execution_id, log_writer=log_writer)
                    train_model.start()
                    result.update({'status': True, 'message': 'Training started. keep execution_id[{}] to track'.format(
                        log_writer.execution_id),
                                   'message_status': 'info', 'project_id': project_id,
                                   'execution_id': log_writer.execution_id})
                    log_writer.log_stop(result)
                    return jsonify(result)
                else:
                    result = {'status': True, 'message': 'Please login to your account',
                              'execution_id': log_writer.execution_id}
                    log_writer.log_stop(result)
                    return jsonify(result)

            except Exception as e:
                result = {'status': False, 'message': str(e), 'message_status': 'info', 'project_id': project_id,
                          'execution_id': log_writer.execution_id}
                log_writer.log_stop(result)
                log_exception = LogExceptionDetail(log_writer.executed_by, log_writer.execution_id)
                log_exception.log(str(e))
                return render_template('error.html',
                                       context=result)


        except Exception as e:
            result = {'status': False,
                      'message': str(e)
                , 'message_status': 'info', 'project_id': project_id, 'execution_id': None}
            return render_template('error.html',
                                   context=result)

    def prediction_output_file(self):
        project_id = None
        try:
            log_writer = LogRequest(executed_by=None, execution_id=str(uuid.uuid4()))
            try:
                # log_writer = LogRequest(executed_by=None, execution_id=str(uuid.uuid4()))
                if 'email_address' in session:
                    log_writer.executed_by = session['email_address']
                    log_writer.log_start(request)
                    project_id = request.args.get('project_id', None)

                    error_message = ""
                    if project_id is None:
                        error_message = error_message + "Project id required"
                    project_id = int(project_id)
                    result = self.project_detail.get_project_detail(project_id=project_id)
                    project_detail = result.get('project_detail', None)
                    project_name = project_detail.get('project_name', None)
                    result = self.registration_obj.validate_access(session['email_address'], operation_type=self.READ)
                    if not result['status']:
                        error_message = error_message + result['message']
                        context = {'status': True, 'project_name': project_name, 'output_file': None,
                                   'message': error_message}
                        log_writer.log_stop(context)
                        return render_template('prediction_output.html', context=context)
                    config = read_params(os.path.join("config", "params.yaml"))
                    prediction_output_file_path = config['artifacts']['prediction_data'][
                        'prediction_output_file_path']
                    prediction_file_name = config['artifacts']['prediction_data']['prediction_file_name']

                    output_file_path=os.path.join(prediction_output_file_path,prediction_file_name)
                    file_content = pd.read_csv(output_file_path)
                    file_content=file_content.head(100)
                    if file_content is None:
                        context = {'status': True, 'project_name': project_name, 'output_file': None,
                                   'message': 'Output file not found'}
                        log_writer.log_stop(context)
                        return render_template('prediction_output.html', context=context)
                    context = {'status': True, 'project_name': project_name,
                               'output_file': file_content.to_html(header="true"),
                               'message': 'Output file retrived', }
                    log_writer.log_stop(context)
                    return render_template('prediction_output.html', context=context)

                else:
                    result = {'status': True, 'message': 'Please login to your account'}
                    log_writer.log_stop(result)
                    return Response(result)
            except Exception as e:
                exc_type, exc_obj, exc_tb = sys.exc_info()
                file_name = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
                exception_type = e.__repr__()
                exception_detail = {'exception_type': exception_type,
                                    'file_name': file_name, 'line_number': exc_tb.tb_lineno,
                                    'detail': sys.exc_info().__str__()}
                print(exception_detail)
                return render_template('error.html',
                                       context={'message': None, 'status ': False, 'message_status': 'info',
                                                'error_message': exception_detail.__str__()})
        except Exception as e:
            exc_type, exc_obj, exc_tb = sys.exc_info()
            file_name = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
            exception_type = e.__repr__()
            exception_detail = {'exception_type': exception_type,
                                'file_name': file_name, 'line_number': exc_tb.tb_lineno,
                                'detail': sys.exc_info().__str__()}
            print(exception_detail)
            return render_template('error.html',
                                   context={'message': None, 'status ': False, 'message_status': 'info',
                                            'error_message': exception_detail.__str__()})

    def get_log_detail(self):
        project_id = None
        try:
            log_writer = LogRequest(executed_by=None, execution_id=str(uuid.uuid4()))

            try:
                # log_writer = LogRequest(executed_by=None, execution_id=str(uuid.uuid4()))
                if 'email_address' in session:
                    log_writer.executed_by = session['email_address']
                    log_writer.log_start(request)
                    project_id = request.args.get('project_id', None)
                    execution_id = request.args.get('execution_id', None)
                    error_message = ""
                    if project_id is None:
                        error_message = error_message + "Project id required"
                    if execution_id is None:
                        error_message = error_message + "Execution id required"
                    result = self.registration_obj.validate_access(session['email_address'], operation_type=self.READ)
                    if not result['status']:
                        error_message = error_message + result['message']

                    if len(error_message) > 0:
                        log_writer.log_stop({'status': True, 'message': error_message})
                        return Response(error_message)
                    result = MongoDBOperation().get_record(Initializer().get_training_thread_database_name(),
                                                           Initializer().get_thread_status_collection_name(),
                                                           {'execution_id': execution_id}
                                                           )
                    if result is None:
                        return Response("We don't have any log yet with execution id {}".format(execution_id))
                    process_type = result['process_type']
                    project_id = int(project_id)
                    return Response(
                        stream_with_context(AppLogger().get_log(project_id=project_id, execution_id=execution_id,
                                                                process_type=process_type)))
                else:
                    result = {'status': True, 'message': 'Please login to your account'}
                    log_writer.log_stop(result)
                    return Response(result)

            except Exception as e:
                exc_type, exc_obj, exc_tb = sys.exc_info()
                file_name = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
                exception_type = e.__repr__()
                exception_detail = {'exception_type': exception_type,
                                    'file_name': file_name, 'line_number': exc_tb.tb_lineno,
                                    'detail': sys.exc_info().__str__()}
                result = {'status': False, 'message': f"{exception_detail}", 'message_status': 'info',
                          'project_id': project_id}
                log_writer.log_stop(result)
                log_exception = LogExceptionDetail(log_writer.executed_by, log_writer.execution_id)
                log_exception.log(f"{exception_detail}")
                return render_template('error.html',
                                       context={'message': None, 'status ': False, 'message_status': 'info',
                                                'error_message': f"{exception_detail}"})

        except Exception as e:
            exc_type, exc_obj, exc_tb = sys.exc_info()
            file_name = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
            exception_type = e.__repr__()
            exception_detail = {'exception_type': exception_type,
                                'file_name': file_name, 'line_number': exc_tb.tb_lineno,
                                'detail': sys.exc_info().__str__()}

            return render_template('error.html',
                                   context={'message': None, 'status ': False, 'message_status': 'info',
                                            'error_message': f"{exception_detail}"})
