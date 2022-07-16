import subprocess
import time
from datetime import datetime, timedelta

from flask import Flask, send_file, session, request, Response, stream_with_context, jsonify, render_template

from threading import Thread, Event

import atexit
import uuid
import os
import os
import pandas as pd

from flask_cors import CORS, cross_origin

from src.utility import read_params

global process_value
from webapp.controller.home_controller.home_controller import HomeController
from webapp.controller.file_operation_controller.file_operation_controller import FileOperationController
from webapp.controller.authentication_contoller.authentication_controller import AuthenticationController
from webapp.controller.project_controller.project_controller import ProjectController
from webapp.controller.machine_learning_controller.machine_learning_controller import MachineLearningController

from webapp.project_library_layer.initializer.initializer import Initializer
from webapp.data_access_layer.mongo_db.mongo_db_atlas import MongoDBOperation
from webapp.controller.scheduler_controller.scheduler_controller import SchedulerController


class Config:
    SCHEDULER_API_ENABLED = True


def on_exit_app():
    print("Applicate failure")
    database_name = Initializer().get_training_thread_database_name()
    collection_name = Initializer().get_thread_status_collection_name()
    running_projects = MongoDBOperation().get_records(database_name=database_name, collection_name=collection_name,
                                                      query={'is_running': True})
    for running_project in running_projects:
        running_project.pop('_id')
        old_record = running_project.copy()
        running_project.update(
            {'is_running': False, 'message': 'Failure due to exit of application', 'is_Failed': True})
        MongoDBOperation().update_record_in_collection(database_name=database_name,
                                                       collection_name=collection_name,
                                                       query=old_record,
                                                       new_value=running_project
                                                       )


try:
    UPLOAD_FOLDER = 'file/'
    ALLOWED_EXTENSIONS = {'txt', 'pdf', 'png', 'jpg', 'jpeg', 'gif'}

    thread = Thread()
    thread_stop_event = Event()

    initial = Initializer()
    # app = Flask(__name__)

    os.putenv('LANG', 'en_US.UTF-8')
    os.putenv('LC_ALL', 'en_US.UTF-8')

    WEBAPP_ROOT = "webapp"
    STATIC_DIR = os.path.join(WEBAPP_ROOT, 'static')
    TEMPLATE_DIR = os.path.join(WEBAPP_ROOT, 'templates')

    app = Flask(__name__, static_folder=STATIC_DIR, template_folder=TEMPLATE_DIR)

    app.secret_key = initial.get_session_secret_key()
    app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
    CORS(app)

    scheduler_controller = SchedulerController()
    scheduler_controller.get_scheduler_object().start()

    atexit.register(on_exit_app)


    @app.route('/download_output_file', methods=['GET'])
    def download_output_file():
        config = read_params(os.path.join("config", "params.yaml"))
        prediction_output_file_path = config['artifacts']['prediction_data'][
            'prediction_output_file_path']
        prediction_file_name = config['artifacts']['prediction_data']['prediction_file_name']
        file_path = os.path.join(prediction_output_file_path, prediction_file_name)
        if os.path.exists(file_path):
            return send_file(file_path, as_attachment=True)
        else:
            return "Output file is not yet generated.<br><a href='/' > HOME</a>"

    


    home_controller = HomeController()
    app.add_url_rule('/', view_func=home_controller.index, methods=['GET', 'POST'])
    authentication_controller = AuthenticationController()
    app.add_url_rule('/validate_email_address', view_func=authentication_controller.validate_email_address,
                     methods=['GET', 'POST'])
    app.add_url_rule('/register', view_func=authentication_controller.register, methods=['GET', 'POST'])
    app.add_url_rule('/login', view_func=authentication_controller.login, methods=['GET', 'POST'])
    app.add_url_rule('/logout', view_func=authentication_controller.logout, methods=['GET', 'POST'])

    file_operation_controller = FileOperationController()
    app.add_url_rule('/cloud_list_directory', view_func=file_operation_controller.cloud_list_directory,
                     methods=['GET', 'POST'])
    app.add_url_rule('/list_directory', view_func=file_operation_controller.list_directory, methods=['GET', 'POST'])
    app.add_url_rule('/upload_files', view_func=file_operation_controller.upload_files, methods=['GET', 'POST'])
    app.add_url_rule('/create_folder', view_func=file_operation_controller.create_folder, methods=['GET', 'POST'])
    app.add_url_rule('/delete_folder', view_func=file_operation_controller.delete_folder, methods=['GET', 'POST'])
    app.add_url_rule('/delete_file', view_func=file_operation_controller.delete_file, methods=['GET', 'POST'])
    app.add_url_rule('/upload_file', view_func=file_operation_controller.upload_file_, methods=['GET', 'POST'])

    project_controller = ProjectController()
    app.add_url_rule('/project', view_func=project_controller.project_list, methods=['GET', 'POST'])
    app.add_url_rule('/project_detail', view_func=project_controller.project_detail, methods=['GET', 'POST'])
    app.add_url_rule('/save_project', view_func=project_controller.save_project_data, methods=['GET', 'POST'])
    app.add_url_rule('/save_project_config', view_func=project_controller.save_project_configuration,
                     methods=['GET', 'POST'])

    machine_learning_controller = MachineLearningController()
    app.add_url_rule('/train', view_func=machine_learning_controller.train_route_client, methods=['GET', 'POST'])
    app.add_url_rule('/predict', view_func=machine_learning_controller.predict_route_client, methods=['GET', 'POST'])
    app.add_url_rule('/stream', view_func=machine_learning_controller.get_log_detail, methods=['GET', 'POST'])
    app.add_url_rule('/prediction_output', view_func=machine_learning_controller.prediction_output_file,
                     methods=['GET', 'POST'])

    app.add_url_rule('/scheduler', view_func=scheduler_controller.scheduler_index, methods=['GET', 'POST'])
    app.add_url_rule('/scheduler_ajax', view_func=scheduler_controller.scheduler_ajax_index, methods=['GET', 'POST'])
    app.add_url_rule('/add_job_at_specific_time', view_func=scheduler_controller.add_job_at_specific_time,
                     methods=['GET', 'POST'])
    app.add_url_rule('/add_job_within_a_day', view_func=scheduler_controller.add_job_within_a_day,
                     methods=['GET', 'POST'])
    app.add_url_rule('/add_job_on_week_day', view_func=scheduler_controller.add_job_in_week_day,
                     methods=['GET', 'POST'])
    app.add_url_rule('/remove_existing_job', view_func=scheduler_controller.remove_existing_job,
                     methods=['GET', 'POST'])

    # port = int(os.getenv("PORT", 5001))
except Exception as e:
    print(e)
finally:
    on_exit_app()

def begin_mlflow_server():
    try:
        out = subprocess.run('mlflow ui -p 5005', shell=True)
    except Exception as e:
        raise e

if __name__ == "__main__":
    #th = Thread(target=begin_mlflow_server)
    #th.start()
    app.run(host='0.0.0.0', port=5000)
