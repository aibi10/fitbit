from src.entryPoint import begin_training, begin_prediction
import os
from webapp.data_access_layer.mongo_db.mongo_db_atlas import MongoDBOperation
from src.utility import read_params

class ExperimentRunner:
    def __init__(self, project_id=None, execution_id=None, executed_by=None):
        self.execution_id = execution_id
        self.executed_by = executed_by
        self.project_id = project_id
        self.mg_db = MongoDBOperation()

    def clean_dvc_file_and_log(self):
        try:

            if os.path.exists("dvc.lock"):
                os.remove(os.path.join("dvc.lock"))
            if os.path.exists("dvc.yaml"):
                os.remove(os.path.join("dvc.yaml"))
        except Exception as e:
            raise e

    def start_training(self):
        try:
            self.clean_dvc_file_and_log()
            response = begin_training(execution_id=self.execution_id,
                                      executed_by=self.executed_by)

            return response
        except Exception as e:
            return {'status': False, 'message': 'Training failed due to .' + str(e)}

    def start_prediction(self):
        try:
            self.clean_dvc_file_and_log()
            response = begin_prediction(execution_id=self.execution_id,
                                        executed_by=self.executed_by)
            return response

        except Exception as e:

            return {'status': False, 'message': 'Prediction failed due to .' + str(e)}

config=read_params(config_path=os.path.join('config','params.yaml'))
project_id=config['base']['project_id']
project_train_and_prediction_mapper = [
    {
        'project_id': project_id,
        'ExperimentRunner': ExperimentRunner
    }
]


def get_experiment_class_reference(project_id):
    try:
        for i in project_train_and_prediction_mapper:
            print(i['project_id'],project_id)
            if i['project_id'] == project_id:
                return i['ExperimentRunner']

    except Exception as e:
        raise e
