from webapp.data_access_layer.mongo_db.mongo_db_atlas import MongoDBOperation
from webapp.entity_layer.project.project import Project
from webapp.entity_layer.project.project_configuration import ProjectConfiguration
from webapp.project_library_layer.initializer.initializer import Initializer
from webapp.exception_layer.predict_model_exception.predict_model_exception import PredictFromModelException
from webapp.project_library_layer.project_training_prediction_mapper.project_training_prediction_mapper import \
    get_experiment_class_reference

import sys


class PredictFromModel:

    def __init__(self, project_id, executed_by, execution_id,):
        try:
            self.project_id = project_id
            self.executed_by = executed_by
            self.execution_id = execution_id
            self.project_detail = Project()
            self.project_config = ProjectConfiguration()
            self.initializer = Initializer()

        except Exception as e:
            predict_from_model_exception = PredictFromModelException(
                "Failed during instantiation in module [{0}] class [{1}] method [{2}]"
                    .format(self.__module__, PredictFromModel.__name__,
                            self.__init__.__name__))
            raise Exception(predict_from_model_exception.error_message_detail(str(e), sys)) from e

    def prediction_from_model(self):
        try:


            if self.project_id is None:
                raise Exception("Project id not found")
            project_detail = self.project_detail.get_project_detail(project_id=self.project_id)
            if not project_detail['status']:
                project_detail.update(
                    {'message_status': 'info', 'project_id': self.project_id})
                return project_detail
            project_detail=project_detail['project_detail']
            project_config_detail = self.project_config.get_project_configuration_detail(project_id=self.project_id)
            if not project_config_detail['status']:
                project_config_detail.update(
                    {'message_status': 'info', 'project_id': self.project_id})
                return project_config_detail
            if 'project_config_detail' in project_config_detail:
                project_config_detail = project_config_detail['project_config_detail']
            if project_config_detail is None:
                response = {'status': False, 'message': 'project configuration not found',
                            'message_status': 'info', 'project_id': self.project_id}

                return response
            prediction_file_path = Initializer().get_prediction_batch_file_path(project_id=self.project_id)
            cloud_storage = None
            if 'cloud_storage' in project_config_detail:
                cloud_storage = project_config_detail['cloud_storage']
            if cloud_storage is None:
                result = {'status': False,
                          'message': 'Cloud Storage location not found',
                          'message_status': 'info', 'project_id': self.project_id}

                return result
            experiment=get_experiment_class_reference(self.project_id)
            response=experiment(project_id=self.project_id,
                                execution_id=self.execution_id,
                                executed_by=self.executed_by).start_prediction()
            if response['status']:
                response = {'status': True, 'message': response['message'], 'is_failed': False,
                                    'message_status': 'info', 'project_id': self.project_id}
            else:
                response = {'status': False, 'message': response['message'], 'is_failed': True,
                    'message_status': 'info', 'project_id': self.project_id}
            return response
        except Exception as e:

            predict_from_model_exception = PredictFromModelException(
                "Failed during prediction from in module [{0}] class [{1}] method [{2}]"
                    .format(self.__module__, PredictFromModel.__name__,
                            self.prediction_from_model.__name__))
            raise Exception(predict_from_model_exception.error_message_detail(str(e), sys)) from e
