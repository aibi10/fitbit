from webapp.thread_layer.train_model_thread.train_model_thread import TrainModelThread
from webapp.thread_layer.predict_from_model_thread.predict_from_model_thread import PredictFromModelThread
from webapp.logging_layer.logger.log_exception import LogExceptionDetail
from webapp.logging_layer.logger.log_request import LogRequest
from webapp.data_access_layer.mongo_db.mongo_db_atlas import MongoDBOperation
from webapp.project_library_layer.initializer.initializer import Initializer
from webapp.entity_layer.scheduler.scheduler_storage import SchedulerStorage
class ScheduleTask:
    def __init__(self,project_id,executed_by,execution_id):
        self.project_id = project_id
        self.executed_by = executed_by
        self.execution_id = execution_id
        self.log_writer = LogRequest(executed_by=self.executed_by,execution_id=self.execution_id)
        self.train_predict_thread_obj_ref=None
        self.sch_storage=SchedulerStorage()




    def start_training(self):
        try:
            train_predict_thread_obj_ref = TrainModelThread(project_id=self.project_id, executed_by=self.executed_by,
                                                            execution_id=self.execution_id,
                                                            log_writer=self.log_writer)
            response = train_predict_thread_obj_ref.get_running_status_of_training_thread()
            previous_execution_thread_status = None
            if response is not None:
                if 'is_running' in response:
                    previous_execution_thread_status = response['is_running']
            if previous_execution_thread_status:
                pass


            if previous_execution_thread_status is None or not previous_execution_thread_status:
                self.sch_storage.update_job_record(job_id=self.execution_id,status='running')
                train_predict_thread_obj_ref.start()

                train_predict_thread_obj_ref.join()
                self.sch_storage.update_job_record(job_id=self.execution_id,status='successful')

        except Exception as e:
            self.sch_storage.update_job_record(job_id=self.execution_id, status='failed')
            if self.log_writer is not None:
                self.log_writer.log_stop({'status': False, 'error_message': str(e)})
            log_exception = LogExceptionDetail(self.log_writer.executed_by, self.log_writer.execution_id)
            log_exception.log(str(e))


    def start_prediction(self):
        try:
            train_predict_thread_obj_ref = PredictFromModelThread(project_id=self.project_id, executed_by=self.executed_by,
                                                                  execution_id=self.execution_id,
                                                                  log_writer=self.log_writer)
            response = train_predict_thread_obj_ref.get_running_status_of_prediction_thread()
            previous_execution_thread_status = None
            if response is not None:
                if 'is_running' in response:
                    previous_execution_thread_status = response['is_running']
            if previous_execution_thread_status:
                pass


            if previous_execution_thread_status is None or not previous_execution_thread_status:
                self.sch_storage.update_job_record(job_id=self.execution_id, status='running')
                train_predict_thread_obj_ref.start()

                train_predict_thread_obj_ref.join()
                self.sch_storage.update_job_record(job_id=self.execution_id, status='successful')
        except Exception as e:
            self.sch_storage.update_job_record(job_id=self.execution_id, status='failed')
            if self.log_writer is not None:
                self.log_writer.log_stop({'status': False, 'error_message': str(e)})
            log_exception = LogExceptionDetail(self.log_writer.executed_by, self.log_writer.execution_id)
            log_exception.log(str(e))



    def start_training_prediction_both(self):
        try:
            self.start_training()
            self.start_prediction()

        except Exception as e:
            if self.log_writer is not None:
                self.log_writer.log_stop({'status': False, 'error_message': str(e)})
            log_exception = LogExceptionDetail(self.log_writer.executed_by, self.log_writer.execution_id)
            log_exception.log(str(e))
