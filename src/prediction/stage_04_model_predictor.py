import sys

import joblib
import os
import argparse

import numpy as np
from sklearn.impute import KNNImputer

from src.utility import create_directory_path
import pandas as pd
from src.utility import get_logger_object_of_prediction
from src.utility import read_params, class_for_name
from sklearn.preprocessing import Normalizer, MinMaxScaler, StandardScaler

from webapp.exception_layer.generic_exception.generic_exception import GenericException
from src.training.stage_04_model_trainer import ModelTrainer

log_collection_name = "prediction_model"


class DataPreProcessing:
    def __init__(self, logger, is_log_enable=True):
        try:
            self.logger = logger
            self.logger.is_log_enable = is_log_enable
        except Exception as e:
            generic_exception = GenericException(
                "Error occurred in module [{0}] class [{1}] method [{2}]"
                    .format(self.__module__, DataPreProcessing.__name__,
                            self.__init__.__name__))
            raise Exception(generic_exception.error_message_detail(str(e), sys)) from e

    def scale_data(self, data, path, is_dataframe_format_required=False, is_new_scaling=True):
        """
        data: dataframe to perform scaling
        path: path to save scaler object
        get_dataframe_format: default scaled output will be return as ndarray but if is true you will get
        dataframe format
        is_new_scaling: default it will create new scaling object and perform transformation.
        if it is false it will load scaler object from mentioned path paramter
        """
        try:
            path = os.path.join(path)
            if not is_new_scaling:
                if os.path.exists(path):
                    scaler = joblib.load(os.path.join(path, "scaler.sav"))
                    output = scaler.transform(data)
                else:
                    raise Exception(f"Scaler object is not found at path: {path}")
            else:
                scaler = StandardScaler()
                output = scaler.fit_transform(data)
                create_directory_path(path)
                joblib.dump(scaler, os.path.join(path, "scaler.sav"))
            if is_dataframe_format_required:
                output = pd.DataFrame(output, columns=data.columns)
            return output
        except Exception as e:
            generic_exception = GenericException(
                "Error occurred in module [{0}] class [{1}] method [{2}]"
                    .format(self.__module__, DataPreProcessing.__name__,
                            self.scale_data.__name__))
            raise Exception(generic_exception.error_message_detail(str(e), sys)) from e


class Predictor:

    def __init__(self, config, logger, is_log_enable):
        try:
            self.logger = logger
            self.logger.is_log_enable = is_log_enable
            self.config = config
            self.prediction_file_path = self.config['artifacts']['prediction_data']['prediction_file_from_db']
            self.master_csv = self.config['artifacts']['prediction_data']['master_csv']
            self.model_path = self.config['artifacts']['model']['model_path']
            self.prediction_output_file_path = self.config['artifacts']['prediction_data'][
                'prediction_output_file_path']
            self.prediction_file_name = self.config['artifacts']['prediction_data']['prediction_file_name']
            self.target_columns = self.config['target_columns']['columns']
            self.null_value_file_path = config['artifacts']['training_data']['null_value_info_file_path']
            self.scaler_path = self.config['artifacts']['training_data']['scaler_path']
        except Exception as e:
            generic_exception = GenericException(
                "Error occurred in module [{0}] class [{1}] method [{2}]"
                    .format(self.__module__, Predictor.__name__,
                            self.__init__.__name__))
            raise Exception(generic_exception.error_message_detail(str(e), sys)) from e

    def get_dataframe(self):
        try:
            master_file_path = os.path.join(self.prediction_file_path, self.master_csv)
            return pd.read_csv(master_file_path)
        except Exception as e:
            generic_exception = GenericException(
                "Error occurred in module [{0}] class [{1}] method [{2}]"
                    .format(self.__module__, Predictor.__name__,
                            self.get_dataframe.__name__))
            raise Exception(generic_exception.error_message_detail(str(e), sys)) from e

    def get_clustered_data(self, data):
        try:
            cluster_model = self.load_model(intermediate_path="cluster")
            cluster = cluster_model.predict(data)
            return cluster
        except Exception as e:
            generic_exception = GenericException(
                "Error occurred in module [{0}] class [{1}] method [{2}]"
                    .format(self.__module__, Predictor.__name__,
                            self.get_clustered_data.__name__))
            raise Exception(generic_exception.error_message_detail(str(e), sys)) from e

    def data_preparation(self):
        try:

            input_features = self.get_dataframe()
            input_features_without_scale = input_features
            preprocessing = DataPreProcessing(logger=self.logger, is_log_enable=self.logger.is_log_enable)

            input_features = preprocessing.scale_data(data=input_features, path=self.scaler_path,
                                                      is_dataframe_format_required=True, is_new_scaling=False
                                                      )

            return input_features, input_features_without_scale

        except Exception as e:
            generic_exception = GenericException(
                "Error occurred in module [{0}] class [{1}] method [{2}]"
                    .format(self.__module__, Predictor.__name__,
                            self.data_preparation.__name__))
            raise Exception(generic_exception.error_message_detail(str(e), sys)) from e

    def get_model_path_list(self):
        try:
            path = os.path.join('config', 'model.yaml')

            config_data = read_params(path)
            model_path = []
            for data in config_data['stack']:
                layer = config_data['stack'][data]
                for model in layer:
                    path = f"{data}/{model}"
                    model_path.append(path)
            return model_path
        except Exception as e:
            generic_exception = GenericException(
                "Error occurred in module [{0}] class [{1}] method [{2}]"
                    .format(self.__module__, Predictor.__name__,
                            self.get_model_path_list.__name__))
            raise Exception(generic_exception.error_message_detail(str(e), sys)) from e

    def load_model(self, intermediate_path=None):
        try:
            if intermediate_path is not None:
                model_path = os.path.join(self.model_path, intermediate_path)
            else:
                model_path = self.model_path
            if not os.path.exists(model_path):
                raise Exception(f"Model directory: {model_path} is not found.")
            model_names = os.listdir(model_path)
            if len(model_names) != 1:
                raise Exception(f"We have expected only one model instead we found {len(model_names)}")
            model_name = model_names[0]
            return joblib.load(os.path.join(model_path, model_name))
        except Exception as e:
            generic_exception = GenericException(
                "Error occurred in module [{0}] class [{1}] method [{2}]"
                    .format(self.__module__, Predictor.__name__,
                            self.load_model.__name__))
            raise Exception(generic_exception.error_message_detail(str(e), sys)) from e

    def predict(self):
        try:
            self.logger.log("Data preparation began.")
            data, data_without_scale = self.data_preparation()
            self.logger.log("Data preparation completed.")
            model = self.load_model()

            data_without_scale['Predicted_Feature'] = model.predict(data)
            prediction_output = data_without_scale
            create_directory_path(self.prediction_output_file_path)
            output_file_path = os.path.join(self.prediction_output_file_path, self.prediction_file_name)
            if prediction_output is not None:
                prediction_output.to_csv(output_file_path, index=None, header=True)
                self.logger.log(f"Prediction file has been generated at {output_file_path}")
        except Exception as e:
            generic_exception = GenericException(
                "Error occurred in module [{0}] class [{1}] method [{2}]"
                    .format(self.__module__, Predictor.__name__,
                            self.predict.__name__))
            raise Exception(generic_exception.error_message_detail(str(e), sys)) from e


def predict_main(config_path: str, datasource: str, is_logging_enable=True, execution_id=None,
                 executed_by=None) -> None:
    try:
        logger = get_logger_object_of_prediction(config_path=config_path, collection_name=log_collection_name,
                                                 execution_id=execution_id, executed_by=executed_by)

        logger.is_log_enable = is_logging_enable
        logger.log("Prediction begin.")
        config = read_params(config_path)
        predictor = Predictor(config=config, logger=logger, is_log_enable=is_logging_enable)
        predictor.predict()
        logger.log("Prediction completed successfully.")

    except Exception as e:
        generic_exception = GenericException(
            "Error occurred in module [{0}] method [{1}]"
                .format(predict_main.__module__,
                        predict_main.__name__))
        raise Exception(generic_exception.error_message_detail(str(e), sys)) from e


if __name__ == '__main__':
    args = argparse.ArgumentParser()
    args.add_argument("--config", default=os.path.join("config", "params.yaml"))
    args.add_argument("--datasource", default=None)
    parsed_args = args.parse_args()
    print(parsed_args.config)
    print(parsed_args.datasource)

    predict_main(config_path=parsed_args.config, datasource=parsed_args.datasource)
