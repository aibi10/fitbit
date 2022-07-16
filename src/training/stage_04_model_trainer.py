import sys

import joblib
import os
import argparse

from sklearn.impute import KNNImputer

from src.utility import create_directory_path
import mlflow
import numpy as np
import pandas as pd
from src.utility import get_logger_object_of_training
from sklearn.model_selection import train_test_split
from src.utility import read_params, class_for_name
from sklearn.preprocessing import Normalizer, MinMaxScaler
from sklearn.metrics import mean_squared_error, confusion_matrix, classification_report
from sklearn.metrics import r2_score
from sklearn.preprocessing import StandardScaler
import seaborn as sns
import matplotlib.pyplot as plt
import uuid
from sklearn.model_selection import GridSearchCV
from webapp.entity_layer.model_factory.model_factory import ModelFactory
from sklearn.metrics import f1_score, recall_score, precision_score, roc_auc_score

from webapp.exception_layer.generic_exception.generic_exception import GenericException
from webapp.entity_layer.model_factory.kmean_clustering import KMeansClustering

log_collection_name = "training_model"


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


    def is_null_present(self, data, null_value_path):
        """
        data: dataframe to check null value
        null_value_path: null value information will be saved into null value path
        =========================================================
        return True/False if null value present or not
        """
        self.logger.log('Entered the is_null_present method of the Preprocessor class')
        null_present = False
        try:
            null_counts = data.isna().sum()  # check for the count of null values per column
            for i in null_counts:
                if i > 0:
                    null_present = True
                    break
            if null_present:  # write the logs to see which columns have null values
                dataframe_with_null = pd.DataFrame()
                dataframe_with_null['columns'] = data.columns
                dataframe_with_null['missing values count'] = np.asarray(data.isna().sum())
                create_directory_path(null_value_path)
                dataframe_with_null.to_csv(os.path.join(null_value_path, "null_values.csv"))
                # storing the null column information to file
            self.logger.log("Finding missing values is a success.Data written to the null values file. Exited the "
                            "is_null_present method of the Preprocessor class")
            return null_present
        except Exception as e:
            generic_exception = GenericException(
                "Error occurred in module [{0}] class [{1}] method [{2}]"
                    .format(self.__module__, ModelTrainer.__name__,
                            self.is_null_present.__name__))
            raise Exception(generic_exception.error_message_detail(str(e), sys)) from e


class ModelTrainer:

    def __init__(self, config, logger, is_log_enable):
        try:
            self.logger = logger
            self.logger.is_log_enable = is_log_enable
            self.config = config
            self.training_file_path = self.config['artifacts']['training_data']['training_file_from_db']
            self.master_csv = self.config['artifacts']['training_data']['master_csv']
            self.target_columns = self.config['target_columns']['columns']
            self.test_size = self.config['base']['test_size']
            self.random_state = self.config['base']['random_state']
            self.plot = self.config['artifacts']['training_data']['plots']
            self.model_path = config['artifacts']['model']['model_path']
            self.null_value_file_path = config['artifacts']['training_data']['null_value_info_file_path']
            self.model_factory = ModelFactory()
            self.scaler_path = self.config['artifacts']['training_data']['scaler_path']
        except Exception as e:
            generic_exception = GenericException(
                "Error occurred in module [{0}] class [{1}] method [{2}]"
                    .format(self.__module__, ModelTrainer.__name__,
                            self.__init__.__name__))
            raise Exception(generic_exception.error_message_detail(str(e), sys)) from e

    def residual_plot(self, y_true, y_pred, title, xlabel, ylabel):
        try:
            y_true = np.array(y_true).reshape(-1)
            y_pred = np.array(y_pred).reshape(-1)
            create_directory_path(self.plot, is_recreate=False)
            sns.scatterplot(x=y_true, y=y_pred)

            plt.title(title)
            plt.xlabel(xlabel)
            plt.ylabel(ylabel)
            path = os.path.join(self.plot, f"scatter_plot_{title}.png")
            plt.savefig(path)
            mlflow.log_artifact(path)
            plt.clf()
        except Exception as e:
            generic_exception = GenericException(
                "Error occurred in module [{0}] class [{1}] method [{2}]"
                    .format(self.__module__, ModelTrainer.__name__,
                            self.residual_plot.__name__))
            raise Exception(generic_exception.error_message_detail(str(e), sys)) from e

    def save_regression_metric_data(self, model, x, y_true, title):
        try:
            y_pred = model.predict(x)
            y_true = np.array(y_true).reshape(-1)
            y_pred = np.array(y_pred).reshape(-1)
            rmse = np.sqrt(mean_squared_error(y_true, y_pred))
            mlflow.log_metric(f"{title}root_mean_squared_error", rmse)
            r_squared_score = r2_score(y_true, y_pred)
            mlflow.log_metric(f"{title}r_squared_score", r_squared_score)
            msg = f"{title} R squared score: {r_squared_score:.3%}"
            self.logger.log(msg)
            print(msg)
            msg = f"{title} Root mean squared error: {rmse:.3}"
            self.logger.log(msg)
            print(msg)
        except Exception as e:
            generic_exception = GenericException(
                "Error occurred in module [{0}] class [{1}] method [{2}]"
                    .format(self.__module__, ModelTrainer.__name__,
                            self.save_regression_metric_data.__name__))
            raise Exception(generic_exception.error_message_detail(str(e), sys)) from e

    def get_dataframe(self):
        try:
            master_file_path = os.path.join(self.training_file_path, self.master_csv)
            return pd.read_csv(master_file_path)
        except Exception as e:
            generic_exception = GenericException(
                "Error occurred in module [{0}] class [{1}] method [{2}]"
                    .format(self.__module__, ModelTrainer.__name__,
                            self.get_dataframe.__name__))
            raise Exception(generic_exception.error_message_detail(str(e), sys)) from e

    def data_preparation(self):
        try:
            data_frame = self.get_dataframe()
            preprocessing = DataPreProcessing(logger=self.logger, is_log_enable=self.logger.is_log_enable)
            input_features, target_features = data_frame.drop(self.target_columns, axis=1), data_frame[
                self.target_columns]


            input_features = preprocessing.scale_data(data=input_features, path=self.scaler_path,
                                                      is_dataframe_format_required=True,
                                                      )
            return input_features, target_features

        except Exception as e:
            generic_exception = GenericException(
                "Error occurred in module [{0}] class [{1}] method [{2}]"
                    .format(self.__module__, ModelTrainer.__name__,
                            self.data_preparation.__name__))
            raise Exception(generic_exception.error_message_detail(str(e), sys)) from e

    def split_dataset(self, input_features, target_features):
        try:
            x_train, x_test, y_train, y_test = train_test_split(input_features, target_features,
                                                                test_size=self.test_size,
                                                                random_state=self.random_state)
            x_train, x_val, y_train, y_val = train_test_split(x_train, y_train,
                                                              test_size=self.test_size,
                                                              random_state=self.random_state)
            return x_train, x_val, x_test, y_train, y_val, y_test
        except Exception as e:
            generic_exception = GenericException(
                "Error occurred in module [{0}] class [{1}] method [{2}]"
                    .format(self.__module__, ModelTrainer.__name__,
                            self.split_dataset.__name__))
            raise Exception(generic_exception.error_message_detail(str(e), sys)) from e

    def plot_confusion_matrix(self, truth_value, predicted_value, title):
        try:
            cm_labels = np.unique(truth_value)
            cm_array = confusion_matrix(truth_value, predicted_value)
            cm_array_df = pd.DataFrame(cm_array, index=cm_labels, columns=cm_labels)
            plt.figure(figsize=(15, 10))
            sns.heatmap(cm_array_df, annot=True, fmt='g', cmap='Blues', linewidths=1)
            plt.ylabel("Truth Value")
            plt.xlabel("Predicted Value")
            plt.title(f"Confusion Matrix {title}")
            create_directory_path(self.plot)
            path = os.path.join(self.plot, f"Confusion_Matrix_{title}.png")
            plt.savefig(path)
            mlflow.log_artifact(path)
            plt.clf()
        except Exception as e:
            generic_exception = GenericException(
                "Error occurred in module [{0}] class [{1}] method [{2}]"
                    .format(self.__module__, ModelTrainer.__name__,
                            self.plot_confusion_matrix.__name__))
            raise Exception(generic_exception.error_message_detail(str(e), sys)) from e

    def save_stacked_models(self, response):
        try:
            for response_data in response:
                model_details = response[response_data]
                for model_data in model_details:
                    model_obj = model_details[model_data]['best_model']
                    self.save_model(model_obj,
                                    type(model_details[model_data]['best_model']).__name__,
                                    f"{response_data}/{model_data}"
                                    )

        except Exception as e:
            generic_exception = GenericException(
                "Error occurred in module [{0}] class [{1}] method [{2}]"
                    .format(self.__module__, ModelTrainer.__name__,
                            self.save_stacked_models.__name__))
            raise Exception(generic_exception.error_message_detail(str(e), sys)) from e

    def generate_score_and_plot_graph(self, model, x, y, title):
        try:
            y_pred = model.predict(x)
            self.plot_confusion_matrix(y, pd.Series(y_pred), title)
            metrics_score = dict()
            calculated_score = model.score(x, y)
            calculated_f1_score = f1_score(y, y_pred, average="weighted")
            calculated_precision_score = precision_score(y, y_pred, average="weighted")
            report = classification_report(y_true=y, y_pred=y_pred, )
            print("*" * 50, title, "*" * 50)
            print(report)
            metrics_score[f'{title}_score'] = calculated_score
            metrics_score[f'{title}_f1_score'] = calculated_f1_score
            metrics_score[f'{title}_precision_score'] = calculated_precision_score
            # metrics_score[f'{title}_report']=report
            mlflow.log_metrics(metrics_score)
        except Exception as e:
            generic_exception = GenericException(
                "Error occurred in module [{0}] class [{1}] method [{2}]"
                    .format(self.__module__, ModelTrainer.__name__,
                            self.generate_score_and_plot_graph.__name__))
            raise Exception(generic_exception.error_message_detail(str(e), sys)) from e

    def begin_training(self):
        try:
            preprocessing = DataPreProcessing(logger=self.logger, is_log_enable=self.logger.is_log_enable)
            mlflow.set_experiment("ModelSearchBegin")
            model_factory = ModelFactory()
            with mlflow.start_run():
                self.logger.log("Preparing data for training")
                input_feature, target_feature = self.data_preparation()
                self.logger.log("Data has been prepared for traning")
                x_train, x_val, x_test, y_train, y_val, y_test = self.split_dataset(input_features=input_feature,
                                                                                    target_features=target_feature)
                self.logger.log("Initializing model for training")
                model_details = model_factory.initialize_model()
                self.logger.log("Started best model paramter search operation")
                response = model_factory.initiate_best_models_parameter_search(model_details, x_train, y_train)
                best_model = None
                best_score = 0
                for data in response:
                    if data['best_score'] > best_score:
                        best_score = data['best_score']
                        best_model = data['best_model']

                # training accuracy
                if best_model is None:
                    msg = "We haven't found any model."
                    self.logger.log(msg)
                    raise Exception(msg)

                # training score and graph plot
                self.save_regression_metric_data(model=best_model, x=x_train, y_true=y_train, title="training")

                # validation accuracy
                self.save_regression_metric_data(model=best_model, x=x_val, y_true=y_val, title="Validation")

                # Testing accuracy
                self.save_regression_metric_data(model=best_model, x=x_test, y_true=y_test, title="Testing")

                self.save_model(model=best_model, model_name=type(best_model).__name__)
                self.logger.log("Training done.")
                mlflow.end_run()
        except Exception as e:
            generic_exception = GenericException(
                "Error occurred in module [{0}] class [{1}] method [{2}]"
                    .format(self.__module__, ModelTrainer.__name__,
                            self.begin_training.__name__))
            raise Exception(generic_exception.error_message_detail(str(e), sys)) from e

    def save_model(self, model, model_name, intermediate_path=None):
        try:

            if intermediate_path is None:
                model_path = os.path.join(self.model_path)
            else:
                model_path = os.path.join(self.model_path, intermediate_path)
            create_directory_path(model_path, )
            model_full_path = os.path.join(model_path, f"{model_name}.sav")
            self.logger.log(f"Saving mode: {model_name} at path {model_full_path}")
            joblib.dump(model, model_full_path)
        except Exception as e:
            generic_exception = GenericException(
                "Error occurred in module [{0}] class [{1}] method [{2}]"
                    .format(self.__module__, ModelTrainer.__name__,
                            self.save_model.__name__))
            raise Exception(generic_exception.error_message_detail(str(e), sys)) from e


def train_main(config_path: str, datasource: str, is_logging_enable=True, execution_id=None, executed_by=None) -> None:
    try:
        logger = get_logger_object_of_training(config_path=config_path, collection_name=log_collection_name,
                                               execution_id=execution_id, executed_by=executed_by)

        logger.is_log_enable = is_logging_enable
        logger.log("Training begin.")
        config = read_params(config_path)
        model_trainer = ModelTrainer(config=config, logger=logger, is_log_enable=is_logging_enable)
        model_trainer.begin_training()
        logger.log("Training completed successfully.")

    except Exception as e:
        generic_exception = GenericException(
            "Error occurred in module [{0}] method [{1}]"
                .format(train_main.__module__,
                        train_main.__name__))
        raise Exception(generic_exception.error_message_detail(str(e), sys)) from e


if __name__ == '__main__':
    args = argparse.ArgumentParser()
    args.add_argument("--config", default=os.path.join("config", "params.yaml"))
    args.add_argument("--datasource", default=None)
    parsed_args = args.parse_args()
    print(parsed_args.config)
    print(parsed_args.datasource)
    train_main(config_path=parsed_args.config, datasource=parsed_args.datasource)
