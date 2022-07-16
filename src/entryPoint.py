import logging
import os
import argparse
import shutil
import sys

from src.utility import get_logger_object_of_training, get_logger_object_of_prediction
from src.training.stage_00_data_loader import loader_main
from src.training.stage_01_data_validator import validation_main
from src.training.stage_02_data_transformer import transform_main
from src.training.stage_03_data_exporter import export_main
from src.training.stage_04_model_trainer import train_main

from src.prediction.stage_00_data_loader import loader_main as pred_loader_main
from src.prediction.stage_01_data_validator import validation_main as pred_validation_main
from src.prediction.stage_02_data_transformer import transform_main as pred_transform_main
from src.prediction.stage_03_data_exporter import export_main as pred_export_main
from src.prediction.stage_04_model_predictor import predict_main
from webapp.exception_layer.generic_exception.generic_exception import GenericException

collection_name = "main_pipeline"

def begin_training(execution_id, executed_by):
    try:
        args = dict()
        args['config'] = os.path.join("config", "params.yaml")
        logger = get_logger_object_of_training(config_path=args['config'],
                                               collection_name=collection_name,
                                               execution_id=execution_id,
                                               executed_by=executed_by
                                               )

        args['datasource'] = None
        parsed_args = args
        logger.log(f"dictionary created.{args}")
        logger.log(f"{parsed_args}")
        logger.log("Data loading begin..")

        loader_main(config_path=parsed_args['config'], datasource=parsed_args['datasource'], execution_id=execution_id,
                    executed_by=executed_by)
        logger.log("Data loading completed..")
        logger.log("Data validation began..")

        validation_main(config_path=parsed_args['config'], datasource=parsed_args['datasource'],
                        execution_id=execution_id,
                        executed_by=executed_by)
        logger.log("Data validation completed..")
        logger.log("Data transformation began..")

        transform_main(config_path=parsed_args['config'], datasource=parsed_args['datasource'],
                       execution_id=execution_id,
                       executed_by=executed_by)
        logger.log("Data transformation completed..")
        logger.log("Export oberation began..")

        export_main(config_path=parsed_args['config'], execution_id=execution_id,
                    executed_by=executed_by)
        logger.log("Export oberation completed..")
        logger.log("Training began..")

        train_main(config_path=parsed_args['config'], datasource=parsed_args['datasource'], execution_id=execution_id,
                   executed_by=executed_by)
        logger.log("Training completed")
        return {'status':True,'message':'Training completed successfully'}
    except Exception as e:
        generic_exception = GenericException(
            "Error occurred in module [{0}] method [{1}]"
                .format(begin_training.__module__,
                        begin_training.__name__))
        raise Exception(generic_exception.error_message_detail(str(e), sys)) from e


def begin_prediction(execution_id, executed_by):
    try:
        args = dict()
        args['config'] = os.path.join("config", "params.yaml")
        logger = get_logger_object_of_prediction(config_path=args['config'],
                                                 collection_name=collection_name,
                                                 execution_id=execution_id,
                                                 executed_by=executed_by
                                                 )
        args['datasource'] = None
        parsed_args = args
        logger.log(f"dictionary created.{args}")

        logger.log(f"{parsed_args}")
        logger.log("Data loading begin..")

        pred_loader_main(config_path=parsed_args['config'], datasource=parsed_args['datasource'],
                         execution_id=execution_id,
                         executed_by=executed_by
                         )
        logger.log("Data loading completed..")
        logger.log("Data validation began..")

        pred_validation_main(config_path=parsed_args['config'], datasource=parsed_args['datasource'],
                             execution_id=execution_id,
                             executed_by=executed_by
                             )
        logger.log("Data validation completed..")
        logger.log("Data transformation began..")

        pred_transform_main(config_path=parsed_args['config'], datasource=parsed_args['datasource'],
                            execution_id=execution_id,
                            executed_by=executed_by
                            )
        logger.log("Data transformation completed..")
        logger.log("Export oberation began..")

        pred_export_main(config_path=parsed_args['config'])
        logger.log("Export operation completed..")
        logger.log("Prediction began..")

        predict_main(config_path=parsed_args['config'], datasource=parsed_args['datasource'],
                     execution_id=execution_id,
                     executed_by=executed_by
                     )
        logger.log("Prediction completed")
        return {'status':True,'message':'Prediction completed successfully'}
    except Exception as e:
        generic_exception = GenericException(
            "Error occurred in module [{0}] method [{1}]"
                .format(begin_prediction.__module__,
                        begin_prediction.__name__))
        raise Exception(generic_exception.error_message_detail(str(e), sys)) from e


def clean():
    dvc_file = "dvc.yaml"
    if os.path.isfile(dvc_file):
        os.remove(dvc_file)
    else:
        print(f"previous {dvc_file} cleaned!")


def train():
    try:
        clean()
        shutil.copy("dvc_pipelines/training/dvc.yaml", "./")
        print("training dvc file copied")
        os.system("dvc repro")
    except Exception as e:
        raise e


def predict():
    try:
        clean()
        shutil.copy("dvc_pipelines/prediction/dvc.yaml", "./")

        os.system("dvc repro")
    except Exception as e:
        raise e


def main(training: str) -> None:
    if training == "True":
        print("training")
        train()
    else:
        print("predicting")
        predict()


if __name__ == '__main__':
    args = argparse.ArgumentParser()
    args.add_argument("--training", default=False)
    parsed_args = args.parse_args()
    print("started")
    main(training=parsed_args.training)
