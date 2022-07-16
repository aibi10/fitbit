import argparse
import sys
import os
from src.prediction.stage_04_model_predictor import Predictor
from src.utility import read_params, get_logger_object_of_prediction
from webapp.exception_layer.generic_exception.generic_exception import GenericException

log_collection_name="model_predictor"

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
