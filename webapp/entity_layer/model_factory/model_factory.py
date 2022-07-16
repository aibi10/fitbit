import importlib

import mlflow
import numpy as np
import sklearn.metrics
import yaml
from webapp.exception_layer.generic_exception.generic_exception import GenericException as ModelFactoryException
import os
import sys


class ModelFactory:
    def __init__(self, config_path=None):
        try:
            if config_path is None:
                config_path = os.path.join('config', 'model.yaml')
            self.config = ModelFactory.read_params(config_path)
            # grid_search_cv formation began from here
            self.grid_search_cv_module = self.config['grid_search']['module']
            self.grid_search_class_name = self.config['grid_search']['class']
            self.grid_search_property_data = dict(self.config['grid_search']['params'])
            if 'stack' in self.config:
                self.stack_detail = self.config['stack']
            else:
                self.stack_detail = None
            self.model_detail = dict(self.config['model_selection'])

        except Exception as e:
            model_factory_exception = ModelFactoryException(
                "Error occurred in module [{0}] class [{1}] method [{2}]"
                    .format(self.__module__, ModelFactory.__name__,
                            self.__init__.__name__))
            raise Exception(model_factory_exception.error_message_detail(str(e), sys)) from e

    @staticmethod
    def update_property_of_class(instance_ref, property_data: dict):
        try:
            if not isinstance(property_data, dict):
                raise Exception("property_data paramter required to dictionary")
            print(property_data)
            for key, value in property_data.items():
                setattr(instance_ref, key, value)
            return instance_ref
        except Exception as e:
            model_factory_exception = ModelFactoryException(
                "Error occurred in module [{0}] class [{1}] method [{2}]"
                    .format(ModelFactory.__module__, ModelFactory.__name__,
                            ModelFactory.update_property_of_class.__name__))
            raise Exception(model_factory_exception.error_message_detail(str(e), sys)) from e

    @staticmethod
    def read_params(config_path: str) -> dict:
        try:
            with open(config_path) as yaml_file:
                config = yaml.safe_load(yaml_file)
            return config
        except Exception as e:
            model_factory_exception = ModelFactoryException(
                "Error occurred in module [{0}] class [{1}] method [{2}]"
                    .format(ModelFactory.__module__, ModelFactory.__name__,
                            ModelFactory.read_params.__name__))
            raise Exception(model_factory_exception.error_message_detail(str(e), sys)) from e

    @staticmethod
    def class_for_name(module_name, class_name):
        try:
            # load the module, will raise ImportError if module cannot be loaded
            module = importlib.import_module(module_name)
            # get the class, will raise AttributeError if class cannot be found
            class_ref = getattr(module, class_name)
            return class_ref
        except Exception as e:
            model_factory_exception = ModelFactoryException(
                "Error occurred in module [{0}] class [{1}] method [{2}]"
                    .format(ModelFactory.__module__, ModelFactory.__name__,
                            ModelFactory.class_for_name.__name__))
            raise Exception(model_factory_exception.error_message_detail(str(e), sys)) from e

    def execute_grid_search_operation(self, estimator, param_grid, input_feature, output_feature):
        """
        excute_grid_search_operation(): function will perform paramter search operation and
        it will return you the best optimistic  model with best paramter:
        estimator: Model object
        param_grid: dictionary of paramter to perform search operation
        input_feature: your all input features
        output_feature: Target/Dependent features
        ================================================================================
        return: Function will return a dictionary
        response = {"grid_search_obj": grid_search_cv,
                        "object": grid_search_cv.best_estimator_,
                        "best_parameters": grid_search_cv.best_params_,
                        "best_score": grid_search_cv.best_score_
                        }
        """
        try:
            # instantiating GridSearchCV class
            print("*"*50,f"training {type(estimator).__name__}","*"*50)
            grid_search_cv_ref = ModelFactory.class_for_name(module_name=self.grid_search_cv_module,
                                                             class_name=self.grid_search_class_name
                                                             )
            # updating property of GridSearchCV instance

            grid_search_cv = grid_search_cv_ref(estimator=estimator, param_grid=param_grid)
            grid_search_cv = ModelFactory.update_property_of_class(grid_search_cv,
                                                                   self.grid_search_property_data)

            grid_search_cv.fit(input_feature, output_feature)
            # mlflow.log_params(grid_search_cv.best_params_)

            response = {"grid_search_obj": grid_search_cv,
                        "best_model": grid_search_cv.best_estimator_,
                        "best_parameters": grid_search_cv.best_params_,
                        "best_score": grid_search_cv.best_score_
                        }
            return response
        except Exception as e:
            model_factory_exception = ModelFactoryException(
                "Error occurred in module [{0}] class [{1}] method [{2}]"
                    .format(self.__module__, ModelFactory.__name__,
                            self.execute_grid_search_operation.__name__))
            raise Exception(model_factory_exception.error_message_detail(str(e), sys)) from e

    def initialize_model(self):
        """
        This function will return a list of model details.
        Example:
            [{'model_serial_number':'value',
            'estimator':'estimator',
            'param_grid':'param_grid',
            'model_name':'model_name'
            },


            ]
        """
        try:
            response = []
            for model_ in self.model_detail.keys():
                response_item = dict()
                # creating model serial number
                response_item['model_serial_number'] = model_

                model_detail = self.model_detail[model_]
                model_obj_ref = ModelFactory.class_for_name(module_name=model_detail['module'],
                                                            class_name=model_detail['class']
                                                            )
                model_obj = model_obj_ref()

                if 'params' in model_detail:
                    model_obj_property_data = dict(model_detail['params'])
                    model_obj = ModelFactory.update_property_of_class(instance_ref=model_obj,
                                                                      property_data=model_obj_property_data)
                response_item['estimator'] = model_obj
                response_item['param_grid'] = model_detail['search_param_grid']
                response_item['model_name'] = f"{model_detail['module']}.{model_detail['class']}"
                response.append(response_item)
            return response
        except Exception as e:
            model_factory_exception = ModelFactoryException(
                "Error occurred in module [{0}] class [{1}] method [{2}]"
                    .format(self.__module__, ModelFactory.__name__,
                            self.initialize_model.__name__))
            raise Exception(model_factory_exception.error_message_detail(str(e), sys)) from e

    def initiate_best_model_parameter_search(self, estimator, param_grid, input_feature, output_feature):
        """

        initiate_best_model_parameter_search(): function will perform paramter search operation and
        it will return you the best optimistic  model with best paramter:
        estimator: Model object
        param_grid: dictionary of paramter to perform search operation
        input_feature: your all input features
        output_feature: Target/Dependent features
        ================================================================================
        return: Function will return a dictionary
        response = {"grid_search_obj": grid_search_cv,
                        "object": grid_search_cv.best_estimator_,
                        "best_parameters": grid_search_cv.best_params_,
                        "best_score": grid_search_cv.best_score_
                        }

        """
        try:
            return self.execute_grid_search_operation(estimator=estimator,
                                                      param_grid=param_grid,
                                                      input_feature=input_feature,
                                                      output_feature=output_feature)
        except Exception as e:
            model_factory_exception = ModelFactoryException(
                "Error occurred in module [{0}] class [{1}] method [{2}]"
                    .format(self.__module__, ModelFactory.__name__,
                            self.initiate_best_model_parameter_search.__name__))
            raise Exception(model_factory_exception.error_message_detail(str(e), sys)) from e

    def initiate_best_models_parameter_search(self, model_details: list, input_feature, output_feature):
        """
        Paramteres:
        ================================================================================================
        1. model_details: It accept list of model detail return from function initialize_model of class ModelFactory
        example:[{'model_serial_number':'value',
            'estimator':'estimator',
            'param_grid':'param_grid',
            'model_name':'model_name'
            },


            ]
        2.input_feature
        3.output_feature
        ================================================================================================
        function will return list of best models.
        [{"grid_search_obj": grid_search_cv,
        "best_model": grid_search_cv.best_estimator_,
        "best_parameters": grid_search_cv.best_params_,
        "best_score": grid_search_cv.best_score_,
        'model_serial_number':'value',
        'estimator':'estimator',
        'param_grid':'param_grid',
        'model_name':'model_name'}
        ]
        """
        try:
            response = []
            for model_detail in model_details:
                result = self.initiate_best_model_parameter_search(estimator=model_detail['estimator'],
                                                                   param_grid=model_detail['param_grid'],
                                                                   input_feature=input_feature,
                                                                   output_feature=output_feature
                                                                   )
                result.update(model_detail)
                response.append(result)
            return response
        except Exception as e:
            model_factory_exception = ModelFactoryException(
                "Error occurred  in module [{0}] class [{1}] method [{2}]"
                    .format(self.__module__, ModelFactory.__name__,
                            self.initiate_best_models_parameter_search.__name__))
            raise Exception(model_factory_exception.error_message_detail(str(e), sys)) from e

    def get_model_detail(self, model_details, model_serial_number):
        try:
            for model_data in model_details:
                if model_data['model_serial_number'] == model_serial_number:
                    return model_data
        except Exception as e:
            raise e

    def use_stacked_model(self, input_feature, output_feature, input_val_set, output_val_set,
                          input_test_set, output_test_set):
        try:
            stacked_model_response = dict()
            if self.stack_detail is None:
                raise Exception("model.yaml file does not have stacking detail")
            print(self.stack_detail)
            model_details = self.initialize_model()
            x, y, x_test, y_test = input_feature, output_feature, input_test_set, output_test_set

            for layer_detail in self.stack_detail:
                stacked_model_response[layer_detail] = dict()
                predict_val, predict_test = [], []
                new_input_feature = None
                new_input_test_feature = None
                for model_serial_number in self.stack_detail[layer_detail]:
                    model_data = self.get_model_detail(model_details=model_details,
                                                       model_serial_number=model_serial_number)
                    response = self.initiate_best_model_parameter_search(estimator=model_data['estimator'],
                                                                         param_grid=model_data['param_grid'],
                                                                         input_feature=x,
                                                                         output_feature=y
                                                                         )
                    stacked_model_response[layer_detail][model_serial_number] = response

                    if self.config['stack_output_layer'] != layer_detail:
                        ##validation set
                        pred_val_set = response['best_model'].predict(input_val_set)
                        predict_val.append(pred_val_set)
                        new_input_feature = np.column_stack(predict_val)

                x, y, x_test, y_test = new_input_feature, output_val_set, new_input_test_feature, y_test
                print("=" * 10)
            return stacked_model_response
        except Exception as e:
            raise e
