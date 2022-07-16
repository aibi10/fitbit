import os
import sys

import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from kneed import KneeLocator
import numpy as np
from src.utility import create_directory_path
from webapp.exception_layer.generic_exception.generic_exception import GenericException
from webapp.entity_layer.model_factory.model_factory import ModelFactory



class KMeansClustering:

    def __init__(self, logger, config_path=None, is_log_enable=True):
        try:
            if config_path is None:
                config_path = os.path.join('config', 'model.yaml')
            self.logger = logger
            self.logger.is_log_enable = is_log_enable
            self.config = ModelFactory.read_params(config_path=config_path)
            self.params = self.config['clustering']['params']
            self.minimum_cluster = self.config['clustering']['search_cluster']['start']
            self.maximum_cluster = self.config['clustering']['search_cluster']['stop']
        except Exception as e:
            generic_exception = GenericException(
                "Error occurred in module [{0}] class [{1}] method [{2}]"
                    .format(self.__module__, KMeansClustering.__name__,
                            self.__init__.__name__))
            raise Exception(generic_exception.error_message_detail(str(e), sys)) from e

    def elbow_plot(self, data, elbow_plot_path):
        """
        Method Name: elbow_plot
        Description: This method saves the plot to decide the optimum number of clusters to the file.
        Output: A picture saved to the directory
        On Failure: Raise Exception
                        """

        try:
            create_directory_path(elbow_plot_path)
            self.logger.log('Entered the elbow_plot method of the KMeansClustering class')
            wcss = []  # initializing an empty list

            for i in range(self.minimum_cluster, self.maximum_cluster + 1):
                kmeans = KMeans(n_clusters=i, init=self.params['init'], random_state=self.params['random_state'])
                # initializing the KMeans object
                kmeans.fit(data)  # fitting the data to the KMeans Algorithm
                wcss.append(kmeans.inertia_)
            plt.plot(range(self.minimum_cluster, self.maximum_cluster + 1), wcss)
            # creating the graph between WCSS and the number of clusters
            plt.title('The Elbow Method')
            plt.xlabel('Number of clusters')
            plt.ylabel('WCSS')
            # plt.show()
            plt.savefig(os.path.join(elbow_plot_path, "K-Means_Elbow.PNG"))  # saving the elbow plot locally
            # finding the value of the optimum cluster programmatically
            plt.clf()
            kn = KneeLocator(range(self.minimum_cluster, self.maximum_cluster+1), wcss, curve='convex',
                             direction='decreasing')
            self.logger.log(f'The optimum number of clusters is: {kn.knee}.'
                            f' Exited the elbow_plot method of the KMeansClustering class')
            return kn.knee

        except Exception as e:
            generic_exception = GenericException(
                "Error occurred in module [{0}] class [{1}] method [{2}]"
                    .format(self.__module__, KMeansClustering.__name__,
                            self.elbow_plot.__name__))
            raise Exception(generic_exception.error_message_detail(str(e), sys)) from e

    def create_clusters(self, data, number_of_cluster):

        """
                                       Method Name: create_clusters
                                       Description: Create a new dataframe consisting of the cluster information.
                                       Output: A datframe with cluster column
                                       On Failure: Raise Exception
        """

        try:
            self.logger.log('Entered the create_clusters method of the KMeansClustering class')
            data = data.copy()
            kmeans = KMeans(n_clusters=number_of_cluster, init=self.params['init'],
                            random_state=self.params['random_state'])
            # self.data = self.data[~self.data.isin([np.nan, np.inf, -np.inf]).any(1)]
            cluster = kmeans.fit_predict(data)  # divide data into clusters

            data['cluster'] = cluster  # create a new column in dataset for storing the cluster information
            self.logger.log(f'successfully created {number_of_cluster} clusters. Exited the create_clusters method of '
                            f'the KMeansClustering class')
            return {'clustered_data': data, 'model': kmeans}
        except Exception as e:
            generic_exception = GenericException(
                "Error occurred in module [{0}] class [{1}] method [{2}]"
                    .format(self.__module__, KMeansClustering.__name__,
                            self.create_clusters.__name__))
            raise Exception(generic_exception.error_message_detail(str(e), sys)) from e
