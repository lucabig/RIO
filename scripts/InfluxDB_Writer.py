"""
    Generic class to create a time-series InfluxDB database and to write data points on it
"""



import pandas as pd
import numpy as np
from datetime import datetime
import calendar
import json
from influxdb import InfluxDBClient
from influxdb import DataFrameClient



class InfluxDB_Writer:

    """ 
        Data writer for a time-series InfluxDB database

        Attributes:

            database_name (str)
            data_frame_client_flag (bool): True for DataFrameClient (queries return a pandas dataframe), False for InfluxDBClient (queries return a dictionary)
            ip_address (str)
            port (str)
            username (str)
            password (str)
            databases (list)
            measurements (dict): key is database, value is measurement 
            client: InfluxDB client server

        Methods:


    """

    def __init__(self, database_name, ip_address, port, 
                       user_name = None, password = None, data_frame_client_flag = True):

        """ 
            Args for initialization:
                database_name (str)
                ip_address, port, user_name (optional), password (optional) (str): for access to the InfluxDB client where the database has to be written
                data_frame_client_flag (bool): True for DataFrameClient, False for InfluxDBClient 
        """

        # Input
        self.database_name = database_name
        self.data_frame_client_flag = data_frame_client_flag
        self.ip_address = ip_address
        self.port = str(port)
        self.user_name = user_name
        self.password = password


        # Own attributes
        self.databases = []
        self.measurements = dict()
        self.client = None



        # Connect to client
        self.client = self._connect_to_client(data_frame_client_flag)



        # Create database if it does not exist
        if self.database_name:

            self.create_database(self.database_name)



        # Read present databases
        self.databases = self._get_databases()



        # Read measurements (for each database or for input database)
        self.measurements = self._get_measurements()
        


    def _connect_to_client(self, data_frame_client_flag):

        if not data_frame_client_flag:

            return InfluxDBClient(self.ip_address, self.port, self.user_name, self.password)

        else:

            return DataFrameClient(self.ip_address, self.port, self.user_name, self.password)



    def _get_databases(self):

        myquery = self.client.query('SHOW DATABASES')
        
        databases = [d['name'] for d in myquery.get_points() if d['name'] != '_internal']

        return databases

     

    def _get_measurements(self):

        for database in self.databases:

            myquery = self.client.query('SHOW measurements', database = database)

            measurements_list = [i['name'] for i in myquery.get_points()]

            self.measurements[database] = measurements_list

        return self.measurements



    def _pre_process_data(self, data_dict):

        # Create pandas dataframe from input dict
        data_df = pd.DataFrame.from_dict(data_dict)

        # Index time
        data_df['time'] = pd.to_datetime(data_df['time'])

        data_df = data_df.set_index('time')

        # Drop rows with Nan values
        if not data_df[data_df.isna().any(axis=1)].empty:

            data_df = data_df.dropna()

        return data_df



    def read_labels(labels_filepath, json_mode = True):

        if json_mode:

            with open(labels_filepath) as json_file:

                self.labels = json.load(json_file)



    def delete_all_databases(self):

        for db in self.databases:

            if db != '_internal':

                print("Delete database: " + db)

                self.client.drop_database(db)



    def delete_database(self, database_name):

        if database_name in self.databases:

            print("Delete database: " + database_name)

            self.client.drop_database(database_name)



    def create_database(self, database_name):

        if database_name not in self.databases:
    
           self.client.create_database(database_name) 

        else:

            print('Database ' + database_name + ' is already exisiting!')



    def create_new_measurement(self, database_name, measurement_name, 
                                     data_dict, tag_names, 
                                     delete_previous_measurement = False):

        if database_name not in self.databases:

            self.create_database(database_name)

        if delete_previous_measurement:

            if measurement_name in self.measurements[database_name]:
        
                self.client.drop_measurement(measurement_name) # Not working!

        # Pre-process dict to pandas dataframe
        data_df = self._pre_process_data(data_dict)          

        # Add points to the new measurement
        self.client.write_points(data_df, measurement_name, 
                                 tag_columns = tag_names, database = database_name, 
                                 protocol='json')