"""
    Generic class to explore a time-series InfluxDB dataset and extract data from it
"""



import pandas as pd
import numpy as np
from datetime import datetime
import calendar
import json
from influxdb import InfluxDBClient
from influxdb import DataFrameClient



class InfluxDB_Reader:

    """ 
        Data reader of a time-series InfluxDB dataset

        Attributes:

            database_name (str)
            data_frame_client_flag (bool): True for DataFrameClient (queries return a pandas dataframe), False for InfluxDBClient (queries return a dictionary)
            ip_address (str)
            port (str)
            username (str)
            password (str)
            databases (list)
            measurements (dict): key is database, value is measurement 
            tags (nested dict): keys are database and measurement, values are tags
            fields (nested dict): keys are database and measurement, values are fields
            data_structure (nested dict): keys are database, measurement, tags and fields
            time_ranges (dict): keys are database, measurement, Start and End
            client: InfluxDB client server

        Methods:

            _get_databases()
            _get_measurements()
            _get_all_tag_key_value_pairs()
            _get_all_field_keys()
            _get_data_structure()
            _get_start_end_timestamps()
            _compose_query(database, measurement, select)
            _build_select_clause(select)
            _build_where_clause(where)
            _build_groupby_clause()
            _build_limit_clause()
            print_info()
            list_tag_keys(database, measurement)
            list_tag_values(database, measurement, tag_key)
            list_field_keys(database, measurement)
            show_all_series_for_single_measurement(database, measurement)
            get_time_range(measurement)
            get_start_end_timestamps(database, measurement)
            query_first_n_points(n, database, measurement, select)
            rfc3339TOUnixMS(time)
            rfc3339TOTime(time)
            query(measurement, select)
            
    """




    def __init__(self, ip_address, port, 
                       user_name = None, password = None, 
                       database_name = None, data_frame_client_flag = True):

        """ 
            Args for initialization:

                ip_address, port, user_name (optional), password (optional) (str): for access to the InfluxDB client where data are read
                database_name (optional) (str)
                data_frame_client_flag (bool): True for DataFrameClient (queries return a pandas dataframe), False for InfluxDBClient (queries return a dictionary)          
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
        self.tags = dict()
        self.fields = dict()
        self.data_structure = dict()
        self.client = None
        self.time_ranges = dict()




        # Connect to client
        self.client = self._connect_to_client(data_frame_client_flag)



        # Read databases: create two empy dictionaries for the tags and the fields for each database
        if self.database_name:

            self.databases = [database_name]

        else:

            self.databases = self._get_databases()


        # Read measurements (for each database or for input database): for each database, create a disctionary with keys given by measurements names
        self.measurements = self._get_measurements()



        # Read tag keys and values for each measurement (for each database or for input database)
        self.tags = self._get_all_tag_key_value_pairs()



        # Read field keys for each measurement (for each database or for input database)
        self.fields  = self._get_all_field_keys()



        # Generate overall data structure
        self.data_structure = self._get_data_structure()



        # Get overall start and end timestamps, for each database and each measurement
        self.time_ranges = self._get_start_end_timestamps()



    def _connect_to_client(self, data_frame_client_flag):

        if not data_frame_client_flag:

            if self.database_name:

                return InfluxDBClient(self.ip_address, self.port, self.user_name, self.password, database= self.database_name)

            else:

                return InfluxDBClient(self.ip_address, self.port, self.user_name, self.password)

        else:

            if self.database_name:

                return DataFrameClient(self.ip_address, self.port, self.user_name, self.password, database= self.database_name)

            else:

                return DataFrameClient(self.ip_address, self.port, self.user_name, self.password)



    def _get_databases(self):

        myquery = self.client.query('SHOW DATABASES')
        
        self.databases = [d['name'] for d in myquery.get_points() if d['name'] != '_internal']

        # Update field and data structure with databases
        self.fields = dict.fromkeys(self.databases)

        self.tags = dict.fromkeys(self.databases)


        for database in self.tags.keys():

            self.fields[database] = dict()

            self.tags[database] = dict()

        return self.databases

     

    def _get_measurements(self):

        for database in self.databases:

            myquery = self.client.query('SHOW measurements', database = database)

            measurements_list = [i['name'] for i in myquery.get_points() if i['name'][0][0] != '.']

            self.measurements[database] = measurements_list

            # Update field and data structure with measurements
            self.fields[database] = dict.fromkeys(measurements_list)

            self.tags[database] = dict.fromkeys(measurements_list)

            for measurements in self.tags[database].keys():

                self.fields[database][measurements] = dict()

                self.tags[database][measurements] = dict()

        return self.measurements



    def _get_all_tag_key_value_pairs(self, verbose = True):

        if not self.databases:

            self.databases = self.list_databases()

        if not self.measurements:

           self.measurements = self.list_measurements() 

        for database in self.databases:

            for measurement in self.measurements[database]:
            
                if verbose: print('database:' , database, 'measurement: ', measurement)


                my_tagKey_query = self.client.query('SHOW TAG KEYS FROM ' + measurement, database = database)



                if my_tagKey_query:         

                    current_tag_keys = [i['tagKey'] for i in my_tagKey_query.get_points()]

                    # Update data structure with tag keys
                    for tag_key in current_tag_keys:

                        my_tagValue_query = self.client.query('SHOW TAG VALUES FROM ' + measurement + ' WITH KEY = ' + tag_key, database = database)

                        if my_tagValue_query:

                            current_tag_values = [i['value'] for i in my_tagValue_query.get_points()]

                            # Update data structure with tag values    
                            self.tags[database][measurement][tag_key] = current_tag_values

                else:

                    self.tags[database][measurement] = []

        return self.tags



    def _get_all_field_keys(self, verbose = False):

        if not self.databases:

            self.databases = self.list_databases()

        if not self.measurements:

           self.measurements = self.list_measurements()

        for database in self.databases:

            for measurement in self.measurements[database]:
            
                if verbose: print('database: ', database, ' measurement: ', measurement)

                myquery = self.client.query('SHOW FIELD KEYS FROM ' + measurement, database=database)
            
                if myquery:
        
                    current_field_keys = [i['fieldKey'] for i in myquery.get_points()]

                    # Update field structure with field keys
                    for key in current_field_keys:

                        self.fields[database][measurement][key] = []
    
        return self.fields 



    def _get_data_structure(self):

        for database in self.databases:

            self.data_structure[database] = dict()

            for measurement in self.measurements[database]:

                self.data_structure[database][measurement] = {'Tags': [], 'Fields': []}

                self.data_structure[database][measurement]['Tags'] = self.tags[database][measurement]

                self.data_structure[database][measurement]['Fields'] = self.fields[database][measurement]                

        return self.data_structure



    def _get_start_end_timestamps(self):

        for database in self.databases:

            self.time_ranges[database] = dict()

            for measurement in self.measurements[database]:

                my_start_query = f'SELECT * FROM {database}."autogen".{measurement} LIMIT 1'

                my_end_query = f'SELECT * FROM {database}."autogen".{measurement} Order By time Desc LIMIT 1'               

                if not self.data_frame_client_flag:

                    start = [i['time'] for i in self.client.query(my_start_query, database=database).get_points()][0]

                    end = [i['time'] for i in self.client.query(my_end_query, database=database).get_points()][0]

                else:

                    start = self.client.query(my_start_query, database=database)[measurement].index.strftime('%Y-%m-%dT%H:%M:%S.%fZ').values[0]

                    end = self.client.query(my_end_query, database=database)[measurement].index.strftime('%Y-%m-%dT%H:%M:%S.%fZ').values[0]

                self.time_ranges[database][measurement] = {'Start': [], 'End': []}

                self.time_ranges[database][measurement]['Start'] = start

                self.time_ranges[database][measurement]['End'] = end
        
        return self.time_ranges



    def _compose_query(self, database, measurement, select, where = None, time_range = None, groupby = None, fill = None, limit = None):

        # Build SELECT clause
        select_clause = self._build_select_clause(select)

        # Build WHERE clause
        if not time_range:

            where_clause = self._build_where_clause(where)

        elif time_range == 'use_global':

            time_range = [self.time_ranges[database][measurement]['Start'], self.time_ranges[database][measurement]['End']]

            where_clause = self._build_where_clause(where, time_range = time_range)

        else:

            where_clause = self._build_where_clause(where, time_range = time_range)

        # Build GROUPBY clause
        groupby_clause = self._build_groupby_clause(groupby) if groupby else None

        # Build LIMIT clause
        limit_clause = self._build_limit_clause(limit) if limit else None

        # Build FILL clause
        fill_clause = 'fill(' + fill + ')' if fill else None

        # Compose query
        my_query = f'SELECT {select_clause} FROM {database}.autogen.{measurement}'

        if where_clause: my_query = my_query + f' WHERE {where_clause}'

        if groupby_clause: my_query = my_query + f' GROUP BY {groupby_clause}'        

        if limit_clause: my_query = my_query + f' {limit_clause}'

        if fill_clause: my_query = my_query + f' {fill_clause}'

        return my_query        



    def _build_select_clause(self, select):

        select_clause = ['']

        i = 0

        for key, value in select.items():

            if i == 0:

                if value:

                    select_clause.append(key + '(' + value + ')')

                else:

                    select_clause.append(key)

            else:

                if value:

                    select_clause.append(',' + key + '(' + value + ')')

                else:

                    select_clause.append(',' + key)  
  
            i += 1

        select_clause = ''.join(select_clause)

        return select_clause



    def _build_where_clause(self, where, time_range = None):

        where_clause = ['']

        i = 0  
      
        if where:

            for key, value in where.items():

                if i == 0:

                    if type(value) is str:

                        where_clause.append(str(key) + ' = ' + '\'' + value + '\'')

                    else:
                        where_clause.append(str(key) + ' = ' + str(value))

                else:

                    if type(value) is str:

                        where_clause.append(' AND ' + str(key) + ' = ' + '\'' + value + '\'')
 
                    else:
                        where_clause.append(' AND ' + str(key) + ' = ' + str(value)) 
          
                i += 1

        if time_range:           

            if i > 0:

                # Add time range
                where_clause.append(' AND time >= \'' + time_range[0] + '\' AND time <= \'' + time_range[1] + '\'')

            else:

                where_clause.append('time >= \'' + time_range[0] + '\' AND time <= \'' + time_range[1] + '\'')

        where_clause = ''.join(where_clause)

        return where_clause



    def _build_groupby_clause(self, groupby):

        groupby_clause = ['']

        i = 0

        for key, value in groupby.items():

            if i == 0:

                if value:

                    groupby_clause.append(key + '(' + value + ')')

                else:

                    groupby_clause.append(key)

            else:

                if value:

                    groupby_clause.append(',' + key + '(' + value + ')')

                else:

                    groupby_clause.append(',' + key)

        groupby_clause = ''.join(groupby_clause)       

        return groupby_clause



    def _build_limit_clause(self, limit):

        key = [key for key, value in limit.items()][0]

        limit_clause = key + ' ' + limit[key]        

        return limit_clause



    def print_info(self):

        print('Databases: ')
        print(self.databases)
        print('Measurements: ')
        print(self.measurements)
        print('Tags: ')
        print(self.tags)
        print('Fields: ')
        print(self.fields)

        print(json.dumps(self.data_structure, sort_keys=True, indent=4))



    def list_tag_keys(self, database, measurement, verbose = False):

        return list(self.tags[database][measurement].keys())



    def list_tag_values(self, database, measurement, tag_key, verbose = False):

        return self.tags[database][measurement][tag_key]



    def list_field_keys(self, database, measurement, verbose = False):

        return list(self.fields[database][measurement].keys())



    def show_all_series_for_single_measurement(self, database, measurement):

        myquery = self.client.query('SHOW SERIES FROM ' + measurement, database = database)  

        return [i for i in myquery.get_points()]



    def get_time_range(self, measurement, database = None, where = None):

        if not database:

            database = self.database_name

        # Build where clause
        where_clause = self._build_where_clause(where)

        # Build queries            
        if not where_clause:

            my_start_query = f'SELECT * FROM {database}."autogen".{measurement} LIMIT 1'

            my_end_query = f'SELECT * FROM {database}."autogen".{measurement} Order By time Desc LIMIT 1'        

        else:

            my_start_query = f'SELECT * FROM {database}.autogen.{measurement} WHERE {where_clause} LIMIT 1'

            my_end_query = f'SELECT * FROM {database}.autogen.{measurement} WHERE {where_clause} Order By time Desc LIMIT 1' 

        # Query database
        start_query_output = self.client.query(my_start_query)

        end_query_output = self.client.query(my_end_query)

        # Get start and end
        if not self.data_frame_client_flag:

            start = [i['time'] for i in start_query_output.get_points()][0] if start_query_output else None

            end = [i['time'] for i in end_query_output.get_points()][0] if end_query_output else None 

        else:

            start = start_query_output[measurement].index.strftime('%Y-%m-%dT%H:%M:%S.%fZ').values[0]

            end = end_query_output[measurement].index.strftime('%Y-%m-%dT%H:%M:%S.%fZ').values[0]            

        return start, end



    def query_first_n_points(self, n, database, measurement, select, where = None, output_tag = None, print_out = True):

        limit = {'LIMIT': str(n)}

        my_query = self._compose_query(database, measurement, select, where = where, limit = limit)

        output_query = self.client.query(my_query, database = database)
        
        if print_out: 

            if not self.data_frame_client_flag:

                for i in output_query.get_points():

                    print(i)

            else:

                print(output_query[measurement])           

        if output_tag:

            return [i[output_tag] for i in output_query.get_points()]



    def query(self, measurement, select, 
                    database = None, where = None, 
                    time_range = None, groupby = None, 
                    fill = None, limit = None, output_tags = None,
                    return_query = False):
       
        if not database:   
     
            database = self.database_name

        # Query dataset
        my_query = self._compose_query(database, measurement, select, time_range = time_range, where = where, groupby = groupby, fill = fill, limit = limit)
        
        output_query = self.client.query(my_query, database = database)

        if output_query:

            if output_tags:

                output = []

                for tag in output_tags:

                    if not self.data_frame_client_flag:

                        output.append([i[tag] for i in output_query.get_points()])

                    else:

                        assert tag in output_query[measurement].keys(), 'Specified output tag was not found'

                        output.append(output_query[measurement][tag])

                if return_query:

                    return [i for i in output] if len(output) > 1 else [i for output[0] in output for i in output[0]], my_query

                else: return [i for i in output] if len(output) > 1 else [i for output[0] in output for i in output[0]]

            else:

                if return_query:

                    return output_query[measurement], my_query

                else:   return output_query[measurement]

        else:

            return pd.DataFrame()



    def rfc3339TOUnixMS(self, time):

        """ converts time to milliseconds

        Parameters:
        -----------
            time (datetime): time to convert
        Returns:
        -----------
            ms (milliseconds): the converted time in milliseconds
        """

        try:

            d = datetime.strptime(time, '%Y-%m-%dT%H:%M:%S.%fZ')

        except:

            try:

                d = datetime.strptime(time, '%Y-%m-%dT%H:%M:%SZ')

            except:

                d = datetime.strptime(time, '%Y-%m-%d')

        ms = int(calendar.timegm(d.timetuple()) * 1000)

        return ms



    def rfc3339TOTime(self,time):

        """ converts time to milliseconds
        Parameters:
        -----------
            time (datetime): time to convert
        Returns:
        -----------
            ms (milliseconds): the converted time in milliseconds
        """

        try:

            d = datetime.strptime(time, '%Y-%m-%dT%H:%M:%S.%fZ')

        except:

            try:

                d = datetime.strptime(time, '%Y-%m-%dT%H:%M:%SZ')

            except:

                d = datetime.strptime(time, '%Y-%m-%d')

        return d






if __name__ == '__main__':



    # Parameters to connect to our CLM InfuxDB database
    source_db_param = {'address':'138.131.217.39',
                       'port':8084,
                       'user':'root',
                       'password':'root'}


    # (Optional) Database name
    source_db_name = 'clm'



    # Select specific input
    part = '3004.003'
    operation = '416'
    cycle_id = '0'
    signal_name = 'stSigSpindleIndicator'
    sampling_time = '1ms'


    # Connect to InfluxDB database (i.e. collection of databases)
    source_db = InfluxDB_Reader(source_db_param['address'], source_db_param['port'], 
                                source_db_param['user'], source_db_param['password'], 
                                database_name = source_db_name, data_frame_client_flag = True)



    # Explore the database
    source_db.print_info()

    print('\nDatabases: ')
    print(source_db.databases, '\n')   
    print('Measurements: ')
    print(source_db.measurements, '\n')
    print('Tags: ')
    print(source_db.tags, '\n')
    print('Fields: ')
    print(source_db.fields, '\n')



    # Select specific database/measurement
    database = source_db.databases[0]
    measurement = source_db.measurements[database][0]


    
    # Get list of tags for specific database/measurement
    tag_keys = source_db.list_tag_keys(database, measurement)
    print('Tag keys for database = ' + database + ' and measurement = ' + measurement + ' :')
    print(tag_keys, '\n')



    # Get list of tag values for specific database/measurement/tag key
    tag_key = tag_keys[0]
    tag_values = source_db.list_tag_values(database, measurement, tag_key)
    print('Tag values for database = ' + database + ', measurement = ' + measurement + ' and tag key = ' + tag_key + ' :')
    print(tag_values, '\n')



    # Get list of fields
    field_keys = source_db.list_field_keys(database, measurement)
    print('Field keys for database = ' + database + ', measurement = ' + measurement + ' :')
    print(field_keys, '\n')



    # Query first n points specific database/measurement/field
    source_db.query_first_n_points(10, database, measurement, {signal_name: ''})


    
    ### Example of a complete data query ###

    # Step 1) Retrieve time range for the wanted field (signal)
    start, end = source_db.get_time_range(measurement, 
                                          #where = {'part': part, 'operation': operation, 'cycle': cycle_id})
                                          where = {'part': part})

    # Step 2) Query for field

    # Example 2a): query for all fields and select one of them afterwards
    all_signals = source_db.query(measurement, {'*': ''}, 
                                      time_range = [start, end])

    current_signal_a = all_signals[signal_name]

    print('\n', current_signal_a, '\n')



    # Example 2b): query for one signal (output: pandas dataframe of time + values)
    current_signal_b = source_db.query(measurement, {signal_name: ''}, 
                                       time_range = [start, end])

    print('\n', current_signal_b, '\n')



    # Example 2c): query for one signal (output: only list of field values)
    current_signal_c = source_db.query(measurement, {signal_name: ''}, 
                                       time_range = [start, end],
                                       output_tags = [signal_name])

    print('\n', current_signal_c, '\n')



    # Example 2d): query for mean signal averaged every n datapoints (output: pandas dataframe of time + values)
    current_signal_d = source_db.query(measurement, {'mean': signal_name}, 
                                       time_range = [start, end],
                                       groupby = {'time': sampling_time}, fill = 'linear')

    print('\n', current_signal_d, '\n')



    # Example 2e): query for mean signal averaged every M=sampling_time datapoints (output: only list of field values)
    current_signal_e = source_db.query(measurement, {'mean': signal_name}, 
                                       time_range = [start, end],
                                       groupby = {'time': sampling_time}, fill = 'linear',
                                       output_tags = ['mean'])

    print('\n', current_signal_e, '\n')



    # Example 2f): query with "where" clause (it's redundant if [start,end] were already retrieved with the same "where" clase)
    current_signal_f = source_db.query(measurement, {'mean': signal_name}, 
                                       where = {'part': part, 'operation': operation, 'cycle': cycle_id},
                                       time_range = [start, end],
                                       groupby = {'time': sampling_time}, fill = 'linear',
                                       output_tags = ['mean'])

    print('\n', current_signal_f, '\n')



    
    



    


    
    
    