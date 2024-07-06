#!/usr/bin/env python
import os
import sqlite3
import datetime
import numpy as np

class LSTIDFitDb(object):
    def __init__(self,db_fname='lstidSinFit.sql',deleteDb=False,db='lstidSinFit'):
        self.db_fname   = db_fname
        self.db         = db

        if os.path.exists(db_fname) and deleteDb:
            os.remove(db_fname)
        
        conn    = sqlite3.connect(self.db_fname)
        crsr    = conn.cursor()

        # Drop the lstidSinFit table if already exists.
        crsr.execute("DROP TABLE IF EXISTS {!s}".format(db))

        schema  = {}
        schema['T_hr']          = 'FLOAT'
        schema['amplitude_km']  = 'FLOAT'
        schema['phase_hr']      = 'FLOAT'
        schema['offset_km']     = 'FLOAT'
        schema['slope_kmph']    = 'FLOAT'
        schema['sTime']         = 'TIMESTAMP'
        schema['eTime']         = 'TIMESTAMP'
        schema['good_data']     = 'INT'
        schema['confirm_fit']   = 'INT'
        self.schema = schema
        
        sch     = ',\n '.join([f'{key} {val}' for key, val in schema.items()])
        table   = f'CREATE TABLE {db} (\n Date TIMESTAMP NOT NULL,\n {sch}\n);'

        crsr.execute(table)
        print("Table is Ready")

        # Close the connection
        conn.close()

    def insert_fit(self,date,params):
        params          = params.copy()
        params['Date']  = date

        keys    = []
        vals    = []
        for key,val in params.items():
            val_str = '"{!s}"'.format(val)

            keys.append(key)
            vals.append(val_str)

        qry = 'INSERT INTO {} ({}) VALUES ({})'.format(self.db, ','.join(keys), ','.join(vals)) 

        conn    = sqlite3.connect(self.db_fname)
        crsr    = conn.cursor()
        crsr.execute(qry)
        conn.commit()
        conn.close()

    def get_fit(self,date):
        conn    = sqlite3.connect(self.db_fname)
        crsr    = conn.cursor()

        p0      = {}
        for key,dtype in self.schema.items():
            qry     = f'SELECT {key} FROM {self.db} WHERE Date = "{date}";'
            crsr.execute(qry)
            result  = crsr.fetchall()[0][0]

            if  dtype == 'TIMESTAMP':
                result  = datetime.datetime.fromisoformat(result)

            if key in ['good_data','confirm_fit']:
                result = bool(result)

            p0[key] = result

        conn.close()

        return p0

if __name__ == '__main__':
    ldb = LSTIDFitDb()

    sDate = datetime.datetime(2018,11,1) 
#    eDate = datetime.datetime(2018,11,1) 
    eDate = datetime.datetime(2019,5,1) 
    dates = [sDate]
    while dates[-1] < eDate:
        dates.append(dates[-1]+datetime.timedelta(days=1))

    for date in dates:
        print('INSERTING: ', date)
        p0  = {}
        p0['T_hr']          = 3
        p0['amplitude_km']  = 200
        p0['phase_hr']      = 0
        p0['offset_km']     = 1400
        p0['slope_kmph']    = 0
        p0['sTime']         = date + datetime.timedelta(hours=13)
        p0['eTime']         = date + datetime.timedelta(hours=23)
        p0['good_data']     = True
        p0['confirm_fit']   = False

        ldb.insert_fit(date,p0)

    print()
    for date in dates:
        print('GETTING: ',date)
        p0  = ldb.get_fit(date)
        for key,val in p0.items():
            print('   {!s}: {!s}'.format(key,val))
        print()

    import ipdb; ipdb.set_trace()
