# -*- coding: utf-8 -*-
"""
Created on Thu Mar  8 15:18:24 2018

@author: Wout Weijtjens
"""
import numpy as np
import datetime
import pytz
import json

def roundedTimestamp(dt=None,delay=1):
   from math import floor
   import pytz

   """
   Produces a timeobject that represents the files to be processed when called
   """
   if dt == None : 
       dt = datetime.datetime.utcnow() # Time in UTC
   flooring=dt.minute-floor(dt.minute/10)*10
   dt = dt-datetime.timedelta(minutes=flooring,seconds=dt.second,microseconds=dt.microsecond) #Rounding
   dt = dt-datetime.timedelta(minutes=delay*10) #Delay
   if dt.tzinfo is None:
       dt=pytz.utc.localize(dt)
   return dt


#%% return TDMS path
def returnTDMSpath(site,location,datasubtype,datatype='TDD',dt=None,root=r'\\192.168.119.14'):
    import os

    if dt == None:
        dt=roundedTimestamp(dt=None,delay=1)
    else:
        dt=dt.astimezone(pytz.utc)

        
    if 'TDD_' not in datasubtype:
        datasubtype='TDD_'+datasubtype

    timestr=dt.strftime('%Y%m%d_%H%M%S') # ' 1:36PM EDT on Oct 18, 2010'
    if site is None:
        filePath=os.path.join(root,location,datatype,datasubtype,dt.strftime('%Y'),dt.strftime('%m'),dt.strftime('%d'),timestr+'.tdms')
    else:
        filePath=os.path.join(root,'data_primary_'+site.lower(),location,datatype,datasubtype,dt.strftime('%Y'),dt.strftime('%m'),dt.strftime('%d'),timestr+'.tdms')
    return filePath
#%% JSON HANDLING

def initJSON(jsonPath,dt,site,location):
    jsonFile=open(jsonPath,'w')
    data_dict=dict()
    data_dict['location']=location
    data_dict['site']=site
    data_dict['timestamp']=dt.__str__()
    json.dump([data_dict],jsonFile,indent=2)
    jsonFile.close()

def returnJSONfilePath(dt,site,location,root='.',fileext='.json'):
    import os
    Folder=os.path.join(root,site,location,dt.strftime('%Y'),dt.strftime('%m'),dt.strftime('%d'))
    if not os.path.isdir(Folder):
        os.makedirs(Folder)
    filePath=os.path.join(Folder,dt.strftime('%Y%m%d_%H%M%S')+fileext)
    if not os.path.isfile(filePath):
        initJSON(filePath,dt,site,location)
    return filePath

#%% POSTGRESQl
def insertJSON2PostgreSQL(filePath,action='INSERT',cluster=False,table=None):
    # table can be specified, if not will result in default
    import os
    if os.path.isdir(filePath):
        # Make list of all files in filePath
        filelist=[]
        for root, dirs, files in os.walk(filePath, topdown=False):
            for name in files:
                filelist.append(os.path.join(root, name))
    else:
        filelist=[filePath]

    #%% Step 1 : Read config based on first file
    f=open(filelist[0])
    data=json.load(f)
    f.close()
    
    site=data[0].pop('site')
    
    #%% Open connection
    (conn,cur,table_config)=connect2postgreSQL(site)
    if table == None:
        table=table_config
    
    if action=='INSERT':
        #%% Write INSERT QUERY
        sql="""INSERT INTO """ +table +""" (site,location,timestamp,metrics) 
        VALUES (%(site)s,%(location)s,%(timestamp)s,%(jsonobj)s) ON CONFLICT DO NOTHING"""
    elif action=='UPDATE':
        sql="""UPDATE """+table+""" SET metrics = %(jsonobj)s WHERE location=%(location)s AND timestamp=%(timestamp)s"""

    
    ind=0
    # Step 4 : Run through all files and execute SQL
    for file in filelist:
        f=open(file)
        ind+=1
        print(str(ind)+'/'+str(len(filelist)),end='\r')
        data=json.load(f)
        f.close()
        timestamp=data[0].pop('timestamp')
        location=data[0].pop('location')
        cur.execute(sql,{'site':site,'location':location,'timestamp':timestamp,'jsonobj':json.dumps(data[0])})
    
    #%% Step 5 : Commit and close connection to postgresSQL database  
    if cluster:
        sql="""CLUSTER """ +table +""" USING """+table+"""_location_timestamp_idx"""
        cur.execute(sql)

    conn.commit()
    cur.close()
    conn.close()
    print(str(len(filelist))+' files transfered into '+table+' for location '+location,end='\r')

#%% 
def pullValuesfromJSON(site,location,dt,parameters=None,root='.',fileext='.json'):
    dt=dt.astimezone(pytz.utc)

    jsonPath=returnJSONfilePath(dt,site,location,root=root,fileext=fileext)
    f=open(jsonPath)
    data=json.load(f)
    f.close()
    data_expt=dict()
    if parameters is None:
        parameters=data[0].keys()
    for par in parameters:
        if par in data[0]:
            data_expt[par]=data[0][par]
        else:
            data_expt[par]=np.empty(1,)
    return data_expt
#%% 
def connect2postgreSQL(site):
    import configparser
    import psycopg2 as postgres
    import pkg_resources
    resource_package = __name__
    # Step 1 : load site config file
    config=configparser.ConfigParser()
    resource_path = '/'.join(('config', site.lower(),site.lower()+'_postprocessing.ini'))
    ini_file=pkg_resources.resource_filename(resource_package,resource_path)
    config.read(ini_file)
    #%% Step 2 Make connection to postgres database
    conn=postgres.connect(host=config['postgreSQL']['host'],port=config['postgreSQL']['port'],database=config['postgreSQL']['database'],user=config['postgreSQL']['user'],password=config['postgreSQL']['password'])
    cur=conn.cursor()
    table = config['postgreSQL']['table']
    return (conn,cur,table)