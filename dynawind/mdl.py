# -*- coding: utf-8 -*-
"""
Created on Fri Mar 30 16:43:22 2018

@author: Wout Weijtjens
"""
import numpy as np
import json
import os

class Model(object):
    def __init__(self,site,location,output):
        self.location=location
        self.site=site
        self.output=output
        self.importModel()
    def importModel(self):
        site=self.site
        mdl=None
        if self.location!='unknown':
            configPath='.\\config\\'+site.lower()+'\\models\\'
            if  os.path.isfile(configPath+self.location+'_'+self.output+'_mdl.json'):
                f=open(configPath+self.location+'_'+self.output+'_mdl.json', 'r')
                mdls=json.load(f)
                f.close()
                mdl=dict()
                for md in mdls:
                    newkey={md['case']:md}
                    mdl.update(newkey)
                
                
        self.mdl=mdl
    def evalModel(self,dt,root='.',fileext='.json'):
        from db import pullValuesfromJSON
        #%%
        data=pullValuesfromJSON(self.site,self.location,dt,root=root,fileext=fileext)
        
        output=dict()
        output['residual']=np.nan
        output['prediction']=np.nan
        output['likelihood']=np.nan
        output['std']=np.nan
        #%% Select from mdl the mdl associated with the case at dt
        if 'all' in self.mdl.keys(): # Universal model for all cases
            case='all'
        else:
            if 'case' in data:
                case=data['case']
            else:
                return output
        
        
        if case in self.mdl:
            mdl=self.mdl[case]
            if 'std' in mdl:
                output['std']=mdl['std']
            else:
                output['std']=0
                    
            #%% 
            output['prediction']=evaluateModel(mdl['model'],data,mdl['parameters'],mdl['theta'])
            #%%
            if self.output in data:
                output['residual']=data[self.output]-output['prediction']
            elif 'median_'+self.location+'_'+self.output+'_FREQ' in data:
                output['residual']=data['median_'+self.location+'_'+self.output+'_FREQ']-output['prediction']
                
            #%% Calculate likelihood
            
            output['likelihood']= output['residual']*((1/output['std'])**2)*output['residual']
            #%%
        return output
        
    def export2dict(self,output='prediction'):
        
        if output=='prediction':
            keystr='MDL_PRE_'
        elif output== 'Residual':
            keystr='MDL_RES_'
        elif output == 'Likelihood':
            keystr='MDL_LH_'
        
        export={keystr+self.output:result}
        return export    
        
    def __repr__(self):
        repr_str='DYNAwind model object\n'+'---------------------\n'+'Parameter: '+self.output
        return repr_str

def evaluateModel(modeltype,data,parameters,theta):
    if modeltype=='polynomial':
        if len(theta)==1:
            result=theta
        else:
            # theta[0]*x**(N-1) + theta[1]*x**(N-2) + ... + theta[N-2]*x + theta[N-1]
            result=np.polyval(theta,data[parameters[0]])
    return result[0]

#%% Class definitions
def readClassDefinitions(site):
    import configparser
    config=configparser.ConfigParser()
    config.read('./config/'+site+'/'+site+'_casedefinitions.ini')
    caseDefinitions=[]
    for section in config:
        if section is not 'DEFAULT':
            caseDef=dict(config[section])
            for key in caseDef.keys():
                caseDef[key]=np.float64(np.array(caseDef[key].split(sep=',')))
            caseDef["case"]=section
            caseDefinitions.append(caseDef)
    return caseDefinitions
def caseClassifier(data,site,caseDefinitions):
    for case in caseDefinitions:
        isCase=True
        for key in case.keys():
            if key not in data:
                return 'no SCADA'
            if key is 'case':
                continue
            if data[key]>=case[key][0] and data[key]<case[key][1]:
                inCase=True
            else:
                inCase=False
            isCase=isCase and inCase
        if isCase:
            return case['case']
            
    else:
        return 'caseless'
def getCaseforTimestamp(dt):
    

    return case