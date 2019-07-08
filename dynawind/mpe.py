# -*- coding: utf-8 -*-
"""
Created on Tue Apr 11 17:52:40 2017

@author: Wout Weijtjens
"""
import numpy as np
import matplotlib.pyplot as plt
import numpy.matlib
#%% calculate base function for  LSCF
def wj(f,fmin,fmax,MO=0,conv='normal',base='exp'):
    if base=='exp':
        pct=0.45 # how much of the unit circle is covered
        if conv=='normal': 
            w =np.exp(-1j*(pct*(f-fmin)/(fmax-fmin))*2*np.pi*MO)
#            w =np.exp(-1j*f*np.pi*MO)
        elif conv=='inverse':
            w =-np.log(f)*(fmax-fmin)/pct+2j*np.pi*fmin
#            w =np.log(f)
    return w
#%%
def correlogram(Signals):

    Ns=len(Signals[0].data)
    No=len(Signals)
    AllSignals=np.zeros((Ns,No))
    index=0
    for signl in Signals:
        AllSignals[:,index]=signl.data-signl.mean()
        index+=1

    #%%
    CORR=np.zeros((2*Ns,No,No));
    temp=np.concatenate((AllSignals,np.zeros(np.shape(AllSignals))))
    Y=np.fft.fft(temp,axis=0)

    for k in range(No):
        GG=np.multiply(np.conj(Y),np.transpose(np.matlib.repmat(Y[:,k],No,1)))
        Rr=np.fft.ifft(GG,axis=0).real;
        CORR[:,:,k]=Rr

    corrpos=CORR[0:,:,:]
    R=corrpos
    return R

#%% LSCF
      
def LSCF(Signals,
         modelorder=48,
         min_modelorder=4,
         base='exp',
         fband=(0,5),
         windowlength=30,  # in seconds
         ref=0,
         beta=0.1,
         powerspectrum='correlogram',
         plotresults=False): 
    
    windowlength = int(windowlength*Signals[0].Fs)
    
    n_o = len(Signals)
    if powerspectrum == 'correlogram':
        # Calculate correlogram
        Nf=int(windowlength/2)
        correlo=correlogram(Signals)
        r=correlo[1:windowlength+1,:,:] # Positive lags
        # Apply exponential window
        Ts=1./Signals[0].Fs
        m=np.linspace(0,windowlength,windowlength)
        window=np.exp(-beta*Ts*abs(m))
        windowedCorrelo=np.empty(r.shape)
        for i in range(0,n_o):
            for j in range(0,n_o):
                windowedCorrelo[:,i,j]=r[:,i,j]*window
        
        spectra = np.fft.fft(windowedCorrelo,axis=0)
        freq = np.fft.fftfreq(windowedCorrelo.shape[0],Ts)
     # Select the frequency band
    def find_nearest(array,value):
        idx = (np.abs(array-value)).argmin()
        return idx
    if fband is None:
        Nf=spectra.shape[0]
        N0=0;
    else:
        Nf = find_nearest(freq,fband[1])
        N0 = find_nearest(freq,fband[0])
    
    if N0 == 0:
        N0=1
    # build observation matrix
    Omega=np.empty((Nf-N0,modelorder),dtype=complex)
    for i in range(Nf-N0):
        for j in range(modelorder):
            Omega[i,j]=wj(freq[N0+i],freq[N0],max(freq[N0:Nf]),MO=j,base=base)
    
    
    G=spectra[N0:Nf,:,ref]
    Gxx=np.empty((G.shape[0]*n_o,modelorder),dtype=complex)
    for i in range(n_o):
        Gxx[G.shape[0]*i:G.shape[0]*(i+1),:]=np.transpose(np.matlib.repmat(G[:,i],modelorder,1))
    
    
    Kb=-1*np.multiply(Gxx,np.matlib.repmat(Omega,n_o,1))
    
    Omega_all=np.empty((G.shape[0]*n_o,n_o*modelorder),dtype=complex)
    for i in range(modelorder):
        Omega_all[:,i*n_o:(i+1)*n_o]=np.transpose(np.kron(np.eye(n_o),Omega[:,i]))
    Omega=Omega_all     
   
    
    # Solve equation in LS sense
#    poles=np.zeros((int(modelorder/2),int(modelorder/2)),dtype=complex)
#    modes=np.zeros((No,int(modelorder/2),int(modelorder/2)),dtype=complex)
#    
    poles=np.zeros((int(modelorder)-1,int(modelorder)-1),dtype=complex)
    modes=np.zeros((n_o,int(modelorder),int(modelorder)),dtype=complex)
#    
#    for MO in range(min_modelorder,modelorder,2):
    for MO in range(min_modelorder,modelorder):
        y=-1*Kb[:,MO-1] #Apply constraint d_nd=1
        K_red=np.concatenate((Kb[:,:MO-1],Omega[:,:n_o*MO]),axis=1)
    
        # Force complex conjugate poles
#        K_red=np.concatenate((np.imag(K_red),np.real(K_red)))
#        y=np.concatenate((np.imag(y),np.real(y)))
#       
        theta, res, rank, s = np.linalg.lstsq(K_red, y, rcond=-1)
        theta=np.insert(theta,MO-1,1)
    
    
        roots=np.roots(np.flipud(theta[:MO]))
        temp_poles=wj(roots,freq[N0],max(freq[N0:Nf]),conv='inverse',base=base)
#        poles[:int(MO/2),int(MO/2-1)]=temp_poles[::2]+beta # Compensate for the window
        poles[:int(MO-1),int(MO-1)]=temp_poles[::1]+beta # Compensate for the window

    poles[poles==0]=np.nan

#    if plotresults:
#        Den=np.dot(Omega[:,:MO],theta[:MO])
#        Num=np.dot(Omega[:,:MO],theta[MO:])
#        y_est=np.divide(Num,Den)
#        plt.semilogy(freq[N0:Nf],abs(G))
#        plt.semilogy(freq[N0:Nf],abs(y_est))

    return poles,modes


def SSICOV(Signals,modelorder=48,jb=10):
    #%% jb is expressed in seconds!
    jb=int(jb*Signals[0].Fs)
    #%%
    
    No=len(Signals)
    R=correlogram(Signals)

    #%% Build Toeplitz matrix
    # Current code outperfoms code based on kron()
    Toeplitz=np.zeros((No*jb,No*jb))
    for i in range(jb):
        Toeplitz[i*No:(i+1)*No,i*No:(i+1)*No]=R[jb-1,:,:]
        for j in range(1,jb-i):
            x=(i+j)*No
            y=(i+j+1)*No
            Toeplitz[i*No:(i+1)*No,x:y]=R[jb-j-1,:,:]
            Toeplitz[x:y,i*No:(i+1)*No]=R[jb-1+j,:,:]
    #%% Calculate eigenvalues
    U,S,V=np.linalg.svd(Toeplitz)
    O=U*np.sqrt(S)
    poles=np.zeros((int(modelorder/2),int(modelorder/2)),dtype='complex')
    modes=np.zeros((No,int(modelorder/2),int(modelorder/2)),dtype='complex')

    for MO in range(2,modelorder,2):
        O_to=O[:No*(jb-1),:MO]
        O_bo=O[No:,:MO]
        A,res,rank,s=np.linalg.lstsq(O_to,O_bo)
        Lambda,Phi=np.linalg.eig(A);
        pole=np.log(Lambda)*Signals[0].Fs;
        poles[:int(MO/2),int(MO/2)]=pole[::2]
        Phi=np.dot(O[:No,:MO],Phi)
        modes[:,:int(MO/2),int(MO/2)]=Phi[:,::2]
        poles[poles==0]=np.nan
    return poles,modes

def DBSCAN(poles,modes,plotClusters=False,min_clustersize=5):
    # DBSCAN algorithm for clustering the poles
    from sklearn.cluster import DBSCAN
    No=modes.shape[0]
    X=np.transpose(np.array((np.reshape(abs(poles)/2/np.pi,-1)*10,np.reshape(np.divide(-np.real(poles),abs(poles))*10,-1))))
    SortedModes=np.reshape(modes,(No,modes.shape[1]**2))
    Xtest=np.isnan(X[:,1])
    Indices=np.array(range(1,poles.size+1))
    X =X[~Xtest,:]
    Indices=Indices[~Xtest]
    # Remove unrealistic (>50%) + unstable damping values
    Xtest=(X[:,1]>5)+(X[:,1]<0)
    X =X[~Xtest,:]
    Indices=Indices[~Xtest]
    #%%
    db = DBSCAN(eps=0.1, min_samples=min_clustersize).fit(X)
    core_samples_mask = np.zeros_like(db.labels_, dtype=bool)
    core_samples_mask[db.core_sample_indices_] = True
    #%%
    # Number of clusters in labels, ignoring noise if present.
    unique_labels=set(db.labels_)
    n_clusters_ = len(unique_labels) - (1 if -1 in db.labels_ else 0)
    colors = plt.cm.Spectral(np.linspace(0, 1, len(unique_labels)+1))
    freq_median=np.zeros(n_clusters_)
    freq_mean=np.zeros(n_clusters_)

    freq_std=np.zeros(n_clusters_)
    damp_median=np.zeros(n_clusters_)
    damp_std=np.zeros(n_clusters_)

    modes_mean=np.zeros([No,n_clusters_],dtype='complex')
    for k, col in zip(sorted(unique_labels),colors):
        if k == -1:
            # gray used for noise.
            col = '#C6CCCF'
            markeredgecolor=col
        else:
            markeredgecolor='k'
        class_member_mask = (db.labels_ == k)
        xy = X[class_member_mask & core_samples_mask]
        ind=Indices[class_member_mask & core_samples_mask]
        #%% Plot results of clustering
        if plotClusters:
            plt.plot(xy[:, 0]/10, xy[:, 1]*10, 'o', markerfacecolor=col,markeredgecolor=markeredgecolor, markersize=14)
            xy2 = X[class_member_mask & ~core_samples_mask]
            plt.plot(xy2[:, 0]/10, xy2[:, 1]*10, 'o', markerfacecolor=col,markeredgecolor=markeredgecolor, markersize=6)
            plt.xlabel('Frequency (Hz)')
            plt.ylabel('Damping (%)')
        #%%
        if xy.size>0:
            freq_median[k]=np.median(xy[:,0]/10)
            freq_std[k]=np.std(xy[:,0]/10)
            freq_mean[k]=np.mean(xy[:,0]/10)
            damp_median[k]=np.median(xy[:,1]*10)
            damp_std[k]=np.std(xy[:,1]*10)
            modes_mean[:No,k]=np.mean(SortedModes[:,ind],axis=1)
    
    modes_mean=modes_mean[:,~np.isnan(freq_median)]    
    freq_std=freq_std[~np.isnan(freq_median)]
    freq_median=freq_median[~np.isnan(freq_median)]
    return freq_mean,freq_median, freq_std,damp_median,damp_std,modes_mean
#%% TrackTarget class
class TrackTarget(object):
    # this class is used to define the tracking parameters
    def __init__(self,Name,Freq,config):
        # Currently only a frequency is used for tracking
        self.name=Name
        self.freq=Freq #Target frequency for tracking
        self.algorithm=config.pop('track_mode') #TrackingAlgorithm
        self.direction=config.pop('direction')
        self.settings=dict()
        for key in config:
            self.settings[key]=float(config[key])
        
#%% Tracking
def tracking(mpe,trackinglist,root='.'):
    import dynawind.mdl as mdl
    def freqtrack(frequencies,target):
        return np.argmin(np.abs(frequencies-target))
    #% currently works using only the frequencies, result stored in dict
    Operators=['freq_median','freq_std','damp_median','damp_std']# Currently no modes
    tracked_all=[] # Tracking results is a list of dicts
    for track in trackinglist:
        treshold=None
        tracked=dict()
        if track.algorithm == 'frequency':
            target=track.freq
            if 'treshold' in track.settings:
                treshold=track.settings['treshold']
            
        elif track.algorithm == 'model':
            print('Model based tracking: '+track.name)
            track_mdl=mdl.Model(mpe.site,mpe.location,track.name)
            track_mdl_pred=track_mdl.evalModel(mpe.timestamp,root=root)
            target=track_mdl_pred['prediction']
            
            if np.isnan(target):
                continue
            if 'treshold' in track.settings:
                treshold=track.settings['treshold']*track_mdl_pred['std']
        TrackIndex=freqtrack(mpe.freq_median,target) # Current tracking only based on frequency
        for j in range(0,len(Operators)):
            if treshold:
                if abs(getattr(mpe,'freq_median')[TrackIndex]-track.freq)<treshold:
                    tracked['name']=track.name
                    tracked[Operators[j]]=getattr(mpe,Operators[j])[TrackIndex]
            else:
                tracked[Operators[j]]=getattr(mpe,Operators[j])[TrackIndex]
                tracked['name']=track.name

        
        if tracked:
            tracked_all.append(tracked)
    return tracked_all


# %%
class MPE(object):
    def __init__(self, signals,
                 algorithm=LSCF,
                 cluster_algorithm=DBSCAN,
                 modelorder=48,
                 settings=None,
                 cluster_settings={'min_clustersize':5},
                 trackinglist=[],
                 direction=None,
                 root='.'):
        import datetime
        #%%
        if not isinstance(signals,list):
            signals = [signals]
        
        #%% Properties
        self.source = signals[0].source # Legacy
        self.location = signals[0].location
        self.site = signals[0].site
        self.timestamp = signals[0].timestamp+datetime.timedelta(0,0,0,0,10,0)
        self.direction = direction # This input allows to differentiate between e.g. the tracking set for FA and SS
        self.algorithm = algorithm
        self.modelorder = modelorder
        self.cluster_algorithm = cluster_algorithm
        self.signals = signals
        # check for Settings in config file
        if not settings:
            (settings,clusterSettings,trackinglist)=self.mpe_config(direction=self.direction)
        # Sensorlevels for modeshapes
        self.levels=[-20] # this should be the mudline of the turbine
        for signl in signals:
            if hasattr(signl, 'level'):
                self.levels.append(signl.level)
            else:
                self.levels.append(42) # the answer to everything
        # Actual MPE
        self.all_poles,self.all_modes=self.algorithm(signals,modelorder=self.modelorder,**settings)
        self.freq_mean,self.freq_median,self.freq_std,self.damp_median,self.damp_std,self.modes_mean=self.cluster_algorithm(self.all_poles,self.all_modes,**cluster_settings)
        self.sortResults()
        # Tracking
        if trackinglist:
           self.tracked=tracking(self,trackinglist,root=root)
    def mpe_config(self,direction=None):
        import configparser
        import os.path
        import pkg_resources
        resource_package = __name__
        
        if self.source != 'unknown':
            site = self.site
            location = self.location
            resource_path = '/'.join(('config', site.lower(),
                              location+'_mpe.ini'))
            ini_file = pkg_resources.resource_filename(resource_package,
                                                       resource_path)
            settings = dict()
            clustersettings = None
            trackinglist = []
            if os.path.isfile(ini_file):
                config = configparser.ConfigParser()
                config.read(ini_file)
                for section in config.sections():
                    if section == 'Modal analysis':
                        for key in config['Modal analysis']:
                            if key == 'modelorder':
                                self.modelorder = int(config['Modal analysis']['modelorder'])
                            elif key == 'algorithm':
                                if config['Modal analysis']['algorithm'].lower() == 'plscf':
                                    self.algorithm = LSCF
                                elif config['Modal analysis']['algorithm'].lower() == 'ssicov':
                                    self.algorithm = SSICOV
                            elif key == 'f_band':
                                # fband is entered as 0,8
                                fband=config['Modal analysis']['f_band'].split(',')
                                settings['fband']=[float(i) for i in fband]
                            else:
                                settings[key]=int(config['Modal analysis'][key])
                    elif section == 'Cluster':
                        clustersettings = dict()
                        for key in config['Cluster']:
                            clustersettings[key] = int(config['Cluster'][key])

                    else:
                        freq = float(config[section].pop('Freq'))
    
                        if direction:
                            if config[section]['direction'] == direction:
                                trackinglist.append(TrackTarget(section,freq,config[section]))
                        else:
                            trackinglist.append(TrackTarget(section,freq,config[section]))
    
                            
            return (settings,clustersettings,trackinglist)
                        
                
    def stab_chart(self, signal=None, xlim=[0,2]):
        
        if signal is None:
            signal = self.signals[0]
        good_poles=self.all_poles
        good_poles[np.real(good_poles)>0]=np.nan # remove unstable poles
        

        test = np.abs(good_poles)/np.pi/2
        fig, ax1 = plt.subplots()
        ax1.set_xlabel('Frequency (Hz)')
        ax1.set_ylabel('Model Order')
        
        plt.plot(np.transpose(test),
                 np.linspace(1, np.size(self.all_poles, 0),
                             num=np.size(self.all_poles, 0))*2,
                             'rx')
        plt.vlines(self.freq_median, 0, np.size(self.all_poles, 0)*2, 'k', ':')

        ax2 = ax1.twinx()
        signal.plotPSD(xlim=xlim)


    def sortResults(self):
        p =  self.freq_median.argsort()
        self.freq_mean=self.freq_mean[p]
        self.freq_median=self.freq_median[p]
        self.freq_std=self.freq_std[p]
        self.damp_median=self.damp_median[p]
        self.damp_std=self.damp_std[p]
        self.modes_mean=self.modes_mean[:,p]
    def plotModeshapes(self):
        print(self.levels)
        plt.figure()
        realmodes=np.real(self.modes_mean)
        modesmax=np.max(abs(realmodes),axis=0)
        realmodes=np.divide(realmodes,np.matlib.repmat(modesmax,realmodes.shape[0],1))
        realmodes = np.concatenate((np.zeros((1,realmodes.shape[1])),realmodes))
        for i in range(realmodes.shape[1]):
            plt.plot(np.real(realmodes[:,i]),self.levels,label='f:{0:.3f}(Hz)|d:{1:.3f}(%)'.format(self.freq_median[i],self.damp_median[i]))
        # Add legend
        plt.legend()
    def plotResults():
        pass
    def plotClusters(self):
        self.clusterAlgorithm(self.all_poles,self.all_modes,plotClusters=True)
        plt.grid()
#    def track2df(self,TrackList):
#        #% currently works using only the frequencies, result stored in dataframe
#        import pandas as pd
#        import datetime,pytz
#        df= pd.DataFrame()
#        multiIndexTuples=[]
#        Operators=['freq_median','freq_std','damp_median','damp_std']# Currently no modes
#        timestamp=self.timestamp+datetime.timedelta(0,0,0,0,10,0)
#        df.loc[0,'time']=timestamp.strftime("%Y-%m-%d %H:%M:%S")
#        for track in TrackList:
#            TrackIndex=np.argmin(np.abs(self.freq_median-track.freq)) # Current tracking only based on frequency
#            df['time']=pd.to_datetime(df['time'],utc=True)
#            for j in range(0,len(Operators)):
#                multiIndexTuples.append((self.source,'MPE',track.name,Operators[j]))# Currently no possibility to have different algorithms for tracking purposes
#                df.loc[0,Operators[j]+'_'+track.name]=getattr(self,Operators[j])[TrackIndex]
#        
#        df.index=df['time']
#        df.index=df.index.tz_localize(tz=pytz.utc,ambiguous ='infer') # TDMS are in UTC!
#        
#        del df['time']
#        multiIndex=pd.MultiIndex.from_tuples(multiIndexTuples)
#        df.columns=multiIndex
#        df.sort_index(axis=1,inplace=True)
#        return df
    def exportdict(self,trackedOnly=True):
        if trackedOnly:
            export=dict()
            for track in self.tracked:
                export.update({'median_'+self.source+'_'+track['name']+'_FREQ':track['freq_median'],
                      'std_'+self.source+'_'+track['name']+'_FREQ':track['freq_std'],
                      'median_'+self.source+'_'+track['name']+'_DAMP':track['damp_median'],
                      'std_'+self.source+'_'+track['name']+'_DAMP':track['damp_std']})
        else:
            if self.direction:
                dirstr=self.direction+'/'
            else:
                dirstr=''
            
            export={dirstr+'freq/median':self.freq_median.tolist()}
            export[dirstr+'freq/std']=self.freq_std.tolist()
            export[dirstr+'damp/median']=self.damp_median.tolist()
            export[dirstr+'damp/std']=self.damp_std.tolist()
        return export
    def __repr__(self):
        # Technicaly you shouldn't use print in __repr__
        print('DYNAwind Modal parameter estimation object')
        print('Algorithm : '+self.algorithm.__name__)
        print('Cluster Algorithm :'+self.cluster_algorithm.__name__)
        print('Freq.\t| Std.\t|| Damp\t| Std.')
        print('(Hz).\t| (Hz).\t|| (%).\t| (%).')
        print('---------------------------------------------------')
        for i in range(len(self.freq_median)):
            print('{0:.3f}\t|{1:.3f}\t||{2:.3f}\t|{3:.3f}'.format(self.freq_mean[i],self.freq_std[i],self.damp_median[i],self.damp_std[i]))
        return ""