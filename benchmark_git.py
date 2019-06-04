#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan 31 10:22:24 2019

@author: txuslopez
"""

from skmultiflow.evaluation.evaluate_prequential_NN import EvaluatePrequential_NN
from skmultiflow.trees import HoeffdingTree,GRF_HoeffdingTree,HAT,GRF_HoeffdingAdaptiveTree
from skmultiflow.data.file_stream import FileStream
from skmultiflow.lazy.knn import KNN
from skmultiflow.lazy.grf_knn import GRF_KNN
from skmultiflow.bayes import NaiveBayes, GRF_NaiveBayes
from sklearn import preprocessing
from sklearn.naive_bayes import GaussianNB
from sklearn.grf_naive_bayes import GRF_GaussianNB
from sklearn.linear_model.passive_aggressive import PassiveAggressiveClassifier
from sklearn.linear_model.grf_passive_aggressive import GRF_PassiveAggressiveClassifier

from sklearn.neural_network.multilayer_perceptron import MLPClassifier
from sklearn.neural_network.grf_multilayer_perceptron import GRF_MLPClassifier

from sklearn.linear_model.stochastic_gradient import SGDClassifier
from sklearn.linear_model.grf_stochastic_gradient import GRF_SGDClassifier

from texttable import Texttable
from skmultiflow.drift_detection import ADWIN
from collections import deque
from statsmodels.stats.contingency_tables import mcnemar

#import OnlineGRF
import pandas as pd
import numpy as np
import math
import warnings
import pickle
import scipy.io as sio
import scipy.stats
import matplotlib.pyplot as plt
import seaborn as sns

warnings.filterwarnings("ignore",category=DeprecationWarning)

#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
############################################################## FUNCTIONS ##############################################################################
#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

def fxn():
    warnings.warn("deprecated", DeprecationWarning) 

def window(seq, n=2):
    it = iter(seq)
    win = deque((next(it, None) for _ in range(n)), maxlen=n)
    yield win
    append = win.append
    for e in it:
        append(e)
        yield win
        
def cargaDatos(datasets,data,severity,speed,lim_data):
    
    if data==0:#weather
    
        stream = FileStream('your_path')
        stream.prepare_for_use() 
    
        df=pd.DataFrame(stream.X)
        x = df.values
        min_max_scaler = preprocessing.MinMaxScaler()
        x_scaled = min_max_scaler.fit_transform(x)
        df = pd.DataFrame(x_scaled)
        
        stream.X=df.as_matrix()
            
    elif data==1:#elec
    
        stream = FileStream('your_path')
        stream.prepare_for_use() 
        
    elif data==2:#covtype
    
        stream = FileStream('your_path')
        stream.prepare_for_use() 
    
        df=pd.DataFrame(stream.X)
        x = df.values
        min_max_scaler = preprocessing.MinMaxScaler()
        x_scaled = min_max_scaler.fit_transform(x)
        df = pd.DataFrame(x_scaled)
    
        df=df[0:5000]#Limitar porque tiene muchas features
        stream.X=df.as_matrix()
        
        #Hay q hacer que las labels vayan de 0-6 para que el tamaño del repositorio de OnlineGRF coincida
        stream.y=stream.y-1
        stream.target_values=list(np.unique(stream.y))
        
    elif data==3:#moving_squares
    
        stream = FileStream('your_path')
        stream.prepare_for_use() 

        df=pd.DataFrame(stream.X)
        df=df[0:lim_data]#Limitar datos a 50k samples        
        stream.X=df.as_matrix()

    elif data==4:#sea_stream
    
        stream = FileStream('your_path')
        stream.prepare_for_use() 
        
    elif data==5:#usenet2
    
        stream = FileStream('your_path')
        stream.prepare_for_use() 
        
    elif data==6:#gmsc
    
        df=pd.read_csv('your_path',sep=',',header=0)
        df = df.drop('Unnamed: 0', 1)#Quitamos la primera columna
        df=df.dropna(how='any')#Se quitan las filas con Nan
        df=df[0:lim_data]#Limitar datos a 50k samples    

        feats=df[['RevolvingUtilizationOfUnsecuredLines', 'age',
       'NumberOfTime30-59DaysPastDueNotWorse', 'DebtRatio', 'MonthlyIncome',
       'NumberOfOpenCreditLinesAndLoans', 'NumberOfTimes90DaysLate',
       'NumberRealEstateLoansOrLines', 'NumberOfTime60-89DaysPastDueNotWorse',
       'NumberOfDependents']]

#        x = feats.values
#        min_max_scaler = preprocessing.MinMaxScaler()
#        x_scaled = min_max_scaler.fit_transform(x)
#        feats = pd.DataFrame(x_scaled)
        
        clas=df[['SeriousDlqin2yrs']]
        
        df_result = pd.concat([feats, clas], axis=1, sort=False)
        
        df_result.to_csv('your_path')

        stream = FileStream('your_path')
        stream.prepare_for_use() 
        
        stream.X=feats.as_matrix()
        stream.y=clas.as_matrix()
        
    elif data==7:#airlines
    
        df = pd.read_csv('your_path', sep=',', header=None)
        
        #Tratar las features nominales: 0,2,3 columns
        #1. Si hacemos OneHot encoding, se convierte en tantas features que al usar GRF y su parametro gamma tarda demasiado
#        df=pd.get_dummies(df, columns=[0,2,3], prefix=["airline", "airport_from", "airport_to"])    
#        df.to_csv("//home//txuslopez//Dropbox//jlopezlobo//Data sets//Non stationary environments//Airlines//airlines2.csv")
        #2. Hacemos Label encoding
        df.iloc[:,0] = df.iloc[:,0].astype('category')
        df.iloc[:,0]=df.iloc[:,0].cat.codes
        
        df.iloc[:,2] = df.iloc[:,2].astype('category')
        df.iloc[:,2]=df.iloc[:,2].cat.codes

        df.iloc[:,3] = df.iloc[:,3].astype('category')
        df.iloc[:,3]=df.iloc[:,3].cat.codes
        
        #Quitamos la primera columna
        df=df.drop([0], axis=1)

#        df=pd.DataFrame(stream.X)
#                
#        x = df.values
#        min_max_scaler = preprocessing.MinMaxScaler()
#        x_scaled = min_max_scaler.fit_transform(x)
#        df = pd.DataFrame(x_scaled)
        df.to_csv('your_path')

        df=df[0:lim_data]#Limitar datos a 50k samples    

        stream = FileStream('your_path')
        stream.prepare_for_use() 

        stream.X=df.as_matrix()
        
    elif data==8 or data==9 or data==10 or data==11:#sinteticos
    
        synt_name=''
        synt_name2=''
        
        if data==8:        
            synt_name='circleG'
            synt_name2='CircleG'
        elif data==9:        
            synt_name='line'
            synt_name2='Line'
        elif data==10:        
            synt_name='sineH'
            synt_name2='SineH'
        elif data==11:        
            synt_name='sine'
            synt_name2='Sine'
        
        path='your_path'
        fil=synt_name+'//data'+synt_name2+'Sev'+str(severity)+'Sp'+str(speed)+'Train.csv'
        
        raw_data= pd.read_csv(path + fil, sep=',',header=None)
        caso=raw_data[raw_data.columns[0:3]]#Delete the last useless column
        caso.iloc[:,2]=(caso.iloc[:,2]).astype(int)#Se convierte la clase a int
        
        new_fil=synt_name+'_'+'Sev'+str(severity)+'_Sp'+str(speed)+'Train.csv'
        caso.to_csv(path+synt_name+'//'+ new_fil)
        
        stream = FileStream(path+synt_name+'//'+ new_fil)
        stream.prepare_for_use() 
        
        if synt_name=='sine':#Hay que escalar los datos
            df=pd.DataFrame(stream.X)
            x = df.values
            min_max_scaler = preprocessing.MinMaxScaler()
            x_scaled = min_max_scaler.fit_transform(x)
            df = pd.DataFrame(x_scaled)
            caso=df
                           
        stream.X=caso.iloc[:,0:2].as_matrix()
        
    elif data==12 or data==13 or data==14 or data==15 or data==16 or data==17 or data==18 or data==19:#sinteticos extendidos
    
        synt_name=''
        synt_name2=''
        
        if data==12 or data==13:        
            synt_name='circleG'
            synt_name2='CircleG'
        elif data==14 or data==15:        
            synt_name='line'
            synt_name2='Line'
        elif data==16 or data==17:        
            synt_name='sineH'
            synt_name2='SineH'
        elif data==18 or data==19:     
            synt_name='sine'
            synt_name2='Sine'
        
        path='your_path'   
        fil=synt_name+'//data'+synt_name2+'Sev'+str(severity)+'Sp'+str(speed)+'Train.csv'
        
        raw_data= pd.read_csv(path + fil, sep=',',header=None)
        caso=raw_data[raw_data.columns[0:3]]#Delete the last useless column
        caso.iloc[:,2]=(caso.iloc[:,2]).astype(int)#Se convierte la clase a int
        
        #Se alargan los concepts estables
        caso2=pd.DataFrame()
        if data==12 or data==14 or data==16 or data==18:#concepto estable 1    
            caso=caso[0:999]
            caso2=caso.iloc[np.tile(np.arange(len(caso)), 50)]        
            new_fil=synt_name+'_'+'concept1.csv'
        elif data==13 or data==15 or data==17 or data==19:#concepto estable 2
            caso=caso[1000:1999]
            caso2=caso.iloc[np.tile(np.arange(len(caso)), 50)]        
            new_fil=synt_name+'_'+'concept2.csv'
            
        
        caso2.to_csv(path+synt_name+'//'+ new_fil)
        
        stream = FileStream(path+synt_name+'//'+ new_fil)
        stream.prepare_for_use() 
        
        if synt_name=='sine':#Hay que escalar los datos para que no sean negativos, sino algunos algoritmos cascan
            df=pd.DataFrame(stream.X)
            x = df.values
            min_max_scaler = preprocessing.MinMaxScaler()
            x_scaled = min_max_scaler.fit_transform(x)
            df = pd.DataFrame(x_scaled)
            caso2=df
                           
        stream.X=caso2.iloc[:,0:2].as_matrix()        
                
    return stream

def cargaParametros(data,sev,sp):
    
    detection=False
    
    if data==0:#weather
    
        #GRF parameters
        gamma=2.0#2.0
        n_gaussianRF=3#3
        
        #KNN parameters
        window_size=50#50
        vecinos=5#5
        hoja_size=2
    
        #GLobal parameters
        pretrain_size=4500#4500
        
        detection=True
        n_wait=500#500
        
    elif data==1:#elec
    
        #GRF parameters
        gamma=2.0#2.0
        n_gaussianRF=5#5#Si incremento mejora notablemente Online GRF
    
        #KNN parameters
        window_size=10
        vecinos=3
        hoja_size=2
    
        #GLobal parameters
        pretrain_size=11000

        detection=True
        n_wait=500#1
    
    elif data==2:#covtype
    
        #GRF parameters
        gamma=2.0#2.0
        n_gaussianRF=9#7
        
        #KNN parameters
        window_size=10
        vecinos=3
        hoja_size=2
        
        #GLobal parameters
        pretrain_size=1250

        detection=True
        n_wait=500#1

    elif data==3:#moving_squares
    
        #GRF parameters
        gamma=2.0#2.0
        n_gaussianRF=11#11
        
        #KNN parameters
        window_size=50#10
        vecinos=20#3
        hoja_size=2#2
        
        #GLobal parameters
        pretrain_size=12500

        detection=True
        n_wait=500#1

    elif data==4:#sea_stream
    
        #GRF parameters
        gamma=2.0#1.7
        n_gaussianRF=3#3
        
        #KNN parameters
        window_size=100#100
        vecinos=15#5
        hoja_size=2
        
        #GLobal parameters
        pretrain_size=10000

        detection=True
        n_wait=500#1

    elif data==5:#usenet2
        
        #GRF parameters
        #OnlineGRF_KNN parameters
        gamma=2.0#2.0
        n_gaussianRF=3#3
        
        #KNN parameters
        window_size=25
        vecinos=5
        hoja_size=2        

        #GLobal parameters
        pretrain_size=375

        detection=True
        n_wait=50
    
    elif data==6:#gmsc
        
        #GRF parameters
        #OnlineGRF_KNN parameters
        gamma=2.0#2.0
        n_gaussianRF=3#3
        
        #KNN parameters
        window_size=10
        vecinos=3
        hoja_size=2        

        #GLobal parameters
        pretrain_size=12500

        detection=True
        n_wait=500#1

    elif data==7:#airlines
        
        #GRF parameters
        #OnlineGRF_KNN parameters
        gamma=2.0#2.0
        n_gaussianRF=3#3
        
        #KNN parameters
        window_size=70
        vecinos=7
        hoja_size=2        

        #GLobal parameters
        pretrain_size=12500
        
        detection=True
        n_wait=500#1
        
    elif data==8:#sinteticos: circleG
    
        #GRF parameters
        if sev==1 and sp==1:            
            gamma=2.0#2.0
            n_gaussianRF=3#7#Cuanto mas sale mejor
        elif sev==1 and sp==3:            
            gamma=2.0#2.0
            n_gaussianRF=3#7#Cuanto mas sale mejor
        elif sev==3 and sp==1:            
            gamma=2.0#2.0
            n_gaussianRF=3#7#Cuanto mas sale mejor
        elif sev==3 and sp==3:            
            gamma=2.0#2.0
            n_gaussianRF=3#7#Cuanto mas sale mejor
                    
        #KNN parameters
        window_size=10
        vecinos=3
        hoja_size=2
        
        #GLobal parameters
        pretrain_size=250
    
    elif data==9:#sinteticos: line
    
        #GRF parameters
        gamma=2.0#1.7
        n_gaussianRF=3#5
                    
        #KNN parameters
        window_size=10
        vecinos=3
        hoja_size=2
        
        #GLobal parameters
        pretrain_size=250
        
    elif data==10:#sinteticos: sineH
    
        #GRF parameters
        gamma=2.0#1.7
        n_gaussianRF=3#5
                    
        #KNN parameters
        window_size=10
        vecinos=3
        hoja_size=2
        
        #GLobal parameters
        pretrain_size=250
        
    elif data==11:#sinteticos: sine
    
        #GRF parameters
        gamma=2.0#1.7
        n_gaussianRF=3#5
                    
        #KNN parameters
        window_size=10
        vecinos=3
        hoja_size=2
        
        #GLobal parameters
        pretrain_size=250        

    elif data==12 or data==13 or data==14 or data==15 or data==16 or data==17 or data==18 or data==19:#sinteticos extendidos
    
        #GRF parameters
        gamma=2.0#2.0
        n_gaussianRF=3#3
                    
        #KNN parameters
        window_size=10
        vecinos=3
        hoja_size=2
        
        #GLobal parameters
        pretrain_size=12500  
        
        n_wait=500#1
        
        
    grf_params=[gamma,n_gaussianRF]
    knn_params=[window_size,vecinos,hoja_size]
    global_params=[pretrain_size]
        
    params=[grf_params,knn_params,global_params,detection,n_wait]#[[gamma,n_gaussianRF],[window_size,vecinos,hoja_size],[pretrain_size],detection,n_wait]
    
    return params

def cargaClassifiers(params,n_classes):

    gamma=params[0][0]
    n_gaussianRF=params[0][1]
    window_size=params[1][0]
    vecinos=params[1][1]
    hoja_size=params[1][2]
    
    #KNN and GRF_KNN
    clf_1 = KNN(n_neighbors=vecinos, leaf_size=hoja_size, max_window_size=window_size)
    
    clf_2 = GRF_KNN(n_neighbors=vecinos, leaf_size=hoja_size, max_window_size=window_size)
    clf_2.gamma=gamma
    clf_2.n_gaussianRF=n_gaussianRF
    
    #HoeffdingTree, HoeffdingTree_GRF
    clf_3 = HoeffdingTree()
    
    clf_4=GRF_HoeffdingTree()
    clf_4.gamma=gamma
    clf_4.n_gaussianRF=n_gaussianRF
    
    #HoeffdingAdaptiveTree and GRF_HoeffdingAdaptiveTree
    clf_5=HAT()
    
    clf_6=GRF_HoeffdingAdaptiveTree()
    clf_6.gamma=gamma
    clf_6.n_gaussianRF=n_gaussianRF
    
    #NaiveBayes and GRF_NaiveBayes
#    clf_7=NaiveBayes()
#    
#    clf_8=GRF_NaiveBayes()
#    clf_8.gamma=gamma
#    clf_8.n_gaussianRF=n_gaussianRF

    #GNB and GRF_GNB
    clf_9=GaussianNB()
    
    clf_10=GRF_GaussianNB()
    clf_10.gamma=gamma
    clf_10.n_gaussianRF=n_gaussianRF

    #SGDClassifier and GRF_SGDClassifier
    clf_11=SGDClassifier(max_iter=1)
    
    clf_12=GRF_SGDClassifier(max_iter=1)
    clf_12.gamma=gamma
    clf_12.n_gaussianRF=n_gaussianRF

    #Perceptron and GRF_Perceptron
    clf_13=SGDClassifier(loss='perceptron', eta0=1, learning_rate='constant', penalty=None,max_iter=1) 
    
    clf_14=GRF_SGDClassifier(loss='perceptron', eta0=1, learning_rate='constant', penalty=None,max_iter=1)
    clf_14.gamma=gamma
    clf_14.n_gaussianRF=n_gaussianRF
    
    #PassiveAggressiveClassifier and GRF_PassiveAggressiveClassifier
    clf_15=PassiveAggressiveClassifier(max_iter=1)
    
    clf_16=GRF_PassiveAggressiveClassifier(max_iter=1)
    clf_16.gamma=gamma
    clf_16.n_gaussianRF=n_gaussianRF
    
    #MLPClassifier and GRF_MLPClassifier
    clf_17=MLPClassifier(batch_size=1,max_iter=1,hidden_layer_sizes=(100,))
    
    clf_18=GRF_MLPClassifier(batch_size=1,max_iter=1,hidden_layer_sizes=(100,))
    clf_18.gamma=gamma
    clf_18.n_gaussianRF=n_gaussianRF
    
    classifiers = [clf_1,clf_2,clf_3,clf_4,clf_5,clf_6,clf_9,clf_10,clf_11,clf_12,clf_13,clf_14,clf_15,clf_16,clf_17,clf_18]
    classifiers_init = [clf_1,clf_2,clf_3,clf_4,clf_5,clf_6,clf_9,clf_10,clf_11,clf_12,clf_13,clf_14,clf_15,clf_16,clf_17,clf_18]

#    classifiers = [clf_1,clf_2]
#    classifiers_init = [clf_1,clf_2]
    
    names=[]
    for c in range(len(classifiers)):
        classifier=classifiers[c]
        class_name=''
        
        if str(classifier)[26:33]=='GRF_KNN':    
            class_name=str(classifier)[26:33]
        elif str(classifier)[22:25]=='KNN':    
            class_name=str(classifier)[22:25]
        elif str(classifier)[34:47]=='HoeffdingTree':
            class_name='HT'
        elif str(classifier)[38:55]=='GRF_HoeffdingTree':
            class_name='GRF_HT'
        elif str(classifier)[43:46]=='HAT':
            class_name=str(classifier)[43:46]
        elif str(classifier)[47:72]=='GRF_HoeffdingAdaptiveTree':
            class_name='GRF_HAT'
#        elif str(classifier)[31:41]=='NaiveBayes':
#            class_name='MNB'            
#        elif str(classifier)[35:49]=='GRF_NaiveBayes':
#            class_name='GRF_MNB'
        elif str(classifier)[0:10]=='GaussianNB':
            class_name='GNB'
        elif str(classifier)[0:14]=='GRF_GaussianNB':
            class_name='GRF_GNB'
        elif str(classifier)[0:13]=='SGDClassifier' and classifier.loss=='hinge':
            class_name='SGD'
        elif str(classifier)[0:17]=='GRF_SGDClassifier' and classifier.loss=='hinge':
            class_name='GRF_SGD'
        elif str(classifier)[0:13]=='SGDClassifier' and classifier.loss=='perceptron':
            class_name='Perceptron'
        elif str(classifier)[0:17]=='GRF_SGDClassifier' and classifier.loss=='perceptron':
            class_name='GRF_Perceptron'
        elif str(classifier)[0:27]=='PassiveAggressiveClassifier':
            class_name='PA'
        elif str(classifier)[0:31]=='GRF_PassiveAggressiveClassifier':
            class_name='GRF_PA'
        elif str(classifier)[0:13]=='MLPClassifier':
            class_name='MLP'
        elif str(classifier)[0:17]=='GRF_MLPClassifier':
            class_name='GRF_MLP'
#        elif str(classifier)[0:9]=='OnlineGRF':
#            class_name=str(classifier)[0:9]
    
        names.append(class_name)
    
    return classifiers,names,classifiers_init

def cargaDetectores(num_detectors,detector):
    
    drift_detectors = []
    drifts_detected=[]
    
    for d in range(num_detectors):
        drift_detectors.append(detector)
        drifts_detected.append([])
    
    return drift_detectors,drifts_detected


def plot_results(ACCURACIES_R,KAPPAS_R,TRAINING_TS_R,TESTING_TS_R,TOTALS_R,DRIFTS_R,data_name,data,severity,speed,names):
    
    mean_accuracies=np.round(np.mean(np.array(ACCURACIES_R),axis=0),3)
    std_accuracies=np.round(np.std(np.array(ACCURACIES_R),axis=0),3)

    mean_kappas=np.round(np.mean(np.array(KAPPAS_R),axis=0),3)
    std_kappas=np.round(np.std(np.array(KAPPAS_R),axis=0),3)
    
    mean_training_ts=np.round(np.mean(np.array(TRAINING_TS_R),axis=0),3)
    std_training_ts=np.round(np.std(np.array(TRAINING_TS_R),axis=0),3)

    mean_testing_ts=np.round(np.mean(np.array(TESTING_TS_R),axis=0),3)
    std_testing_ts=np.round(np.std(np.array(TESTING_TS_R),axis=0),3)

    mean_totals=np.round(np.mean(np.array(TOTALS_R),axis=0),3)
    std_totals=np.round(np.std(np.array(TOTALS_R),axis=0),3)

    mean_drifts=np.round(np.mean(np.array(DRIFTS_R),axis=0),3)
    std_drifts=np.round(np.std(np.array(DRIFTS_R),axis=0),3)
    
    t = Texttable()
    
    if data>=8:
        print ('Results for dataset ',data_name+'_'+str(severity)+str(speed),' are:')            
    else:    
        print ('Results for dataset ',data_name,' are:')    
        
    for c in range(len(classifiers)):
        t.add_rows([['METHODS OF BENCHMARK', 'Accuracy','Kappa','Training time','Testing time','Total time','Drifts'],[names[c],str(mean_accuracies[c])+str('+-')+str(std_accuracies[c]),str(mean_kappas[c])+str('+-')+str(std_kappas[c]),str(mean_training_ts[c])+str('+-')+str(std_training_ts[c]),str(mean_testing_ts[c])+str('+-')+str(std_testing_ts[c]),str(mean_totals[c])+str('+-')+str(std_totals[c]),str(mean_drifts[c])+str('+-')+str(std_drifts[c])]])
    
    print (t.draw())

def plot_final_curves(KAPPAS_R,DRIFTS_DETECTED_R,data_name,data,time_steps,ylim,pre_training_size,model1,model2):
    
    mean_kappas=np.round(np.mean(np.array(KAPPAS_R),axis=0),3)
#    std_kappas=np.round(np.std(np.array(KAPPAS_R),axis=0),3)
    
    plt.figure(figsize=(size_X,size_Y))
    plt.title(title)
    plt.xlabel('Samples')
    plt.ylabel(ejeylabel)
    plt.xlim(0,time_steps)
    plt.ylim(0.0,ylim)
    
    #Shaded area optimization and pre-training
    plt.axvspan(0, pre_training_size, color='b', alpha=0.2, lw=0)  

    plt.plot(mean_kappas[model1],label='Kappa evolution model 1', color='b')
    plt.plot(mean_kappas[model2],label='Kappa evolution model 2', color='g')

    for d in range(len(DRIFTS_DETECTED_R[0][model1])):
        plt.axvline(x=DRIFTS_DETECTED_R[0][model1][d],color='b', linestyle='--')     
    for d in range(len(DRIFTS_DETECTED_R[0][model2])):
        plt.axvline(x=DRIFTS_DETECTED_R[0][model2][d],color='g', linestyle='--')     

    plt.legend(loc='lower right')   
    plt.show()


def save_data(output_pickle,data,ACCURACIES_R,KAPPAS_R,TRAINING_TS_R,TESTING_TS_R,TOTALS_R,DRIFTS_R):
    
    #SAVING DATA#    
    output = open(output_pickle+'ACCURACIES_R_data_'+str(data)+'.pkl', 'wb')
    pickle.dump(ACCURACIES_R, output)
    output.close()
    sio.savemat(output_pickle+'ACCURACIES_R_data_'+str(data)+'.mat', {'ACCURACIES_R_data_'+str(data):ACCURACIES_R})

    output = open(output_pickle+'KAPPAS_R_data_'+str(data)+'.pkl', 'wb')
    pickle.dump(KAPPAS_R, output)
    output.close()
    sio.savemat(output_pickle+'KAPPAS_R_data_'+str(data)+'.mat', {'KAPPAS_R_data_'+str(data):KAPPAS_R})
    
    output = open(output_pickle+'TRAINING_TS_R_data_'+str(data)+'.pkl', 'wb')
    pickle.dump(TRAINING_TS_R, output)
    output.close()
    sio.savemat(output_pickle+'TRAINING_TS_R_data_'+str(data)+'.mat', {'TRAINING_TS_R_data_'+str(data):TRAINING_TS_R})
    
    output = open(output_pickle+'TESTING_TS_R_data_'+str(data)+'.pkl', 'wb')
    pickle.dump(TESTING_TS_R, output)
    output.close()
    sio.savemat(output_pickle+'TESTING_TS_R_data_'+str(data)+'.mat', {'TESTING_TS_R_data_'+str(data):TESTING_TS_R})
    
    output = open(output_pickle+'TOTALS_R_data_'+str(data)+'.pkl', 'wb')
    pickle.dump(TOTALS_R, output)
    output.close()
    sio.savemat(output_pickle+'TOTALS_R_data_'+str(data)+'.mat', {'TOTALS_R_data_'+str(data):TOTALS_R})

    output = open(output_pickle+'DRIFTS_R_data_'+str(data)+'.pkl', 'wb')
    pickle.dump(DRIFTS_R, output)
    output.close()
    sio.savemat(output_pickle+'DRIFTS_R_data_'+str(data)+'.mat', {'DRIFTS_R_data_'+str(data):DRIFTS_R})

def load_data(output_pickle,data):
    
    fil = open(output_pickle+'ACCURACIES_R_data_'+str(data)+'.pkl','rb')
    ACCURACIES_R = pickle.load(fil)
    fil.close()

    fil = open(output_pickle+'KAPPAS_R_data_'+str(data)+'.pkl','rb')
    KAPPAS_R = pickle.load(fil)
    fil.close()

    fil = open(output_pickle+'TRAINING_TS_R_data_'+str(data)+'.pkl','rb')
    TRAINING_TS_R = pickle.load(fil)
    fil.close()

    fil = open(output_pickle+'TESTING_TS_R_data_'+str(data)+'.pkl','rb')
    TESTING_TS_R = pickle.load(fil)
    fil.close()

    fil = open(output_pickle+'TOTALS_R_data_'+str(data)+'.pkl','rb')
    TOTALS_R = pickle.load(fil)
    fil.close()

    fil = open(output_pickle+'DRIFTS_R_data_'+str(data)+'.pkl','rb')
    DRIFTS_R = pickle.load(fil)
    fil.close()    
    
    return ACCURACIES_R,KAPPAS_R,TRAINING_TS_R,TESTING_TS_R,TOTALS_R,DRIFTS_R
    
def plot_mcnemar_evolution(size_X,size_Y,title,drift,pretraining_size,drift_period,mcnemar_values,ejeylabel,ylim,limite,output_res,data,model1,model2,n_wait):

    fig=plt.figure(figsize=(size_X,size_Y))
    ax = fig.add_subplot(111)
#    plt.title(title)
#    ax.set_title("My Plot Title")
    ax.set_xlabel("Samples", fontsize=32)
    ax.set_ylabel(ejeylabel, fontsize=32)
        
    plt.setp(ax.get_xticklabels(), fontsize=32)
    plt.setp(ax.get_yticklabels(), fontsize=32)
    
#    plt.xlim(0,len(mcnemar_values))
#    plt.ylim(0.0,ylim)
    

    plt.plot(mcnemar_values)
    plt.plot(mcnemar_values,label=model1+'_vs_'+model2,color='b')

    plt.axhline(y=limite,color='k', linestyle='--')     

    plt.legend(loc='upper right',prop={'size': 16})   
    plt.show()
    
    fig.savefig(output_res+'/mcnemar_data'+str(data)+'_'+model1+'_vs_'+model2+'_'+str(n_wait)+'.svg', bbox_inches='tight')
    fig.savefig(output_res+'/mcnemar_data'+str(data)+'_'+model1+'_vs_'+model2+'_'+str(n_wait)+'.pdf', bbox_inches='tight')
    
#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
############################################################## DATA ##############################################################################
#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

datasets=['weather','elec','covtype','moving_squares','sea_stream','usenet2','gmsc','airlines',
         'circleG','line','sineH','sine','circleG_concept1','circleG_concept2','line_concept1',
         'line_concept2','sineH_concept1','sineH_concept2','sine_concept1','sine_concept2' 
         ]

data=13#Dataset selection
lim_data=50000#Number of samples to deal with

#In case of synthetic datasets: 'circleG','line','sineH','sine'
severity=3
speed=1

#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
############################################################## PARAMETERS AND VARIABLES ##############################################################################
#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

#Process variables
batch_size=1
n_wait=0
metrics=['accuracy','kappa']
output_res='your_path'
output_pickle='your_path'
detector=ADWIN()

runs=2
ACCURACIES_R=[]
KAPPAS_R=[]
TRAINING_TS_R=[]
TESTING_TS_R=[]
TOTALS_R=[]
DRIFTS_R=[]
DRIFTS_DETECTED_R=[]

#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
############################################################## PROCESS ##############################################################################
#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

with warnings.catch_warnings():
    
    warnings.simplefilter("ignore")
    fxn()

    #Loading dataset
    stream=cargaDatos(datasets,data,severity,speed,lim_data)
    
    for r in range(runs):
    
        print ('######### Running loop ',r,' #########')
    
        ACCURACIES=[]
        KAPPAS=[]
        TRAINING_TS=[]
        TESTING_TS=[]
        TOTALS=[]
        DRIFTS=[]
        SPIKES=[]
    
        #Classifiers setup
        params=cargaParametros(data,severity,speed)#[[gamma,n_gaussianRF],[window_size,vecinos,hoja_size],[pretrain_size]]
        classifiers,names,classifiers_init=cargaClassifiers(params,stream.n_classes)
        n_wait=params[4]
        
        #Variables para McNemar test
        MCNEMARS_preds=[]
        MCNEMARS_trues=[]
        for c in range(len(classifiers)):
            MCNEMARS_preds.append([])
            MCNEMARS_trues.append([])
            SPIKES.append([])
        
        #Drift detectors setup
        drift_detectors,drifts_detected=cargaDetectores(len(classifiers),detector)
        
        # Setup the evaluator
        output_file=''
        if data==8 or data==9 or data==10 or data==11:
            output_file=output_res+'results_'+datasets[data]+'_'+str(severity)+str(speed)+'.csv'
        else:
            output_file=output_res+'results_'+datasets[data]+'.csv'
            
        evaluator = EvaluatePrequential_NN(pretrain_size=params[2][0], batch_size=batch_size, n_wait=n_wait,output_file=output_file, show_plot=False, metrics=metrics,drift_detectors=drift_detectors,drifts_detected=drifts_detected,classifiers_init=classifiers_init,detection=params[3],MCNEMARS_preds=MCNEMARS_preds,MCNEMARS_trues=MCNEMARS_trues,SPIKES=SPIKES)
        
        # Run evaluation
        evaluator.evaluate(stream=stream, model=classifiers,model_names=names)
    
        #Run update
        for c in range(len(classifiers)):
            accuracy=evaluator.mean_eval_measurements[c].get_accuracy()
            kappa=evaluator.mean_eval_measurements[c].get_kappa()
            training_t=evaluator.running_time_measurements[c]._training_time
            testing_t=evaluator.running_time_measurements[c]._testing_time
            total_t=evaluator.running_time_measurements[c].get_current_total_running_time()
            drifts=len(drifts_detected[c])
        
            ACCURACIES.append(accuracy)
            KAPPAS.append(kappa)
            TRAINING_TS.append(training_t)
            TESTING_TS.append(testing_t)
            TOTALS.append(total_t)     
            DRIFTS.append(drifts)            

        ACCURACIES_R.append(ACCURACIES)
        KAPPAS_R.append(KAPPAS)
        TRAINING_TS_R.append(TRAINING_TS)
        TESTING_TS_R.append(TESTING_TS)
        TOTALS_R.append(TOTALS)                
        DRIFTS_R.append(DRIFTS)                
        DRIFTS_DETECTED_R.append(drifts_detected)                
           
    
    #Save data to .pickle and .mat
    save_data(output_pickle,data,ACCURACIES_R,KAPPAS_R,TRAINING_TS_R,TESTING_TS_R,TOTALS_R,DRIFTS_R)
    
    #Load data from .pickle
    ACCURACIES_R,KAPPAS_R,TRAINING_TS_R,TESTING_TS_R,TOTALS_R,DRIFTS_R=load_data(output_pickle,data)
        
    # Plot Results    
    plot_results(ACCURACIES_R,KAPPAS_R,TRAINING_TS_R,TESTING_TS_R,TOTALS_R,DRIFTS_R,datasets[data],data,severity,speed,names)
    
    #McNemar tests entre 2 modelos
#    pair_comparisons=[[0,1],[2,3],[4,5],[6,7],[8,9],[10,11],[12,13],[14,15],[16,17]]
    pair_comparisons=[[0,1],[2,3],[4,5],[8,9],[10,11],[12,13],[14,15],[16,17]]
    
    for cp in range(len(pair_comparisons)):
    
        pair=pair_comparisons[cp]
    
        model1=pair[0]
        model2=pair[1]
        
        list_predictions_model1=[]
        for each in window(MCNEMARS_preds[model1], n_wait):
            list_preds=list(each)
            list_predictions_model1.append(list_preds)
    
        list_predictions_model2=[]
        for each in window(MCNEMARS_preds[model2], n_wait):
            list_preds=list(each)
            list_predictions_model2.append(list_preds)
    
        list_reales=[]
        for each in window(MCNEMARS_trues[model1], n_wait):
            list_trues=list(each)
            list_reales.append(list_trues)
            
        mcnemar_statistics=[]    
        mcnemar_ps=[]    
        for i in range(len(list_reales)):
            preds_m1=list_predictions_model1[i]
            preds_m2=list_predictions_model2[i]
            verdades=list_reales[i]
            
            acierto_m1_acierto_m2=0
            acierto_m1_fallo_m2=0
            fallo_m1_acierto_m2=0
            fallo_m1_fallo_m2=0
            
            for j in range(len(verdades)):
                verdad=verdades[j]
                pred_m1=preds_m1[j]
                pred_m2=preds_m2[j]
                
                if pred_m1==verdad and pred_m2==verdad:
                    acierto_m1_acierto_m2+=1
                elif pred_m1==verdad and pred_m2!=verdad:
                    acierto_m1_fallo_m2+=1
                elif pred_m1!=verdad and pred_m2==verdad:
                    fallo_m1_acierto_m2+=1
                elif pred_m1!=verdad and pred_m2!=verdad:
                    fallo_m1_fallo_m2+=1
                    
            #https://machinelearningmastery.com/mcnemars-test-for-machine-learning/                
#            contingency_table=[[acierto_m1_acierto_m2,acierto_m1_fallo_m2],[fallo_m1_acierto_m2,fallo_m1_fallo_m2]]
#            result = mcnemar(contingency_table, exact=False, correction=True)
#            mcnemar_statistics.append(result.statistic)
#            mcnemar_ps.append(result.pvalue)

            #McNemar test a mano: statistic = (Yes/No - No/Yes)^2 / (Yes/No + No/Yes)
            if acierto_m1_fallo_m2==0 and fallo_m1_acierto_m2==0:
                statistic=0
            else:
                statistic=(acierto_m1_fallo_m2-fallo_m1_acierto_m2)**2/(acierto_m1_fallo_m2+fallo_m1_acierto_m2)
            mcnemar_statistics.append(statistic)
    
    
        # Plot McNemar test evolution 
        
        drift_period=0
        drift=1000
        if data==8 or data==9 or data==10 or data==11:
            if speed==1:
                drift_period=50
            elif speed==2:
                drift_period=250
            elif speed==3:
                drift_period=500
        
        
        size_X=20
        size_Y=10
        ejeylabel='M statistic (sliding window='+str(n_wait)+')'
        ylim=1.0
        title='McNemar statistic evolution for the problem: '+str(datasets[data])
#        threshold=6.635#confidence level of 0.99
        threshold=3.841459#confidence level of 0.95
        
        plot_mcnemar_evolution(size_X,size_Y,title,drift,params[2][0],drift_period,mcnemar_statistics,ejeylabel,ylim,threshold,output_res,datasets[data],names[model1],names[model2],n_wait)

        #MCNEMAR % rechazo Hipotesis nula:
        n_rechazos=0
        for p in range(len(mcnemar_statistics)):
            if mcnemar_statistics[p]>threshold:
                n_rechazos+=1
                
        print ('Se rechaza la hipótesis nula en un ',np.round((n_rechazos/len(mcnemar_statistics))*100,2),'%')
    
    ##################################PLOTS heatmaps
    neurons_C0=[]
    neurons_C1=[]
    modelo=1#Tiene que ser uno de los que aplica GRF, claro
    lim=1000#Vamos a contar con las primeras 1000 samples
    output_img='your_path'
    
    for i in range(len(stream.y)):
        lab=stream.y[i]
        if lab==0:
            neurons_C0.append(SPIKES[modelo][i])
        elif lab==1:
            neurons_C1.append(SPIKES[modelo][i])
                   
    #FIGURE 3: EXAMPLE
#    sns.set(font_scale=1.4)
#    fig, (ax,ax2) = plt.subplots(ncols=2,figsize=(20,10))
#    fig.subplots_adjust(wspace=0.01)
#
#    sns.heatmap(neurons_C0[0:lim], ax=ax, cbar=False,vmax=0.99,vmin=0.0)
#    fig.colorbar(ax.collections[0], ax=ax,location="left", use_gridspec=False, pad=0.15)
#
#    sns.heatmap(neurons_C1[0:lim], ax=ax2, cbar=False,vmax=0.99,vmin=0.0)
#    fig.colorbar(ax2.collections[0], ax=ax2,location="right", use_gridspec=False, pad=0.15)
#    
#    ax2.yaxis.tick_right()
#    ax2.tick_params(rotation=0)
#    
#    #Dashed lines
#    ax.vlines([params[0][1]], *ax.get_ylim(), linestyle='--',color='w')
#    ax2.vlines([params[0][1]], *ax2.get_ylim(), linestyle='--',color='w')
#    
#    #Text
#    ax.text(1.5, -15.15,'Feature 1', fontsize=14)    
#    ax.text(6.5, -15.15,'Feature 2', fontsize=14)    
#    ax.text(11.5, -15.15,'Feature 1', fontsize=14)    
#    ax.text(17.0, -15.15,'Feature 2', fontsize=14)    
# 
#    ax.set_title(label='Class 0', fontsize=23)
#    ax2.set_title(label='Class 1', fontsize=23)
#    
#    ax.set_ylabel(ylabel='Samples', fontsize=17)
#    ax.set_xlabel(xlabel='GRFs', fontsize=17)    
#    ax2.set_xlabel(xlabel='GRFs', fontsize=17)   
#
#    ax.yaxis.set_major_locator(plt.MaxNLocator(11))
#    ax2.yaxis.set_major_locator(plt.MaxNLocator(11))
#    
#    ax.set_yticklabels(['0','100','200','300','400','500','600','700','800','900','1000'])
#    ax2.set_yticklabels(['0','100','200','300','400','500','600','700','800','900','1000'])
#    
#    plt.show()  
#        
#    fig.savefig(output_img+'GRF_example.svg', bbox_inches='tight')
#    fig.savefig(output_img+'GRF_example.pdf', bbox_inches='tight')
    
    
    
    #FIGURE 4: IMPACT OF GAMMA
    sns.set(font_scale=2.0)
    fig, (ax,ax2) = plt.subplots(ncols=2,figsize=(20,10))
    fig.subplots_adjust(wspace=0.01)

    sns.heatmap(neurons_C0[0:lim], ax=ax, cbar=False,vmax=0.99,vmin=0.0)
#    fig.colorbar(ax.collections[0], ax=ax,location="left", use_gridspec=False, pad=0.15)

    sns.heatmap(neurons_C1[0:lim], ax=ax2, cbar=False,vmax=0.99,vmin=0.0)
#    fig.colorbar(ax2.collections[0], ax=ax2,location="right", use_gridspec=False, pad=0.15)
    
    ax2.yaxis.tick_right()
    ax2.tick_params(rotation=0)
    
    #Dashed lines
    ax.vlines([params[0][1]], *ax.get_ylim(), linestyle='--',color='k')
    ax2.vlines([params[0][1]], *ax2.get_ylim(), linestyle='--',color='k')
    
    #Text
#    ax.text(1.0, -15.15,'Feature 1', fontsize=14)    
#    ax.text(4.0, -15.15,'Feature 2', fontsize=14)    
#    ax.text(7.0, -15.15,'Feature 1', fontsize=14)    
#    ax.text(10.0, -15.15,'Feature 2', fontsize=14)    
# 
#    ax.set_title(label='Class 0', fontsize=23)
#    ax2.set_title(label='Class 1', fontsize=23)
    
#    ax.set_ylabel(ylabel='Samples', fontsize=17)
#    ax.set_xlabel(xlabel='GRFs', fontsize=17)    
#    ax2.set_xlabel(xlabel='GRFs', fontsize=17)   

    ax.yaxis.set_major_locator(plt.MaxNLocator(11))
    ax2.yaxis.set_major_locator(plt.MaxNLocator(11))
    
    ax.set_yticklabels(['0','100','200','300','400','500','600','700','800','900','1000'])
    ax2.set_yticklabels(['0','100','200','300','400','500','600','700','800','900','1000'])
    ax2.set_yticklabels([])
    ax.tick_params(rotation=0)
    ax2.tick_params(rotation=0)
    
    plt.show()  
        
    fig.savefig(output_img+'gamma_impact_'+str(params[0][0])[0]+'_'+str(params[0][0])[2]+'.svg', bbox_inches='tight')
    fig.savefig(output_img+'gamma_impact_'+str(params[0][0])[0]+'_'+str(params[0][0])[2]+'.pdf', bbox_inches='tight')
    
    
    
    #FIGURE 4: IMPACT OF N_GRFs
#    sns.set(font_scale=2.0)
#    fig, (ax,ax2) = plt.subplots(ncols=2,figsize=(20,10))
#    fig.subplots_adjust(wspace=0.01)
#
#    sns.heatmap(neurons_C0[0:lim], ax=ax, cbar=False,vmax=0.99,vmin=0.0)
##    fig.colorbar(ax.collections[0], ax=ax,location="left", use_gridspec=False, pad=0.15)
#
#    sns.heatmap(neurons_C1[0:lim], ax=ax2, cbar=False,vmax=0.99,vmin=0.0)
##    fig.colorbar(ax2.collections[0], ax=ax2,location="right", use_gridspec=False, pad=0.15)
#    
#    ax2.yaxis.tick_right()
#    ax2.tick_params(rotation=0)
#    
#    #Dashed lines
#    ax.vlines([params[0][1]], *ax.get_ylim(), linestyle='--',color='k')
#    ax2.vlines([params[0][1]], *ax2.get_ylim(), linestyle='--',color='k')
#    
#    #Text
##    ax.text(1.0, -15.15,'Feature 1', fontsize=14)    
##    ax.text(4.0, -15.15,'Feature 2', fontsize=14)    
##    ax.text(7.0, -15.15,'Feature 1', fontsize=14)    
##    ax.text(10.0, -15.15,'Feature 2', fontsize=14)    
## 
##    ax.set_title(label='Class 0', fontsize=23)
##    ax2.set_title(label='Class 1', fontsize=23)
#    
##    ax.set_ylabel(ylabel='Samples', fontsize=17)
##    ax.set_xlabel(xlabel='GRFs', fontsize=17)    
##    ax2.set_xlabel(xlabel='GRFs', fontsize=17)   
#
#    ax.yaxis.set_major_locator(plt.MaxNLocator(11))
#    ax2.yaxis.set_major_locator(plt.MaxNLocator(11))
#    
#    ax.set_yticklabels(['0','100','200','300','400','500','600','700','800','900','1000'])
#    ax2.set_yticklabels(['0','100','200','300','400','500','600','700','800','900','1000'])
#    ax2.set_yticklabels([])
#    ax.tick_params(rotation=0)
#    ax2.tick_params(rotation=0)
#
##    ax.xaxis.set_major_locator(plt.MaxNLocator(5))
##    ax2.xaxis.set_major_locator(plt.MaxNLocator(5))    
#    ax.xaxis.set_major_locator(plt.MaxNLocator(7))
#    ax2.xaxis.set_major_locator(plt.MaxNLocator(7))    
##    ax.set_xticklabels(['0','2','4','6','9'])
##    ax2.set_xticklabels(['0','2','4','6','9'])
##    ax.set_xticklabels(['0','7','14','21','29','36','49'])
##    ax2.set_xticklabels(['0','7','14','21','29','36','49'])
#    ax.set_xticklabels(['0','14','28','56','70','84','99'])
#    ax2.set_xticklabels(['0','14','28','56','70','84','99'])
#    ax.tick_params(rotation=0)
#    ax2.tick_params(rotation=0)
#    
#    plt.show()  
#        
#    fig.savefig(output_img+'GRFs_impact_'+str(params[0][1])+'_gamma_'+str(params[0][0])[0]+'_'+str(params[0][0])[2]+'.svg', bbox_inches='tight')
#    fig.savefig(output_img+'GRFs_impact_'+str(params[0][1])+'_gamma_'+str(params[0][0])[0]+'_'+str(params[0][0])[2]+'.pdf', bbox_inches='tight')
    
