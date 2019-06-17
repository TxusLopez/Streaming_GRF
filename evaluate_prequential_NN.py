import os
import warnings
import numpy as np
import pandas as pd
from numpy import unique
from timeit import default_timer as timer
from skmultiflow.evaluation.base_evaluator import StreamEvaluator
from skmultiflow.utils import constants
import math
import scipy.stats

class EvaluatePrequential_NN(StreamEvaluator):
    """ The prequential evaluation method, or interleaved test-then-train method,
    is an alternative to the traditional holdout evaluation, inherited from
    batch setting problems.

    The prequential evaluation is designed specifically for stream settings,
    in the sense that each sample serves two purposes, and that samples are
    analysed sequentially, in order of arrival, and become immediately
    inaccessible.

    This method consists of using each sample to test the model, which means
    to make a predictions, and then the same sample is used to train the model
    (partial fit). This way the model is always tested on samples that it
    hasn't seen yet.

    Parameters
    ----------
    n_wait: int (Default: 200)
        The number of samples to process between each test. Also defines when to update the plot if `show_plot=True`.
        Note that setting `n_wait` too small can significantly slow the evaluation process.

    max_samples: int (Default: 100000)
        The maximum number of samples to process during the evaluation.

    batch_size: int (Default: 1)
        The number of samples to pass at a time to the model(s).

    pretrain_size: int (Default: 200)
        The number of samples to use to train the model before starting the evaluation. Used to enforce a 'warm' start.

    max_time: float (Default: float("inf"))
        The maximum duration of the simulation (in seconds).

    metrics: list, optional (Default: ['accuracy', 'kappa'])
        | The list of metrics to track during the evaluation. Also defines the metrics that will be displayed in plots
          and/or logged into the output file. Valid options are
        | *Classification*
        | 'accuracy'
        | 'kappa'
        | 'kappa_t'
        | 'kappa_m'
        | 'true_vs_predicted'
        | *Multi-target Classification*
        | 'hamming_score'
        | 'hamming_loss'
        | 'exact_match'
        | 'j_index'
        | *Regression*
        | 'mean_square_error'
        | 'mean_absolute_error'
        | 'true_vs_predicted'
        | *Multi-target Regression*
        | 'average_mean_squared_error'
        | 'average_mean_absolute_error'
        | 'average_root_mean_square_error'

    output_file: string, optional (Default: None)
        File name to save the summary of the evaluation.

    show_plot: bool (Default: False)
        If True, a plot will show the progress of the evaluation. Warning: Plotting can slow down the evaluation
        process.

    restart_stream: bool, optional (default: True)
        If True, the stream is restarted once the evaluation is complete.

    data_points_for_classification: bool(Default: False)
        If True , the visualization used is a cloud of data points
        (only works for classification)

    Notes
    -----
    1. This evaluator can process a single learner to track its performance; or multiple learners  at a time, to
       compare different models on the same stream.

    2. The metric 'true_vs_predicted' is intended to be informative only. It corresponds to evaluations at a specific
       moment which might not represent the actual learner performance across all instances.

    Examples
    --------
    >>> # The first example demonstrates how to evaluate one model
    >>> from skmultiflow.data import SEAGenerator
    >>> from skmultiflow.trees import HoeffdingTree
    >>> from skmultiflow.evaluation import EvaluatePrequential
    >>>
    >>> # Set the stream
    >>> stream = SEAGenerator(random_state=1)
    >>> stream.prepare_for_use()
    >>>
    >>> # Set the model
    >>> ht = HoeffdingTree()
    >>>
    >>> # Set the evaluator
    >>>
    >>> evaluator = EvaluatePrequential(max_samples=10000,
    >>>                                 max_time=1000,
    >>>                                 show_plot=True,
    >>>                                 metrics=['accuracy', 'kappa'])
    >>>
    >>> evaluator.evaluate(stream=stream, model=ht, model_names=['HT'])
    >>>
    >>> # Run evaluation
    >>> evaluator.evaluate(stream=stream, model=ht, model_names=['HT'])

    >>> # The second example demonstrates how to compare two models
    >>> from skmultiflow.data import SEAGenerator
    >>> from skmultiflow.trees import HoeffdingTree
    >>> from skmultiflow.bayes import NaiveBayes
    >>> from skmultiflow.evaluation import EvaluateHoldout
    >>>
    >>> # Set the stream
    >>> stream = SEAGenerator(random_state=1)
    >>> stream.prepare_for_use()
    >>>
    >>> # Set the models
    >>> ht = HoeffdingTree()
    >>> nb = NaiveBayes()
    >>>
    >>> evaluator = EvaluatePrequential(max_samples=10000,
    >>>                                 max_time=1000,
    >>>                                 show_plot=True,
    >>>                                 metrics=['accuracy', 'kappa'])
    >>>
    >>> # Run evaluation
    >>> evaluator.evaluate(stream=stream, model=[ht, nb], model_names=['HT', 'NB'])

    >>> # The third example demonstrates how to evaluate one model
    >>> # and visualize the predictions using data points.
    >>> # Note: You can not in this case compare multiple models
    >>> from skmultiflow.data import SEAGenerator
    >>> from skmultiflow.trees import HoeffdingTree
    >>> from skmultiflow.evaluation import EvaluatePrequential
    >>> # Set the stream
    >>> stream = SEAGenerator(random_state=1)
    >>> stream.prepare_for_use()
    >>> # Set the model
    >>> ht = HoeffdingTree()
    >>> # Set the evaluator
    >>> evaluator = EvaluatePrequential(max_samples=200,
    >>>                                 n_wait=1,
    >>>                                 pretrain_size=1,
    >>>                                 max_time=1000,
    >>>                                 show_plot=True,
    >>>                                 metrics=['accuracy'],
    >>>                                 data_points_for_classification=True)
    >>> evaluator.evaluate(stream=stream, model=ht, model_names=['HT'])
    >>> # Run evaluation
    >>> evaluator.evaluate(stream=stream, model=ht, model_names=['HT'])

    """

    def __init__(self,
                 n_wait=200,
                 max_samples=100000,
                 batch_size=1,
                 pretrain_size=200,
                 max_time=float("inf"),
                 metrics=None,
                 output_file=None,
                 show_plot=False,
                 restart_stream=True,
                 data_points_for_classification=False,drift_detectors=None,drifts_detected=None,classifiers_init=None,detection=True,MCNEMARS_preds=None,MCNEMARS_trues=None,SPIKES=None):

        super().__init__()
        self._method = 'prequential'
        self.n_wait = n_wait
        self.max_samples = max_samples
        self.pretrain_size = pretrain_size
        self.batch_size = batch_size
        self.max_time = max_time
        self.output_file = output_file
        self.show_plot = show_plot
        self.data_points_for_classification = data_points_for_classification
        self.drift_detectors=drift_detectors
        self.drifts_detected=drifts_detected
        self.classifiers_init=classifiers_init
        self.detection=detection
        self.MCNEMARS_preds=MCNEMARS_preds
        self.MCNEMARS_trues=MCNEMARS_trues
        self.SPIKES=SPIKES

        if metrics is None and data_points_for_classification is False:
            self.metrics = [constants.ACCURACY, constants.KAPPA]

        elif data_points_for_classification is True:
            self.metrics = [constants.DATA_POINTS]

        else:
            self.metrics = metrics

        self.restart_stream = restart_stream
        self.n_sliding = n_wait

        warnings.filterwarnings("ignore", ".*invalid value encountered in true_divide.*")
        warnings.filterwarnings("ignore", ".*Passing 1d.*")

    def evaluate(self, stream, model, model_names=None):
        """ Evaluates a learner or set of learners on samples from a stream.

        Parameters
        ----------
        stream: Stream
            The stream from which to draw the samples.

        model: StreamModel or list
            The learner or list of learners to evaluate.

        model_names: list, optional (Default=None)
            A list with the names of the learners.

        Returns
        -------
        StreamModel or list
            The trained learner(s).

        """
        self._init_evaluation(model=model, stream=stream, model_names=model_names)

        if self._check_configuration():
            self._reset_globals()
            # Initialize metrics and outputs (plots, log files, ...)
            self._init_metrics()
            self._init_plot()
            self._init_file()

            self.model = self._train_and_test()

            if self.show_plot:
                self.visualizer.hold()

            return self.model

    def _train_and_test(self):
        """ Method to control the prequential evaluation.

        Returns
        -------
        BaseClassifier extension or list of BaseClassifier extensions
            The trained classifiers.

        Notes
        -----
        The classifier parameter should be an extension from the BaseClassifier. In
        the future, when BaseRegressor is created, it could be an extension from that
        class as well.

        """
        self._start_time = timer()
        self._end_time = timer()
        print('Prequential Evaluation')
        print('Evaluating {} target(s).'.format(self.stream.n_targets))

        actual_max_samples = self.stream.n_remaining_samples()
        if actual_max_samples == -1 or actual_max_samples > self.max_samples:
            actual_max_samples = self.max_samples

        first_run = True
        if self.pretrain_size > 0:
            print('Pre-training on {} sample(s).'.format(self.pretrain_size))

            X, y = self.stream.next_sample(self.pretrain_size)            

            for i in range(self.n_models):
                if self._task_type == constants.CLASSIFICATION:
                    
                    #If estimator is based on Gaussian Receptive Fileds Population Encoding
                    if str(self.model[i])[26:33]=='GRF_KNN' or str(self.model[i])[38:55]=='GRF_HoeffdingTree' or str(self.model[i])[47:72]=='GRF_HoeffdingAdaptiveTree' or str(self.model[i])[0:14]=='GRF_GaussianNB' or str(self.model[i])[0:17]=='GRF_SGDClassifier' or str(self.model[i])[0:31]=='GRF_PassiveAggressiveClassifier' or str(self.model[i])[0:17]=='GRF_MLPClassifier':
                        # Training time computation
                        self.running_time_measurements[i].compute_training_time_begin()

                        spikes=[]  
                        #Calculo de min_max_data   
                        min_max_data=np.zeros((X.shape[1],2))
                        
                        df_pre=pd.DataFrame(X)
        
                        for dim in range(X.shape[1]):
                            feat_min_data=np.min(df_pre.iloc[:,dim])
                            feat_max_data=np.max(df_pre.iloc[:,dim])
                        
                            min_max_data[dim]=(feat_min_data,feat_max_data)
        
                        for j in range(X.shape[0]):
                            #Convertimos las features en spikes
                            cut_points=[]
                            for d in range(X.shape[1]):
                                cut_points_feat=self.get_response_times(d,self.model[i].gamma,self.model[i].n_gaussianRF,min_max_data,X[j],self.model[i].time_coding)    
                                cut_points=np.concatenate((cut_points,np.array(cut_points_feat)), axis=0)                
                        
                            spikes.append(list(cut_points))
                            self.SPIKES[i].append(list(cut_points))
                            
                            
                        self.model[i].min_max_data=min_max_data
                                            
                        # Training time computation
                        self.model[i].partial_fit(X=spikes, y=y, classes=self.stream.target_values)
                        self.running_time_measurements[i].compute_training_time_end()

#                    elif str(self.model[i])[0:9]=='OnlineGRF':
#
#                        # Training time computation
#                        self.running_time_measurements[i].compute_training_time_begin()
#
#                        spikes=[]                                 
#                        #Calculo de min_max_data   
#                        min_max_data=np.zeros((X.shape[1],2))
#                        
#                        df_pre=pd.DataFrame(X)
#        
#                        for dim in range(X.shape[1]):
#                            feat_min_data=np.min(df_pre.iloc[:,dim])
#                            feat_max_data=np.max(df_pre.iloc[:,dim])
#                        
#                            min_max_data[dim]=(feat_min_data,feat_max_data)
#        
#                        for j in range(X.shape[0]):
#                            #Convertimos las features en spikes
#                            cut_points=[]
#                            for d in range(X.shape[1]):
#                                cut_points_feat=self.get_response_times(d,self.model[i].gamma,self.model[i].n_gaussianRF,min_max_data,X[j],self.model[i].time_coding)    
#                                cut_points=np.concatenate((cut_points,np.array(cut_points_feat)), axis=0)                
#                        
#                            spikes.append(list(cut_points))
#                            self.SPIKES[i].append(list(cut_points))
#                            
#                            
#                        self.model[i].min_max_data=min_max_data
#                                            
#                        # Training time computation
#                        self.model[i].partial_fit_pre_train(X=spikes, y=y, classes=self.stream.target_values)
#                        self.running_time_measurements[i].compute_training_time_end()
                        
                    else:                        
                        # Training time computation
                        self.running_time_measurements[i].compute_training_time_begin()
                        self.model[i].partial_fit(X=X, y=y, classes=self.stream.target_values)
                        self.running_time_measurements[i].compute_training_time_end()
                        
                elif self._task_type == constants.MULTI_TARGET_CLASSIFICATION:
                    
                    self.running_time_measurements[i].compute_training_time_begin()
                    self.model[i].partial_fit(X=X, y=y, classes=unique(self.stream.target_values))
                    self.running_time_measurements[i].compute_training_time_end()
                else:
                    self.running_time_measurements[i].compute_training_time_begin()
                    self.model[i].partial_fit(X=X, y=y)
                    self.running_time_measurements[i].compute_training_time_end()
                self.running_time_measurements[i].update_time_measurements(self.pretrain_size)
            self.global_sample_count += self.pretrain_size
            first_run = False

        update_count = 0
        print('Evaluating...')
        while ((self.global_sample_count < actual_max_samples) & (self._end_time - self._start_time < self.max_time)
               & (self.stream.has_more_samples())):
            try:
                X, y = self.stream.next_sample(self.batch_size)

                if X is not None and y is not None:
                    
                    #Here we can update the min_max_data
                    #TO DO
                    
                    # Test
                    prediction = [[] for _ in range(self.n_models)]
                    for i in range(self.n_models):
                        try:
                                                        
                            spikes=[]              
                            p=[]                               
                            # Testing time                            
                            if str(self.model[i])[26:33]=='GRF_KNN' or str(self.model[i])[38:55]=='GRF_HoeffdingTree' or str(self.model[i])[47:72]=='GRF_HoeffdingAdaptiveTree' or str(self.model[i])[0:14]=='GRF_GaussianNB' or str(self.model[i])[0:17]=='GRF_SGDClassifier' or str(self.model[i])[0:31]=='GRF_PassiveAggressiveClassifier' or str(self.model[i])[0:17]=='GRF_MLPClassifier':
                                
                                # Training time computation
                                self.running_time_measurements[i].compute_testing_time_begin()

                                for j in range(X.shape[0]):
                                    #Convertimos las features en spikes
                                    cut_points=[]
                                    for d in range(X.shape[1]):
                                        cut_points_feat=self.get_response_times(d,self.model[i].gamma,self.model[i].n_gaussianRF,self.model[i].min_max_data,X[j],self.model[i].time_coding)    
                                        cut_points=np.concatenate((cut_points,np.array(cut_points_feat)), axis=0)                
                                
                                    spikes.append(list(cut_points))
                                    self.SPIKES[i].append(list(cut_points))                                 
                                    
                                p=self.model[i].predict(spikes)
                                prediction[i].extend(p)
                                self.running_time_measurements[i].compute_testing_time_end()                                
                                
                            else:                                                        
                                self.running_time_measurements[i].compute_testing_time_begin()
                                p=self.model[i].predict(X)
#                                print ('count: ',update_count)
                                prediction[i].extend(p)                                
                                self.running_time_measurements[i].compute_testing_time_end()

                            #Data for MCNEMAR tests
                            self.MCNEMARS_preds[i].append(p)                                
                            self.MCNEMARS_trues[i].append(y)                                

                            #DRIFT DETECTION   
                            if self.detection==True:

                                self.drift_detectors[i].add_element((p==y)[0])     
                                
                                if self.drift_detectors[i].detected_change():            
                                    self.drifts_detected[i].append(update_count)

                                    #Reset models
                                    if str(self.model[i])[22:25]=='KNN' or str(self.model[i])[26:33]=='GRF_KNN':
                                        #Después de inicializar KNN, hay que pre-entrenarlo
                                        df=pd.DataFrame(X)
                                        num_samp=self.model[i].max_window_size
                                        labls=df[200+update_count-num_samp:200+update_count+1][df.columns[-1]]
                                        feats=df[200+update_count-num_samp:200+update_count+1][df.columns[0:len(df.columns)-1]]
                                        
                                        self.model[i]=self.classifiers_init[i]
                                        self.model[i].partial_fit(np.array(feats), np.array(labls),classes=self.stream.target_values)
                                    
                                    else:                                    
                                        self.model[i]=self.classifiers_init[i]
                                        
                                        #Here we can re-calculate the min_max_data, and pre-train the classifiers
                                        #TO DO                                        

                            else:

                                if update_count==1000:
                                    self.drifts_detected[i].append(update_count)
    
                                    #Reset models
                                    if str(self.model[i])[22:25]=='KNN' or str(self.model[i])[26:33]=='GRF_KNN':
                                        #Después de inicializar KNN, hay que pre-entrenarlo
                                        df=pd.DataFrame(X)
                                        num_samp=self.model[i].max_window_size
                                        labls=df[200+update_count-num_samp:200+update_count+1][df.columns[-1]]
                                        feats=df[200+update_count-num_samp:200+update_count+1][df.columns[0:len(df.columns)-1]]
                                        
                                        self.model[i]=self.classifiers_init[i]
                                        self.model[i].partial_fit(np.array(feats), np.array(labls),classes=self.stream.target_values)
                                    
                                    else:                                    
                                        self.model[i]=self.classifiers_init[i]
#                                        self.model[i].n_gaussianRF=7#Estrategia de adaptacion para los GRF

                        except TypeError:
                            raise TypeError("Unexpected prediction value from {}"
                                            .format(type(self.model[i]).__name__))
                    self.global_sample_count += self.batch_size

                    for j in range(self.n_models):
                        for i in range(len(prediction[0])):
                            self.mean_eval_measurements[j].add_result(y[i], prediction[j][i])
                            self.current_eval_measurements[j].add_result(y[i], prediction[j][i])
                    self._check_progress(actual_max_samples)

                    # Train
                    if first_run:
                    
                        for i in range(self.n_models):
                                                                                    
                            if self._task_type != constants.REGRESSION and \
                               self._task_type != constants.MULTI_TARGET_REGRESSION:
                                                                                       
                                if str(self.model[i])[26:33]=='GRF_KNN' or str(self.model[i])[38:55]=='GRF_HoeffdingTree' or str(self.model[i])[47:72]=='GRF_HoeffdingAdaptiveTree' or str(self.model[i])[0:14]=='GRF_GaussianNB' or str(self.model[i])[0:17]=='GRF_SGDClassifier' or str(self.model[i])[0:31]=='GRF_PassiveAggressiveClassifier' or str(self.model[i])[0:17]=='GRF_MLPClassifier':
                                    
                                    # Accounts for the moment of training beginning
                                    self.running_time_measurements[i].compute_training_time_begin()
                                    self.model[i].partial_fit(spikes, y, self.stream.target_values)
                                    # Accounts the ending of training
                                    self.running_time_measurements[i].compute_training_time_end()
                                
                                else:
                                   
                                    # Accounts for the moment of training beginning
                                    self.running_time_measurements[i].compute_training_time_begin()
                                    self.model[i].partial_fit(X, y, self.stream.target_values)
                                    # Accounts the ending of training
                                    self.running_time_measurements[i].compute_training_time_end()
                                    
                            else:
                                self.running_time_measurements[i].compute_training_time_begin()
                                self.model[i].partial_fit(X, y)
                                self.running_time_measurements[i].compute_training_time_end()

                            # Update total running time
                            self.running_time_measurements[i].update_time_measurements(self.batch_size)
                        first_run = False
                    else:
                                                
                        for i in range(self.n_models):
                            
                            if str(self.model[i])[26:33]=='GRF_KNN' or str(self.model[i])[38:55]=='GRF_HoeffdingTree' or str(self.model[i])[47:72]=='GRF_HoeffdingAdaptiveTree' or str(self.model[i])[0:14]=='GRF_GaussianNB' or str(self.model[i])[0:17]=='GRF_SGDClassifier' or str(self.model[i])[0:31]=='GRF_PassiveAggressiveClassifier' or str(self.model[i])[0:17]=='GRF_MLPClassifier':
                                self.running_time_measurements[i].compute_training_time_begin()
                                self.model[i].partial_fit(spikes, y)
                                self.running_time_measurements[i].compute_training_time_end()
                                self.running_time_measurements[i].update_time_measurements(self.batch_size)

                            else:
                                self.running_time_measurements[i].compute_training_time_begin()
                                self.model[i].partial_fit(X, y)
                                self.running_time_measurements[i].compute_training_time_end()
                                self.running_time_measurements[i].update_time_measurements(self.batch_size)

                    if ((self.global_sample_count % self.n_wait) == 0 or
                            (self.global_sample_count >= self.max_samples) or
                            (self.global_sample_count / self.n_wait > update_count + 1)):
                        if prediction is not None:
                            self._update_metrics()
                        update_count += 1

                self._end_time = timer()
            except BaseException as exc:
                print('fallo en model: ',str(self.model[i]))
                print ('X: ',X)
                print ('y: ',y)
                print ('spikes: ',spikes)
                print(exc)
                if exc is KeyboardInterrupt:
                    self._update_metrics()
                break

#        self.evaluation_summary()

        if self.restart_stream:
            self.stream.restart()

        return self.model

    def partial_fit(self, X, y, classes=None, weight=None):
        """ Partially fit all the learners on the given data.

        Parameters
        ----------
        X: Numpy.ndarray of shape (n_samples, n_features)
            The data upon which the algorithm will create its model.

        y: Array-like
            An array-like containing the classification targets for all samples in X.

        classes: list
            Stores all the classes that may be encountered during the classification task.

        weight: Array-like
            Instance weight. If not provided, uniform weights are assumed.

        Returns
        -------
        EvaluatePrequential
            self

        """
        if self.model is not None:
            for i in range(self.n_models):
                self.model[i].partial_fit(X, y, classes, weight)
            return self
        else:
            return self

    def predict(self, X):
        """ Predicts the labels of the X samples, by calling the predict
        function of all the learners.

        Parameters
        ----------
        X: Numpy.ndarray of shape (n_samples, n_features)
            All the samples we want to predict the label for.

        Returns
        -------
        list
            A list containing the predicted labels for all instances in X in
            all learners.

        """
        predictions = None
        if self.model is not None:
            predictions = []
            for i in range(self.n_models):
                predictions.append(self.model[i].predict(X))

        return predictions

    def set_params(self, parameter_dict):
        """ This function allows the users to change some of the evaluator's parameters,
        by passing a dictionary where keys are the parameters names, and values are
        the new parameters' values.

        Parameters
        ----------
        parameter_dict: Dictionary
            A dictionary where the keys are the names of attributes the user
            wants to change, and the values are the new values of those attributes.

        """
        for name, value in parameter_dict.items():
            if name == 'n_wait':
                self.n_wait = value
            elif name == 'max_samples':
                self.max_samples = value
            elif name == 'pretrain_size':
                self.pretrain_size = value
            elif name == 'batch_size':
                self.batch_size = value
            elif name == 'max_time':
                self.max_time = value
            elif name == 'output_file':
                self.output_file = value
            elif name == 'show_plot':
                self.show_plot = value

    def get_info(self):
        filename = "None"
        if self.output_file is not None:
            _, filename = os.path.split(self.output_file)
        return 'Prequential Evaluator: n_wait: ' + str(self.n_wait) + \
               ' - max_samples: ' + str(self.max_samples) + \
               ' - max_time: ' + str(self.max_time) + \
               ' - output_file: ' + filename + \
               ' - batch_size: ' + str(self.batch_size) + \
               ' - pretrain_size: ' + str(self.pretrain_size) + \
               ' - task_type: ' + self._task_type + \
               ' - show_plot: ' + ('True' if self.show_plot else 'False') + \
               ' - metrics: ' + (str(self.metrics) if self.metrics is not None else 'None')

    def get_response_times(self,dimension,gamma,n_gaussianRF,min_max_tr_data,X,time_coding):
        
        #Gaussian Receptive Fields: Creating the CENTER and the WIDTH for the features
        width=(min_max_tr_data[dimension][1]-min_max_tr_data[dimension][0])/(gamma*(n_gaussianRF-2))    
        temp=(min_max_tr_data[dimension][1]-min_max_tr_data[dimension][0])/(n_gaussianRF-2)
    
        firing_times=[]        
        for i in range(1,(n_gaussianRF+1)):
            center=min_max_tr_data[dimension][0] + (((2*i-3)/2.0) * temp)
            
            if width==0:
                firing_time=0.0
            else:
                firing_time=math.exp((-(X[dimension]-center)**2) / (2*(width**2)))#Funcion gaussiana para calcular los firing times
    
            firing_times.append(firing_time)
            
        return firing_times 
    
    def mcnemar_p(self,b, c):
        '''
        Where Yes/No is the count of test instances that Classifier1 got correct and Classifier2 got incorrect, 
        and No/Yes is the count of test instances that Classifier1 got incorrect and Classifier2 got correct.
        
        The default assumption, or null hypothesis, of the test is that the two cases disagree to the same amount. 
        If the null hypothesis is rejected, it suggests that there is evidence to suggest that the cases disagree in 
        different ways, that the disagreements are skewed.
        
        Given the selection of a significance level, the p-value calculated by the test can be interpreted as follows:
        - p > alpha: fail to reject H0, no difference in the disagreement (e.g. treatment had no effect).
        - p <= alpha: reject H0, significant difference in the disagreement (e.g. treatment had an effect).
        '''
    #  """Computes McNemar's test.
    #  Args:
    #    b: the number of "wins" for the first condition.
    #    c: the number of "wins" for the second condition.
    #  Returns:
    #    A p-value for McNemar's test.
        
        n = b + c
        x = min(b, c)
        dist = scipy.stats.binom(n, .5)
        return 2. * dist.cdf(x)
        
    ## Example of calculating the mcnemar test
    #from statsmodels.stats.contingency_tables import mcnemar
    ## define contingency table
    #table = [[4, 2],
    #		 [1, 3]]
    ## calculate mcnemar test
    #result = mcnemar(table, exact=True)
    ## summarize the finding
    #print('statistic=%.3f, p-value=%.3f' % (result.statistic, result.pvalue))
    ## interpret the p-value
    #alpha = 0.05
    #if result.pvalue > alpha:
    #	print('Same proportions of errors (fail to reject H0)')
    #else:
    #	print('Different proportions of errors (reject H0)')   
