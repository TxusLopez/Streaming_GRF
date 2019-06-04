# Streaming_GRF
Gaussian Receptive Fields for Stream Learning 

--------
CODE
--------
- Required frameworks: scikit-multiflow, scikit-learn
- Dependencies: texttable, collections, statsmodels.stats.contingency_tables, pandas, numpy, math, warnings, pickle, scipy.io, scipy.stats, matplotlib.pyplot, seaborn 

The "benchmark_git.py" script is used to generate the final results. It uses the scikit-learn framework (GaussianNB, PassiveAgressiveClassifier, MLPClassifier, and SGDClassifier techniques). It also uses the scikit-multiflow framework (HoeffdingTree, HoeffdingAdaptiveTree, and KNN techniques, and also the ADWIN drift detector).

The "evaluate_prequential_NN.py" script is used for the streaming evaluation. This file should be placed in the corresponding folder of the scikit-multiflow package: '.../scikit-multiflow-master/src/skmultiflow/evaluation'

The following scripts are modified versions of the original algorithms for the scikit-multiflow framework in order to consider the GRF attributes. They should be placed in the corresponding folder. They are: 
GRF_HoeffdingTree.py--> '.../scikit-multiflow-master/src/skmultiflow/trees/'
GRF_HoeffdingAdaptiveTree.py--> '.../scikit-multiflow-master/src/skmultiflow/trees/'
GRF_KNN.py--> '.../scikit-multiflow-master/src/skmultiflow/lazy/'

The following scripts are modified versions of the original algorithms for the scikit-learn framework in order to consider the GRF attributes. They should be placed in the corresponding folder. They are: 
GRF_GaussianNB.py--> '.../lib/python3.6/site-packages/sklearn'
GRF_PassiveAgressiveClassifier.py--> '.../lib/python3.6/site-packages/sklearn/linear_model'
GRF_MLPClassifier.py--> '.../lib/python3.6/site-packages/sklearn/neural_network'
GRF_SGDClassifier.py--> '.../lib/python3.6/site-packages/sklearn/linear_model'

--------
DATA
--------

The following synthetic datasets are provided:
- circle1 (first concept of the circleG dataset)
- circle2 (second concept of the circleG dataset)
- line1 (first concept of the line dataset)
- line2 (second concept of the line dataset)
- sine1 (first concept of the sineV dataset)
- sine2 (second concept of the sineV dataset)
- sineH1 (first concept of the sineH dataset)
- sineH2 (second concept of the sineH dataset)
- SEA

The following real-world datasets are provided:
- weather
- electricity
- moving squares
- Airlines
