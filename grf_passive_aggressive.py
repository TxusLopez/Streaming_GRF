#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb  4 10:40:33 2019

@author: txuslopez
"""

# Authors: Rob Zinkov, Mathieu Blondel
# License: BSD 3 clause

from .stochastic_gradient import BaseSGDClassifier
from .stochastic_gradient import BaseSGDRegressor
from .stochastic_gradient import DEFAULT_EPSILON


class GRF_PassiveAggressiveClassifier(BaseSGDClassifier):
    """Passive Aggressive Classifier

    Read more in the :ref:`User Guide <passive_aggressive>`.

    Parameters
    ----------

    C : float
        Maximum step size (regularization). Defaults to 1.0.

    fit_intercept : bool, default=False
        Whether the intercept should be estimated or not. If False, the
        data is assumed to be already centered.

    max_iter : int, optional
        The maximum number of passes over the training data (aka epochs).
        It only impacts the behavior in the ``fit`` method, and not the
        `partial_fit`.
        Defaults to 5. Defaults to 1000 from 0.21, or if tol is not None.

        .. versionadded:: 0.19

    tol : float or None, optional
        The stopping criterion. If it is not None, the iterations will stop
        when (loss > previous_loss - tol). Defaults to None.
        Defaults to 1e-3 from 0.21.

        .. versionadded:: 0.19

    early_stopping : bool, default=False
        Whether to use early stopping to terminate training when validation.
        score is not improving. If set to True, it will automatically set aside
        a fraction of training data as validation and terminate training when
        validation score is not improving by at least tol for
        n_iter_no_change consecutive epochs.

        .. versionadded:: 0.20

    validation_fraction : float, default=0.1
        The proportion of training data to set aside as validation set for
        early stopping. Must be between 0 and 1.
        Only used if early_stopping is True.

        .. versionadded:: 0.20

    n_iter_no_change : int, default=5
        Number of iterations with no improvement to wait before early stopping.

        .. versionadded:: 0.20

    shuffle : bool, default=True
        Whether or not the training data should be shuffled after each epoch.

    verbose : integer, optional
        The verbosity level

    loss : string, optional
        The loss function to be used:
        hinge: equivalent to PA-I in the reference paper.
        squared_hinge: equivalent to PA-II in the reference paper.

    n_jobs : int or None, optional (default=None)
        The number of CPUs to use to do the OVA (One Versus All, for
        multi-class problems) computation.
        ``None`` means 1 unless in a :obj:`joblib.parallel_backend` context.
        ``-1`` means using all processors. See :term:`Glossary <n_jobs>`
        for more details.

    random_state : int, RandomState instance or None, optional, default=None
        The seed of the pseudo random number generator to use when shuffling
        the data.  If int, random_state is the seed used by the random number
        generator; If RandomState instance, random_state is the random number
        generator; If None, the random number generator is the RandomState
        instance used by `np.random`.

    warm_start : bool, optional
        When set to True, reuse the solution of the previous call to fit as
        initialization, otherwise, just erase the previous solution.
        See :term:`the Glossary <warm_start>`.

        Repeatedly calling fit or partial_fit when warm_start is True can
        result in a different solution than when calling fit a single time
        because of the way the data is shuffled.

    class_weight : dict, {class_label: weight} or "balanced" or None, optional
        Preset for the class_weight fit parameter.

        Weights associated with classes. If not given, all classes
        are supposed to have weight one.

        The "balanced" mode uses the values of y to automatically adjust
        weights inversely proportional to class frequencies in the input data
        as ``n_samples / (n_classes * np.bincount(y))``

        .. versionadded:: 0.17
           parameter *class_weight* to automatically weight samples.

    average : bool or int, optional
        When set to True, computes the averaged SGD weights and stores the
        result in the ``coef_`` attribute. If set to an int greater than 1,
        averaging will begin once the total number of samples seen reaches
        average. So average=10 will begin averaging after seeing 10 samples.

        .. versionadded:: 0.19
           parameter *average* to use weights averaging in SGD

    n_iter : int, optional
        The number of passes over the training data (aka epochs).
        Defaults to None. Deprecated, will be removed in 0.21.

        .. versionchanged:: 0.19
            Deprecated

    Attributes
    ----------
    coef_ : array, shape = [1, n_features] if n_classes == 2 else [n_classes,\
            n_features]
        Weights assigned to the features.

    intercept_ : array, shape = [1] if n_classes == 2 else [n_classes]
        Constants in decision function.

    n_iter_ : int
        The actual number of iterations to reach the stopping criterion.
        For multiclass fits, it is the maximum over every binary fit.

    Examples
    --------
    >>> from sklearn.linear_model import PassiveAggressiveClassifier
    >>> from sklearn.datasets import make_classification

    >>> X, y = make_classification(n_features=4, random_state=0)
    >>> clf = PassiveAggressiveClassifier(max_iter=1000, random_state=0,
    ... tol=1e-3)
    >>> clf.fit(X, y)
    PassiveAggressiveClassifier(C=1.0, average=False, class_weight=None,
                  early_stopping=False, fit_intercept=True, loss='hinge',
                  max_iter=1000, n_iter=None, n_iter_no_change=5, n_jobs=None,
                  random_state=0, shuffle=True, tol=0.001,
                  validation_fraction=0.1, verbose=0, warm_start=False)
    >>> print(clf.coef_)
    [[-0.6543424   1.54603022  1.35361642  0.22199435]]
    >>> print(clf.intercept_)
    [0.63310933]
    >>> print(clf.predict([[0, 0, 0, 0]]))
    [1]

    See also
    --------

    SGDClassifier
    Perceptron

    References
    ----------
    Online Passive-Aggressive Algorithms
    <http://jmlr.csail.mit.edu/papers/volume7/crammer06a/crammer06a.pdf>
    K. Crammer, O. Dekel, J. Keshat, S. Shalev-Shwartz, Y. Singer - JMLR (2006)

    """
    def __init__(self, C=1.0, fit_intercept=True, max_iter=None, tol=None,
                 early_stopping=False, validation_fraction=0.1,
                 n_iter_no_change=5, shuffle=True, verbose=0, loss="hinge",
                 n_jobs=None, random_state=None, warm_start=False,
                 class_weight=None, average=False, n_iter=None):
        super(GRF_PassiveAggressiveClassifier, self).__init__(
            penalty=None,
            fit_intercept=fit_intercept,
            max_iter=max_iter,
            tol=tol,
            early_stopping=early_stopping,
            validation_fraction=validation_fraction,
            n_iter_no_change=n_iter_no_change,
            shuffle=shuffle,
            verbose=verbose,
            random_state=random_state,
            eta0=1.0,
            warm_start=warm_start,
            class_weight=class_weight,
            average=average,
            n_jobs=n_jobs
#            n_iter=n_iter
            )

        self.C = C
        self.loss = loss
        
        self.min_max_data = []
        self.gamma = 0.0
        self.n_gaussianRF = 0
        self.time_coding = 1.0        
        
    def partial_fit(self, X, y, classes=None):
        """Fit linear model with Passive Aggressive algorithm.

        Parameters
        ----------
        X : {array-like, sparse matrix}, shape = [n_samples, n_features]
            Subset of the training data

        y : numpy array of shape [n_samples]
            Subset of the target values

        classes : array, shape = [n_classes]
            Classes across all calls to partial_fit.
            Can be obtained by via `np.unique(y_all)`, where y_all is the
            target vector of the entire dataset.
            This argument is required for the first call to partial_fit
            and can be omitted in the subsequent calls.
            Note that y doesn't need to contain all labels in `classes`.

        Returns
        -------
        self : returns an instance of self.
        """
        self._validate_params(for_partial_fit=True)
        if self.class_weight == 'balanced':
            raise ValueError("class_weight 'balanced' is not supported for "
                             "partial_fit. For 'balanced' weights, use "
                             "`sklearn.utils.compute_class_weight` with "
                             "`class_weight='balanced'`. In place of y you "
                             "can use a large enough subset of the full "
                             "training set target to properly estimate the "
                             "class frequency distributions. Pass the "
                             "resulting weights as the class_weight "
                             "parameter.")
        lr = "pa1" if self.loss == "hinge" else "pa2"
        return self._partial_fit(X, y, alpha=1.0, C=self.C,
                                 loss="hinge", learning_rate=lr, max_iter=1,
                                 classes=classes, sample_weight=None,
                                 coef_init=None, intercept_init=None)

    def fit(self, X, y, coef_init=None, intercept_init=None):
        """Fit linear model with Passive Aggressive algorithm.

        Parameters
        ----------
        X : {array-like, sparse matrix}, shape = [n_samples, n_features]
            Training data

        y : numpy array of shape [n_samples]
            Target values

        coef_init : array, shape = [n_classes,n_features]
            The initial coefficients to warm-start the optimization.

        intercept_init : array, shape = [n_classes]
            The initial intercept to warm-start the optimization.

        Returns
        -------
        self : returns an instance of self.
        """
        self._validate_params()
        lr = "pa1" if self.loss == "hinge" else "pa2"
        return self._fit(X, y, alpha=1.0, C=self.C,
                         loss="hinge", learning_rate=lr,
                         coef_init=coef_init, intercept_init=intercept_init)


    def get_info(self):
        return 'GRF_PassiveAggressiveClassifier'