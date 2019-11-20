"""
(NEWER VERSION FROM HW11)

K-Fold Holistic Cross Validation for comparing hyper-parameter sets
across multiple training set sizes in terms of folds.
"""
import pandas as pd
import numpy as np
import scipy.stats as stats
import os, re, fnmatch
import pathlib, itertools
import time as timelib
import seaborn
import matplotlib.pyplot as plt
import matplotlib.patheffects as peffects

from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D
from sklearn.pipeline import Pipeline
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.model_selection import cross_val_score, cross_val_predict
from sklearn.metrics import explained_variance_score, confusion_matrix
from sklearn.metrics import mean_squared_error, roc_curve, auc
from sklearn.linear_model import LinearRegression, SGDRegressor
from sklearn.externals import joblib

##############################################################################

def generate_paramsets(param_lists):
    '''
    Construct the Cartesian product of the parameters
    PARAMS:
        params_lists: dict of lists of values to try for each parameter.
                      keys of the dict are the names of the parameters
                      values are lists of values to try for the 
                      corresponding parameter
    RETURNS: a list of dicts that make up the Cartesian product of the 
    parameters
    '''
    keys, values = zip(*param_lists.items())
    # Determines cartesian product of parameter values
    combos = itertools.product(*values)
    # Constructs list of dictionaries
    combos_dicts = [dict(zip(keys, vals)) for vals in combos]
    return list(combos_dicts)


class KFoldHolisticCrossValidation():
    def __init__(self, model, paramsets, eval_func, opt_metric, 
                 maximize_opt_metric=False, trainsizes=[1], rotation_skip=1):
        ''' 
        Object for managing and performing cross validation for a given 
        model for a list of parameter sets and train set sizes. Note, 
        train set size is in terms of number of folds (not samples)
        
        General Procedure:
        + iter over hyper-parameter sets
          1. set hyper-parameters of the model
          2. iter over train set sizes
             a. iter over splits/rotations
                  i. train the model
                 ii. evaluate the model on train, val, and test sets
                iii. record the results
             b. record the results by size
          3. record the results by hyper-parameter set

        PARAMS:
            model: base ML model
            
            paramsets: list of dicts of parameter sets to give to the model
            
            eval_func: handle to function used to evaluate/score the model
                       The eval_func definition must have the following  
                       arguments: model, X, ytrue, ypreds; and return a dict 
                       of numpy arrays with shape 1-by-n, where n is the
                       number of outputs if using multiple regression.
                       template function header: 
                           def eval_func(model, X, y, preds)
                       template output: 
                           {'metrics1':1_by_n_array, ...}
                       
            opt_metric: the optimized metric. one of the metric key names 
                        returned from eval_func to use to pick the best 
                        parameter sets
                        
            maximize_opt_metric: True if opt_metric is maximized; 
                                 False if minimized
            
            trainsizes: list of training set sizes (in number of folds) to try
            
            rotation_skip: build model and evaluate every ith rotation (1=all 
                           possible rotations; 2=every other rotation, etc.)
        ''' 
        self.model = model
        self.paramsets = paramsets
        self.trainsizes = trainsizes
        self.eval_func = eval_func
        self.opt_metric = opt_metric + '_mean'
        self.maximize_opt_metric = maximize_opt_metric
        self.rotation_skip = rotation_skip
        
        # Results attributes
        # Full recording of all results for all paramsets, sizes, rotations,
        # and metrics. This is a list of dictionaries for each paramset
        self.results = None
        # Validation summary report of all means and standard deviations for 
        # all metrics, for all paramsets, and sizes. This is a 3D s-by-r-by-p 
        # numpy array. Where s is the number of sizes, r the number of summary 
        # metrics +2, and p is the number of paramsets
        self.report_by_size = None
        # List of the indices of the best paramset for each size
        self.best_param_inds = None

    def perform_cross_validation(self, all_Xfolds, all_yfolds, 
                                 trainsize, verbose=0):
        ''' TODO: FILL IN WITH YOUR SOLUTION FROM HW6
        Perform cross validation for a singular train set size and single 
        hyper-parameter set, by evaluating the model's performance over 
        multiple data set rotations all of the same size.

        NOTE: This function assumes the hyper-parameters have already been 
              set in the model
            
        PARAMS:
            all_Xfolds: list containing all of the input data folds
            all_yfolds: list containing all of the output data folds
            trainsize: number of folds to use for training
            verbose: flag to display simple debugging information
            
        RETURNS: train, val, and test set results for all rotations of the 
                 data sets and the summary (i.e. the averages over all the 
                 rotations) of the results. 
                 results is a dictionary of dictionaries of r-by-n numpy 
                 arrays. Where r is the number of rotations, and n is the 
                 number of outputs from the model.
                 summary is a dictionary of dictionaries of 1-by-n numpy 
                 arrays. 

                 General form:
                     results.keys() = ['train', 'val', 'test']

                     results['train'].keys() = ['metric1', 'metric2', ...]
                     
                     results['train']['metric1'] = numpy_array
                     
                     results = 
                     {
                        'train':
                                 {
                                     'mse'      : r_by_n_numpy_array,
                                     'rmse_rads': r_by_n_numpy_array, 
                                     'rmse_degs': r_by_n_numpy_array,
                                     ...
                                 },
                        'val'  : {...},
                        'test' : {...}
                     }
                     
                     summary = 
                     {
                        'train':
                                 {
                                     'mse_mean'      : 1_by_n_numpy_array,
                                     'mse_std'       : 1_by_n_numpy_array,
                                     'rmse_rads_mean': 1_by_n_numpy_array, 
                                     'rmse_rads_std' : 1_by_n_numpy_array,
                                     ...
                                 },
                        'val'  : {...},
                        'test' : {...}
                     }

                    For example, you can access the MSE results for the 
                    validation set like so:
                        results['val']['mse'] 
                    For example, you can access the summary (i.e. the average 
                    results over all the rotations) for the test set for the
                    rMSE in degrees like so:
                        summary['test']['rmse_degs_mean']                
        '''
        
        # Verify a valid train set size was provided
        nfolds = len(all_Xfolds)
        if trainsize < 1 or trainsize > nfolds - 2: 
            err_msg = "ERROR: KFoldHolisticCrossValidation.perform_cross_validation() - "
            err_msg += "trainsize (%d) must be between 1 and nfolds (%d) - 2" % (trainsize, nfolds)
            raise ValueError(err_msg)
            
        # Verify rotation skip
        if self.rotation_skip < 1: 
            err_msg = "ERROR: KFoldHolisticCrossValidation.__init__() - "
            err_msg += "rotation_skip (%d) can't be less than 1" % self.rotation_skip
            raise ValueError(err_msg)
        
        # Set up results recording for each rotation
        results = {'train': None, 'val': None, 'test': None}
        summary = {'train': {}, 'val': {}, 'test': {}}
        
        model = self.model
        evaluate = self.eval_func
        
        # Rotate through different train, val, and test sets
        for rotation in range(0, nfolds, self.rotation_skip):
            # Determine fold indices for train, val, and test set. 
            # The val and tests are each only 1 fold
            trainfolds = (np.arange(trainsize) + rotation) % nfolds
            valfold = (nfolds - 2 + rotation) % nfolds
            testfold = (valfold + 1) % nfolds
        
            # Construct train set by concatenating the individual  
            # training folds
            X = np.concatenate(np.take(all_Xfolds, trainfolds))
            y = np.concatenate(np.take(all_yfolds, trainfolds))

            # Construct validation set
            Xval = all_Xfolds[valfold]
            yval = all_yfolds[valfold]
            
            # Construct test set
            Xtest = all_Xfolds[testfold]
            ytest = all_yfolds[testfold]
            
            # DEBUGGING
            if verbose:
                print("TRAIN", X.shape, y.shape, trainfolds)
                print("VAL", Xval.shape, yval.shape, valfold)
                print("TEST", Xtest.shape, ytest.shape, testfold)
            
            # Train model using the training set
            model.fit(X, y) # make sure warm_start is False
            
            # Predict with the model for train, val, and test sets
            preds = model.predict(X)
            preds_val = model.predict(Xval)
            preds_test = model.predict(Xtest)
            
            # Evaluate the model for each set
            res_train = evaluate(model, X, y, preds)
            res_val = evaluate(model, Xval, yval, preds_val)
            res_test = evaluate(model, Xtest, ytest, preds_test)

            # Record the train, val, and test set results. These are dicts 
            # of result metrics, returned by the evaluate function
            # For the first rotation, store the results from evaluating
            # with the train, val, and tests by setting the values of   
            # the appropriate items within the results dict
            if results['train'] is None: 
                results['train'] = res_train
                results['val'] = res_val
                results['test'] = res_test
            else:
                # Append the results for each rotation
                for metric in res_train.keys():
                    results['train'][metric] = np.append(results['train'][metric], 
                                                         res_train[metric], axis=0)
                    results['val'][metric] = np.append(results['val'][metric], 
                                                       res_val[metric], axis=0)
                    results['test'][metric] = np.append(results['test'][metric], 
                                                        res_test[metric], axis=0)

        # Compute/record mean and standard deviation for the size for each metric
        for metric in results['train'].keys():
            for stat_set in ['train', 'val', 'test']:
                summary[stat_set][metric+'_mean'] = np.mean(results[stat_set][metric], 
                                                            axis=0).reshape(1, -1)
                summary[stat_set][metric+'_std'] = np.std(results[stat_set][metric], 
                                                          axis=0).reshape(1, -1)

        return results, summary

    def grid_cross_validation(self, all_Xfolds, all_yfolds, verbose=0):
        '''
        (MAIN PROCEDURE) Perform cross validation for multiple sets of 
        parameters and train set sizes. Calls self.perform_cross_validation(). 
        This is the procedure that executes cross validation for all parameter 
        sets and all sizes.
        
        General Procedure:
        + iter over hyper-parameter sets
          1. set hyper-parameters of the model
          2. iter over train set sizes
             a. iter over splits/rotations
                  i. train the model
                 ii. evaluate the model on train, val, and test sets
                iii. record the results
             b. record the results by size
          3. record the results by hyper-parameter set
        
        PARAMS:
            all_Xfolds: all the input data folds (list of folds, as it was 
                        loaded from the files)
            all_yfolds: all the output data folds (list of folds)
            verbose: flag to print out simple debugging information
            
        RETURNS: best parameter set for each train set size as a list of 
                 parameter indices. Additionally, returns self.report_by_size,
                 the 3D array of validation means (overall rotations) for all 
                 paramsets, for each metric, for all sizes. The structure of 
                 the returned object is a dictionary of the following form: 
                 { 
                   'report_by_size' : self.report_by_size, 
                   'best_param_inds': self.best_param_inds
                 }
        ''' 
        sizes = self.trainsizes
        paramsets = self.paramsets
        nparamsets = len(paramsets)
        print("nparamsets", nparamsets)
        
        # Set up all results
        all_results = []
        
        # Iterate over parameter sets
        for params in paramsets:
            # Set up paramset results 
            param_res = {'train':{}, 'val':{}, 'test':{}}
            param_smry = None
            
            # Set model parameters
            print("Current paramset\n", params)
            self.model.set_params(**params)

            # Iterate over the different train set sizes
            for size in sizes:
                # Cross-validation for current model and train size
                res, smry = self.perform_cross_validation(all_Xfolds, 
                                                          all_yfolds, 
                                                          size, verbose)

                # Save the results
                #param_res.append(res) 
                if param_res['train'] == {}: 
                    param_res = res
                else:
                    # Append the results for each size (r-by-n-by-s)
                    # r: # of rotations; n: # of outputs; s: # of sizes
                    for metric in res['train'].keys():
                        for set_name in ['train', 'val', 'test']:
                            param_res_set = param_res[set_name][metric]
                            if param_res_set.ndim < 3:
                                param_res_set = np.expand_dims(param_res_set, axis=2)
                            stat = np.expand_dims(res[set_name][metric], axis=2)
                            param_res[set_name][metric] = np.append(param_res_set, 
                                                                    stat, axis=2)
                
                # Save the mean and standard deviation statistics (summary)
                if param_smry is None: param_smry = smry
                else:
                    # For each metric measured, append the summary results
                    for metric in smry['train'].keys():
                        for stat_set in ['train', 'val', 'test']:
                            param_smry_set = param_smry[stat_set][metric]
                            stat = smry[stat_set][metric]
                            param_smry[stat_set][metric] = np.append(param_smry_set,
                                                                     stat, axis=0)
            
            # Append the results and summary for the parameter set
            all_results.append({'params':params, 'results':param_res, 
                                'summary':param_smry})
        
        # Generate reports and determine best params for each size 
        self.results = all_results
        self.report_by_size = self.get_reports()
        self.best_param_inds = self.get_best_params(self.opt_metric, 
                                                    self.maximize_opt_metric)
        return {'report_by_size':self.report_by_size, 
                'best_param_inds':self.best_param_inds}

    def get_reports(self):
        ''' 
        Get the mean validation summary of all the parameters for each size
        for all metrics. This is used to determine the best parameter set  
        for each size
        
        RETURNS: the report_by_size as a 3D s-by-r-by-p array. Where s is 
                 the number of train sizes tried, r is the number of summary  
                 metrics evaluated+2, and p is the number of parameter sets.
        '''
        results = self.results
        sizes = np.reshape(self.trainsizes, (1, -1))
        
        nsizes = sizes.shape[1]
        nparams = len(results)
        
        # Set up the reports objects
        metrics = list(results[0]['summary']['val'].keys())
        colnames = ['params', 'size'] + metrics 
        report_by_size = np.empty((nsizes, len(colnames), nparams), dtype=object)

        # Determine mean val for each paramset for each size for all metrics
        for p, paramset_result in enumerate(results):
            params = paramset_result['params']
            res_val = paramset_result['summary']['val']

            # Compute mean val result for each train size for each metric
            means_by_size = [np.mean(res_val[metric], axis=1) 
                             for metric in metrics]
            # Include the train set sizes into the report
            means_by_size = np.append(sizes, means_by_size, axis=0)
            # Include the parameter sets into the report
            param_strgs = np.reshape([str(params)]*nsizes, (1, -1))
            means_by_size = np.append(param_strgs, means_by_size, axis=0).T
            # Append the parameter set means into the report 
            report_by_size[:,:,p] = means_by_size
        return report_by_size

    def get_best_params(self, opt_metric, maximize_opt_metric):
        ''' 
        Determines the best parameter set for each train size,  
        based on a specific metric.
        
        PARAMS:
            opt_metric: optimized metric. one of the metrics returned 
                        from eval_func, with '_mean' appended for the
                        summary stat. This is the mean metric used to  
                        determine the best parameter set for each size
                        
            maximize_opt_metric: True if the max of opt_metric should be
                                 used to determine the best parameters.
                                 False if the min should be used.
        RETURNS: list of best parameter set indicies for each size 
        '''
        results = self.results
        report_by_size = self.report_by_size 
                
        metrics = list(results[0]['summary']['val'].keys())
        
        # Determine best params for each size, for the optimized metric
        best_param_inds = None
        metric_idx = metrics.index(opt_metric)
        
        # Report info for all paramsets for the optimized metric
        report_opt_metric = report_by_size[:, metric_idx+2, :]
        
        if maximize_opt_metric:
            # Add two for the additional cols for params and size
            best_param_inds = np.argmax(report_opt_metric, axis=1)
        else: 
            best_param_inds = np.argmin(report_opt_metric, axis=1)
        # Return list of best params indices for each size
        return best_param_inds
    
    def get_best_params_strings(self):
        ''' 
        Generates a list of strings of the best params for each size
        RETURNS: list of strings of the best params for each size
        '''
        best_param_inds = self.best_param_inds
        results = self.results
        return [str(results[p]['params']) for p in best_param_inds]

    def get_report_best_params_for_size(self, size):
        ''' 
        Get the mean validation summary for the best parameter set 
        for a specific size for all metrics.
        PARAMS:
            size: index of desired train set size for the best  
                  paramset to come from. Size here is the index in 
                  the trainsizes list, NOT the actual number of folds.
        RETURNS: the best parameter report for the size as an s-by-m  
                 dataframe. Where each row is for a different size, and 
                 each column is for a different validation summary metric.
        '''
        best_param_inds = self.best_param_inds
        report_by_size = self.report_by_size 

        # Obtain the index of the best parameter set for the size
        bp_index = best_param_inds[size]

        # Obtain the list of metrics
        metrics = list(self.results[0]['summary']['val'].keys())
        colnames = ['params', 'size'] + metrics
        
        # Create DataFame with all summary stats for the parameter set
        report_best_params_for_size = pd.DataFrame(report_by_size[:,:,bp_index],
                                                   columns=colnames)
        return report_best_params_for_size
    
    def get_report_best_params_all_sizes(self):
        """
        Construct a summary dataframe of the best parameter sets for each size 
        in number of folds, showing the parameter set index and the actual 
        hyper-parameter values
        
        RETURN: a DataFrame
        """
        print("Best Parameter Sets For Each Train Set Size")

        best_params_info = pd.DataFrame((self.trainsizes, self.best_param_inds, 
                                  self.get_best_params_strings()),
                                  index=['train_size','param_index','paramset'])
        return best_params_info.T
    
    def get_result(self, metric, paramidx=0, sizeidx=0, set_type='val', output_names=None):
        """
        Obtain the results from each rotation for a particular parameter set, set type 
        (i.e. 'train', 'val', or 'test'), metric, and size, for all outputs.
        PARAMS:
            metric: a metric returned by eval_func
            paramidx: index of desired parameter set
            sizeidx: index of the desired train size in the trainsizes list (can 
                     be a list)
            set_type: either 'train', 'val', or 'test'
            output_names: a list of the names of each output
        RETURNS: a r-by-n pandas dataframe of results for the specified metric. 
            r is the number of rotations and n is the number of outputs
        """
        results = pd.DataFrame(self.results[paramidx]['results'][set_type][metric][:,:,sizeidx])
        nrotations = results.shape[0]
        row_names = ['rotation_%02d' % i for i in range(nrotations)]
        if output_names is None: 
            noutputs = results.shape[1]
            output_names = ['output_%02d' % i for i in range(noutputs)]
        results.columns = output_names
        results.index = row_names
        return results
    
    def get_summary(self, metric, paramidx=0, sizeidx=0, set_type='val', output_names=None):
        """
        Obtain the summary results for each size for a particular parameter set, set type 
        (i.e. 'train', 'val', or 'test'), metric, and size, for all outputs.
        PARAMS:
            metric: a metric returned by eval_func
            paramidx: index of desired parameter set
            sizeidx: index of the desired train size in the trainsizes list (can 
                     be a list)
            set_type: either 'train', 'val', or 'test'
            output_names: a list of the names of each output
        RETURNS: a s-by-n pandas dataframe of results for the specified metric. 
            s is the number of train sizes and n is the number of outputs
        """
        summary = pd.DataFrame(self.results[paramidx]['summary'][set_type][metric])
        nrotations = summary.shape[0]
        row_names = ['%02d folds' % i for i in self.trainsizes]
        if output_names is None: 
            noutputs = results.shape[1]
            output_names = ['output_%02d' % i for i in range(noutputs)]
        summary.columns = output_names
        summary.index = row_names
        return summary

    def plot_cv(self, foldsindices, results, summary, metrics, size):
        ''' 
        Plotting function for after perform_cross_validation(), 
        displaying the train and val set performances for each rotation 
        of the training set. 
        
        PARAMS:
            foldsindices: indices of the train sets tried
            results: results from perform_cross_validation()
            summary: mean and standard deviations of the results
            metrics: list of result metrics to plot. Available metrics 
                     are the keys in the dict returned by eval_func
            size: train set size
            
        RETURNS: the figure and axes handles
        '''
        nmetrics = len(metrics)

        # Initialize figure plots
        fig, axs = plt.subplots(nmetrics, 1, figsize=(12,6))
        fig.subplots_adjust(hspace=.35)
        # When 1 metric is provided, allow the axs to be iterable
        axs = np.array(axs).ravel()

        # Construct each subplot
        for metric, ax in zip(metrics, axs):
            # Compute the mean for multiple outputs
            res_train = np.mean(results['train'][metric], axis=1)
            res_val = np.mean(results['val'][metric], axis=1)
            #res_test = np.mean(results['test'][metric], axis=1)
            # Plot
            ax.plot(foldsindices, res_train, label='train')
            ax.plot(foldsindices, res_val, label='val')
            #ax.plot(foldsindices, res_test, label='test')
            ax.set(ylabel=metric)
        axs[0].legend(loc='upper right')
        axs[0].set(xlabel='Fold Index')
        axs[0].set(title='Performance for Train Set Size ' + str(size))
        return fig, axs

    def plot_param_train_val(self, metrics, paramidx=0, view_test=False):
        ''' 
        Plotting function for after grid_cross_validation(), 
        displaying the mean (summary) train and val set performances 
        for each train set size.
        
        PARAMS:
            metrics: list of summary metrics to plot. '_mean' or '_std'
                     must be append to the end of the base metric name. 
                     These base metric names are the keys in the dict 
                     returned by eval_func
            paramidx: parameter set index
            view_test: flag to view the test set results
            
        RETURNS: the figure and axes handles
        '''
        sizes = self.trainsizes
        results = self.results

        summary = results[paramidx]['summary']
        params = results[paramidx]['params']
        
        nmetrics = len(metrics)

        # Initialize figure plots
        fig, axs = plt.subplots(nmetrics, 1, figsize=(12,6))
        fig.subplots_adjust(hspace=.35)
        # When 1 metric is provided, allow the axs to be iterable
        axs = np.array(axs).ravel()

        # Construct each subplot
        for metric, ax in zip(metrics, axs):
            # Compute the mean for multiple outputs
            res_train = np.mean(summary['train'][metric], axis=1)
            res_val = np.mean(summary['val'][metric], axis=1)
            # Plot
            ax.plot(sizes, res_train, label='train')
            ax.plot(sizes, res_val, label='val')
            if view_test:
                res_test = np.mean(summary['test'][metric], axis=1)
                ax.plot(sizes, res_test, label='test')
            ax.set(ylabel=metric)
        axs[-1].set(xlabel='Train Set Size (# of folds)')
        axs[0].set(title=str(params))
        axs[0].legend(loc='upper right')
        return fig, axs
    
    def plot_allparams_val(self, metrics):
        ''' 
        Plotting function for after grid_cross_validation(), displaying  
        mean (summary) validation set performances for each train size 
        for all parameter sets for the specified metrics.
        
        PARAMS:
            metrics: list of summary metrics to plot. '_mean' or '_std' 
                     must be append to the end of the base metric name. 
                     These base metric names are the keys in the dict 
                     returned by eval_func
                     
        RETURNS: the figure and axes handles
        '''
        sizes = self.trainsizes
        results = self.results
        
        nmetrics = len(metrics)

        # Initialize figure plots
        fig, axs = plt.subplots(nmetrics, 1, figsize=(10,6))
        fig.subplots_adjust(hspace=.35)
        # When 1 metric is provided, allow the axs to be iterable
        axs = np.array(axs).ravel()

        # Construct each subplot
        for metric, ax in zip(metrics, axs):
            for p, param_results in enumerate(results):
                summary = param_results['summary']
                params = param_results['params']
                # Compute the mean for multiple outputs
                res_val = np.mean(summary['val'][metric], axis=1)                
                ax.plot(sizes, res_val, label=str(params))
            ax.set(ylabel=metric)
        axs[-1].set(xlabel='Train Set Size (# of folds)')
        axs[0].set(title='Validation Performance')
        axs[0].legend(bbox_to_anchor=(1.02, 1), loc='upper left',
                      ncol=1, borderaxespad=0., prop={'size': 8})
        return fig, axs

    def plot_best_params_by_size(self):
        ''' 
        Plotting function for after grid_cross_validation(), displaying 
        mean (summary) train and validation set performances for the best 
        parameter set for each train size for the optimized metric.
                     
        RETURNS: the figure and axes handles
        '''
        results = self.results
        metric = self.opt_metric
        best_param_inds = self.best_param_inds
        sizes = np.array(self.trainsizes)

        # Unique set of best params for the legend
        unique_param_sets = np.unique(best_param_inds)
        lgnd_params = [self.paramsets[p] for p in unique_param_sets]

        # Initialize figure
        fig, axs = plt.subplots(2, 1, figsize=(10,6))
        fig.subplots_adjust(hspace=.35)
        # When 1 metric is provided, allow the axs to be iterable
        axs = np.array(axs).ravel()
        set_names = ['train', 'val']

        # Construct each subplot
        for i, (ax, set_name) in enumerate(zip(axs, set_names)):
            for p in unique_param_sets:
                # Obtain indices of sizes this paramset was best for
                param_size_inds = np.where(best_param_inds == p)[0]
                param_sizes = sizes[param_size_inds]
                # Compute the mean over multiple outputs for each size
                param_summary = results[p]['summary'][set_name]
                metric_scores = np.mean(param_summary[metric][param_size_inds,:], axis=1)
                # Plot the param results for each size it was the best for
                ax.scatter(param_sizes, metric_scores, s=120, marker=(p+2, 1))
                #ax.grid(True)

            set_name += ' Set Performance'
            ax.set(ylabel=metric, title=set_name)

        axs[-1].set(xlabel='Train Set Size (# of folds)')
        axs[0].legend(lgnd_params, bbox_to_anchor=(1.02, 1), loc='upper left',
                      ncol=1, borderaxespad=0., prop={'size': 7})
        return fig, axs