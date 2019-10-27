    def perform_cross_validation(self, all_Xfolds, all_yfolds, 
                                 trainsize, verbose=0):
        '''
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
