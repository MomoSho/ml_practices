def plot_param_val_surface(crossval, metric, param_lists, sizeidx=0):
    ''' PROVIDED
    Plotting function for after grid_cross_validation(), 
    displaying the mean (summary) train and val set performances 
    for each alpha and l1_ratio, for the ElasticNet
    
    REQUIRES: from mpl_toolkits.mplot3d import Axes3D

    PARAMS:
        crossval: cross validation object
        metric: summary metric to plot. '_mean' or '_std' must be 
                append to the end of the base metric name. These 
                base metric names are the keys in the dict returned
                by eval_func
        param_lists: dictionary of the list of alpha and  l1_ratios
        sizeidx: train size index

    RETURNS: the figure and axes handles
    '''
    sizes = crossval.trainsizes
    results = crossval.results
    best_param_inds = crossval.best_param_inds

    alphas = list(param_lists['alpha'])
    l1_ratios = param_lists['l1_ratio']

    nalphas = len(alphas)
    nl1_ratios = len(l1_ratios)

    nsizes = len(sizes)
    nmetrics = len(metrics)

    Z_train = np.empty((nl1_ratios, nalphas))
    Z_val = np.empty((nl1_ratios, nalphas))

    for param_res in results:
        params = param_res['params']
        summary = param_res['summary']

        alpha_idx = alphas.index(params['alpha'])
        l1_idx = l1_ratios.index(params['l1_ratio'])

        # Compute the mean for multiple outputs
        res_train = np.mean(summary['train'][metric][sizeidx, :])
        Z_train[l1_idx, alpha_idx] = res_train

        res_val = np.mean(summary['val'][metric][sizeidx, :])
        Z_val[l1_idx, alpha_idx] = res_val
    
    # Initialize figure plots
    fig = plt.figure(figsize=(12,5))
    X, Y = np.meshgrid(alphas, l1_ratios)
    for i, (Z, set_name) in enumerate(zip((Z_val, Z_train), ('Validation', 'Training'))):
        # Plot the surface
        ax = fig.add_subplot(1, 2, i+1, projection='3d')
        #ax = Axes3D(axs[i]) #fig.gca(projection='3d') #Axes3D(fig)
        #fig.subplots_adjust(hspace=.05)
        surf = ax.plot_surface(X, Y, Z, cmap=cm.coolwarm, linewidth=0, antialiased=False)
        fig.colorbar(surf, shrink=0.5, aspect=10, label=metric)
        title = "%s Performance, Train Size %d Folds" % (set_name, sizes[sizeidx])
        ax.set(title=title)
        ax.set(xlabel=r"$\alpha$", ylabel='l1_ratio', zlabel=metric)
    return fig
