#!/usr/bin/env python
# coding: utf-8

# In[6]:


import pandas as pd
import patsy as pt
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn import preprocessing
from sklearn import model_selection
import pymc3 as pm
import re
import pickle
from scipy.stats import norm
from collections import OrderedDict
from glob import glob
from pyemma.coordinates.clustering import KmeansClustering
from pyemma.msm import MaximumLikelihoodMSM
import time

# Directory locations
# This directory
root_dir = '/Users/robertarbon/OneDrive - University of Bristol/Research/optimize_msms/Ala1/'
# This is where the 'random' trials come from. 
input_dir = root_dir+'outputs/'
# this is where all the output goes
output_dir = root_dir+'outputs/bayes_opt_gp_m52/'
# this is where the features are
data_dir = root_dir+'data/features/'


# Instructions: 
# 1. Set the global parameters 
# 2. Load the trial data and then give the hyperparameters sensible names. 
# 3. Set parameters in the score_trial function

# Global parameters

# In[7]:


def gamma(alpha, beta):
    def g(x):
        return pm.Gamma(x, alpha=alpha, beta=beta)
    return g

def hcauchy(beta):
    def g(x):
        return pm.HalfCauchy(x, beta=beta)
    return g


def fit_model_1(y, X, kernel_type='rbf'):
    """
    function to return a pymc3 model
    y : dependent variable
    X : independent variables
    prop_Xu : number of inducing varibles to use
    
    X, y are dataframes. We'll use the column names. 
    """
    with pm.Model() as model:
        # Covert arrays
        X_a = X.values
        y_a = y.values
        X_cols = list(X.columns)
        
        # Globals
        prop_Xu = 0.1 # proportion of observations to use as inducing variables
        l_prior = gamma(1, 0.05)
        eta_prior = hcauchy(2)
        sigma_prior = hcauchy(2)

        # Kernels
        # 3 way interaction
        eta = eta_prior('eta')
        cov = eta**2
        for i in range(X_a.shape[1]):
            var_lab = 'l_'+X_cols[i]
            if kernel_type.lower()=='rbf':
                cov = cov*pm.gp.cov.ExpQuad(X_a.shape[1], ls=l_prior(var_lab), active_dims=[i])
            if kernel_type.lower()=='exponential':
                cov = cov*pm.gp.cov.Exponential(X_a.shape[1], ls=l_prior(var_lab), active_dims=[i])
            if kernel_type.lower()=='m52':
                cov = cov*pm.gp.cov.Matern52(X_a.shape[1], ls=l_prior(var_lab), active_dims=[i])
            if kernel_type.lower()=='m32':
                cov = cov*pm.gp.cov.Matern32(X_a.shape[1], ls=l_prior(var_lab), active_dims=[i])

        # Covariance model
        cov_tot = cov 

        # Model
        gp = pm.gp.MarginalSparse(cov_func=cov_tot, approx="FITC")

        # Noise model
        sigma_n =sigma_prior('sigma_n')

        # Inducing variables
        num_Xu = int(X_a.shape[0]*prop_Xu)
        Xu = pm.gp.util.kmeans_inducing_points(num_Xu, X_a)

        # Marginal likelihood
        y_ = gp.marginal_likelihood('y_', X=X_a, y=y_a,Xu=Xu, noise=sigma_n)
        mp = pm.find_MAP()
        
    return gp, mp, model


# In[8]:


def create_dmatrix(df, formula, target):
    # create data matrix/dataframe
    # X = raw data matrix
    # Xc = (dummy) coded matrix
    if (target is not None) and (target in df.columns): 
        y = df.loc[:, target]
    else:
        y = None
    
    # dummy coding of basis
    Xc = pt.dmatrix(formula, data=df, return_type='dataframe')
    Xc = Xc.rename(columns=lambda x: re.sub('C','',x))
    
    return y, Xc

def scale_dmatrix(X, scaler=None):
    # scales matrices and returns scaler
    if scaler is None: 
        scaler = preprocessing.MinMaxScaler()
        scaler.fit(X.values)
    
    Xs = scaler.transform(X.values)
    Xs = pd.DataFrame(Xs, columns=[x+'_s' for x in X.columns])
    return Xs, scaler

def create_grid(search_space):
    # creates prediction grid from search space.
    Xnew = np.meshgrid(*search_space.values())
    Xnew = np.concatenate([x.reshape(-1, 1) for x in Xnew], axis=1)
    Xnew = pd.DataFrame(Xnew, columns=search_space.keys())
    for k, v in search_space.items():
        Xnew.loc[:, k] = Xnew.loc[:, k].astype(v.dtype)
    return Xnew

def get_predictions(gp, mp, model, Xnew):
    # Get predictions for evalution
    with model:
        # predict latent
        mu, var = gp.predict(Xnew.values, point=mp, diag=True,pred_noise=False)
        sd = np.sqrt(var)
        
    # put in data frame
    pred = pd.DataFrame({'mu': mu, 'sigma': sd})
    pred = pred.join(Xnew)
    return pred

def exp_imp(mu, sigma, mu_max, xsi=0):
    """
    mu: mean of response surface
    sigma: sd of response surface
    xsi: explore/exploit tradeoff parameter
    mu_max: the incumbent
    """
    Z = (mu - mu_max - xsi)/sigma
    # Z is zero for sigma = 0
    zero_idx = np.abs(Z) > 1e8
    Z[zero_idx] = 0
    pdf = norm.pdf(Z)
    cdf = norm.cdf(Z)
    ei = (mu - mu_max - xsi)*cdf + sigma*pdf
    return ei

# def get_incumbent(gp, mp, model, X):
#     pred = get_predictions(gp, mp, model, X)
#     pred.sort_values(by='mu', ascending=False, inplace=True)
#     return pred.iloc[0, :]

# def add_ei(pred, mu_max, xsi):
#     pred['ei'] = exp_imp(pred['mu'], pred['sigma'], mu_max=mu_max, xsi=xsi)
#     return pred
  
def score_trial(suggest, data_dir,n_splits = 20, lag = 9, k = 5, method = 'VAMP2', stride = 10, 
                n_trajs = 750, test_size = 0.5, max_iter = 1000):
#     GLOBAL HYPERPARAMETERS
#     n_splits = 20 # CV splits
#     lag = 9 # msm lag time
#     k = 5 # num eigenvalues in score
#     method = 'VAMP2' # score method
#     stride = 10 # stride for clustering
#     n_trajs = 750 # number of trajectories - for checking whether we've loaded all the data. 
#     test_size = 0.5 # train/test split
#     max_iter = 1000 # for clustering algo
    
    # SUGGESTED HYPERPARAMETERS 
    # the 'suggest' dataframe 
    n = int(suggest.loc[suggest.index[0], 'n'])
    feat = suggest.loc[suggest.index[0], 'basis']
    feat_dir = data_dir + feat

    # load data
    traj_paths = glob(feat_dir+'/*.npy')
    assert len(traj_paths) == n_trajs
    trajs = [np.load(x) for x in traj_paths]

    # cross-validation loop
    cv = model_selection.ShuffleSplit(n_splits=n_splits, test_size=test_size)
    test_scores = []
    train_scores = []
    
    for train_idx, test_idx in cv.split(trajs):
        train = [trajs[i] for i in train_idx]
        test = [trajs[i] for i in test_idx]

        # pipeline
        cluster = KmeansClustering(n_clusters=n, max_iter=max_iter, stride=stride)
        mod = MaximumLikelihoodMSM(lag=lag, score_k=k, score_method=method)

        # train
        z = cluster.fit_transform(train)
        z = [x.flatten() for x in z]
        mod.fit(z)
        score = mod.score(z)
        train_scores.append(score)
        
        # test
        z = cluster.transform(test)
        z = [x.flatten() for x in z]
        score = mod.score(z)
        test_scores.append(score)
        
    return {'n': n, 'basis': feat, 'test_scores': test_scores, 'train_scores': train_scores}

def add_results(results, history):
    # Adds the results of the trial to the history
    
    for col in ['test', 'train']:
        results[col+'_mean'] = np.mean(results[col+'_scores']) #.apply(f, stat=np.mean)
        results[col+'_std'] =  np.std(results[col+'_scores']) #.apply(f, stat=np.std)
        results[col+'_min'] =  np.min(results[col+'_scores']) #.apply(f, stat=np.min)
        results[col+'_max'] =  np.max(results[col+'_scores']) #.apply(f, stat=np.max)
        results[col+'_q25'] =  np.quantile(results[col+'_scores'], q=0.25) #.apply(f, stat=np.quantile, q=0.25)
        results[col+'_q75'] =  np.quantile(results[col+'_scores'], q=0.75) #.apply(f, stat=np.quantile, q=0.75)
        # pop the scores off as we don't need them (we hope). 
        results.pop(col+'_scores')
        
    results['trial_id'] = history['trial_id'].max()+1
    results['method'] = 'bayes'
    
    # add to the trials history
    results = pd.concat([history, pd.DataFrame(results, index=[1])], 
                        axis=0, join='outer', ignore_index=True, 
                        sort=False)
    
    return results

def plot_ei_rs(*args, **kwargs):
    data=kwargs.pop('data')
    color=kwargs.pop('color')
    ylim = kwargs.pop('ylim')

    # plot response surface
    ax = plt.gca()
    ax2 = ax.twinx()
    ax.plot(data['n'], data['mu'], color=color, label=r'$f(\chi, n)$')
    ax.fill_between(data['n'], data['mu']-2*data['sigma'], data['mu']+2*data['sigma'], 
                    color=color, alpha=0.5, label=r'$2*\sigma$') 

    # plot acquisition function
    ax2.plot(data['n'], data['ei'], color='k', label='Expected Improvement')
    ax2.set_ylim(0, ylim)


# ## Loop starts here

# Everything from here is considered a Bayesian optimisation run **starting from random data**. 
# 
# We could do everything here multiple times and it would get a sense of how efficient the whole method is. 

# In[9]:


# These labels should self consistent! 
relabel = {'project_name': 'basis', 
           'cluster__n_clusters': 'n'}

search_space = OrderedDict([
    ('basis', np.array(['psi', 'phi', 'rmsd', 'positions', 'phipsi'])), 
    ('n', np.arange(10, 1001, 10, dtype='float64'))])
predictors = list(search_space.keys())
target = 'test_mean'

# For the response surface. 
formula = '0 + np.log(n) + C(basis)'
kernel = 'm52'

n_basis = search_space['basis'].shape[0]

# Start calculating the surface with this many observations
start_n = 40 # start_n/n_basis  = average per feature

# start optimization with this many observations
start_bayes_n = 50 # start_bayes_n/n_basis  = average per feature

# number of Bayesian optimisation trials
num_trials = 10

# number of independent iterations
n_iters = 10

# number of CV splits
n_splits = 20


# In[ ]:





# create prediction grid and scale. Use this scaler for everything. 
# This shouldn't, but does, have an effect on the predictions and expected improvement. 
# I will investigate this later. 
newX = create_grid(search_space)
_, newXc = create_dmatrix(newX, formula=formula, target=None)
newXs, scaler = scale_dmatrix(newXc, scaler=None)

for j in range(n_iters):
    # record the results in these: 
    incumbents = {'iteration': [], 'n_obs': [], 'mu': [],  'sigma': [], 'basis': [], 'n': []}
    candidates = {'iteration': [], 'n_obs': [], 'ei': [],  'mu': [],  'sigma': [], 'basis': [], 'n': []}
    mods = {'iteration': [], 'n_obs': [], 'mod': []}
    preds = [] # surface
    
    # Load clean set of random data. 
    trials = pd.read_csv(input_dir+'ala1_trials_clean.csv')
    trials = trials.rename(columns=relabel)

    # Subset with all the random data we'll need. 
    trials = trials.groupby('basis', group_keys=False).apply(lambda x: x.sample(n=int(start_bayes_n/n_basis)))
    trials = trials.sample(frac=1)
    
    # Add labels for tracking iterations
    trials['method'] = 'random'
    trials['trial_id'] = np.arange(trials.shape[0])

    # outname
    save_path = output_dir + 'start_obs-{0}_iter-{1}_'.format(start_bayes_n, j)

    for i in range(start_n, start_bayes_n+num_trials+1):
        
        # ESTIMATE RESPONSE SURFACE
        X = trials.loc[trials.trial_id < i, predictors+[target]]
        print(i, X.shape, trials.shape)
        y, Xc = create_dmatrix(X, formula = formula, target = target)
        Xs, _ = scale_dmatrix(Xc, scaler=scaler)
        
        # only use new model if it successfully fits. 
        try:
            a, b, c = fit_model_1(y, Xs, kernel_type=kernel)
            gp = a
            mp = b
            model = c
        except: 
            print('ERROR in FIT', j, i)
            
        # RECORD MODEL
        mods['iteration'].append(j)
        mods['n_obs'].append(X.shape[0])
        mods['mod'].append({'mod': model, 'gp': gp, 'mp': mp})
        
        # GET INCUMBENT
        # # make predictions at input values. 
        # # concat the untransformed variables to make it easier to read. 
        pred_X = get_predictions(gp, mp, model, Xs)
        pred_X = pd.concat([X.reset_index(drop=True), pred_X.reset_index(drop=True)], axis=1)
        pred_X['iteration'] = j
        pred_X['n_obs'] = X.shape[0]
        # # select the incumbent. 
        pred_X = pred_X.sort_values(by='mu', ascending=False, inplace=False)
 
        # RECORD INCUMBENT
        pix = pred_X.index[0]
        mu_max = pred_X.loc[pix, 'mu']
        for k in incumbents.keys():
            incumbents[k].append(pred_X.loc[pix, k])
            
        # PRINT INCUMBENT
        print('ITERATION: ', i)
        print('INCUMBENT:')
        print(pred_X.head(1).T)
        print()
        
        # MAKE TRIAL SUGGESTION
        # get predictions over search space and add back untransformed variables. 
        pred = get_predictions(gp, mp, model, newXs)
        pred = pd.concat([newX.reset_index(drop=True), pred.reset_index(drop=True)], axis=1)
        # calculate expected improvement and find max value
        pred['ei'] = exp_imp(pred['mu'], pred['sigma'], mu_max=mu_max, xsi=0)   
        pred['iteration'] = j
        pred['n_obs'] = X.shape[0]
        pred = pred.sort_values(by='ei', ascending=False, inplace=False)

        # RECORD SURFACE
        preds.append(pred)

        # RECORD SUGGESTION
        pix = pred.index[0]
        mu_max = pred.loc[pix, 'mu']
        for k in candidates.keys():
            candidates[k].append(pred.loc[pix, k])

        print('SUGGESTION:')
        print(pred.head(1).T)
        print()
        print('-'*78)
        
        # SCORE NEW SUGGESTION
        if i >= start_bayes_n:

            # SCORE TRIAL
            results = score_trial(pred, data_dir, n_splits=n_splits)
            trials = add_results(results, history = trials)
            
    # SAVE
    trials.to_csv(save_path+'trials.csv', index=False)
    pd.DataFrame(incumbents).to_csv(save_path+'incumbents.csv', index=False)
    pd.DataFrame(candidates).to_csv(save_path+'candidates.csv', index=False)
    pd.concat(preds).to_csv(save_path+'response_surface.csv', index=False)
    pickle.dump(file=open(save_path+'models.p', 'wb'), obj=mods)


# In[ ]:




