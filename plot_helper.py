import pandas as pd
from matplotlib import pyplot as plt
import numpy as np

def plot_results(clf, idx = 800, s=2):
    results_df = pd.DataFrame(clf.cv_results_)
    results_df.drop(columns=['std_fit_time', 'mean_score_time', 'std_score_time', 'params', 
                         'std_test_score']).sort_values('rank_test_score').head(10)
    
    f, ax = plt.subplots(2, 3, figsize=(20,8))
    ax = ax.reshape(-1)
    ax[0].scatter(np.arange(len(results_df)), results_df['param_max_depth'], s=s)
    ax[0].scatter(np.arange(len(results_df))[idx], results_df['param_max_depth'][idx], s=20, c='g')
    ax[0].set_title('max_depth vs iterations')
    ax[1].scatter(np.arange(len(results_df)), np.log10(np.array(results_df['param_learning_rate'].values, dtype='float')), s=s)
    ax[1].scatter(np.arange(len(results_df))[idx], np.log10(np.array(results_df['param_learning_rate'].values, dtype='float'))[idx], s=20, c='g')
    ax[1].set_title('learning_rate vs iterations')
    ax[2].scatter(np.arange(len(results_df)), results_df['param_min_child_samples'].values,s=s)
    ax[2].scatter(np.arange(len(results_df))[idx], results_df['param_min_child_samples'].values[idx],s=20, c='g')
    ax[2].set_title('min_child_sample vs iterations')
    ax[3].scatter(np.arange(len(results_df)), np.log10(np.array(results_df['param_reg_lambda'].values, dtype='float')),s=s)
    ax[3].scatter(np.arange(len(results_df))[idx], np.log10(np.array(results_df['param_reg_lambda'].values, dtype='float'))[idx],s=20,  c='g')
    ax[3].set_title('reg_lambda vs iterations')
    ax[4].scatter(np.arange(len(results_df)), np.log10(np.array(results_df['param_min_child_weight'].values, dtype='float')), s=s)
    ax[4].scatter(np.arange(len(results_df))[idx], np.log10(np.array(results_df['param_min_child_weight'].values, dtype='float'))[idx], s=20, c='g')
    ax[4].set_title('min_child_weight vs iterations')
    ax[5].scatter(np.log10(np.array(results_df['param_learning_rate'].values, dtype='float')), results_df['mean_fit_time'].values,  s=s)
    ax[5].scatter(np.log10(np.array(results_df['param_learning_rate'].values, dtype='float'))[idx], results_df['mean_fit_time'].values[idx],  s=20, c='g')
    ax[5].set_title('fit time vs learning_rate')
    return results_df