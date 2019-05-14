import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as pl
import numpy as np
import pickle
import os
import platform
import pandas as pd
from scipy.stats import ttest_ind


def entropy(vec):
    """ Compute entropy over vector """
    # Normalization
    eps = np.finfo(float).eps
    d = vec / sum(vec + eps)
    logd = np.log2(d + eps)
    return -sum(d * logd) / np.log2(len(d))


def cohens_d(x, y):
    """ Measure Cohen's d, which is an effect size measure for paired t-test
       -> 0.2 (small), 0.5 (medium), 0.8 (large)
       Use pooled variance to deal with unequal population sizes
    References:
        [1] https://en.wikipedia.org/wiki/Pooled_variance
        [2] http://trendingsideways.com/index.php/cohens-d-formula/
    """
    nx, ny = len(x), len(y)
    pooled_variance = ((nx - 1) * np.std(x, ddof=1) ** 2 +
                       (ny - 1) * np.std(y, ddof=1) ** 2) / \
                      ((nx - 1) + (ny - 1))
    return (np.mean(x) - np.mean(y)) / np.sqrt(pooled_variance)


if __name__ == '__main__':

    fontsize = 14

    do_create_entropy_boxplot = True
    
    dir_results = os.path.join('..', "results")

    fe_labels = sorted(['bitteli', 'pymus', 'cnn-1', 'cnn-2'])
    num_fe = len(fe_labels)
    ds_labels = sorted(['wjd', 'melodia_music_genre', 'idmt_smt_monotimbral', 'idmt_smt_guitar'])
    num_ds = len(ds_labels)
    num_folds = 3

    fn_pckl = os.path.join(dir_results, 'feature_importance.pckl')
    with open(fn_pckl, 'rb') as f:
        feature_importance = pickle.load(f)

    # compute entropy over random forest feature importance values
    feat_imp_entropy = np.zeros((num_ds, num_fe, num_folds))
    for d, ds_label in enumerate(ds_labels):
        # iterate over feature extractors
        for f, fe_label in enumerate(fe_labels):
            # iterate over CV folds
            for i in range(num_folds):
                feat_imp_entropy[d, f, i] = entropy(feature_importance[(d, f, i)])

    # flatten to 2D arrays
    feat_imp_entropy = np.reshape(feat_imp_entropy, (num_fe, num_ds * num_folds))

    fe_is_cnn = np.array(['cnn' in fe_labels[i] for i in range(num_fe)])
    merge_labels = ('data-driven', 'hand-crafted')
    feat_merged = np.vstack((feat_imp_entropy[fe_is_cnn, :].flatten(),
                             feat_imp_entropy[np.logical_not(fe_is_cnn), :].flatten()))

    # collect data in dataframe
    # df = pd.DataFrame(data=feat_imp_entropy.T, columns=fe_labels)
    df = pd.DataFrame(data=feat_merged.T, columns=merge_labels)

    # (1) create entropy box plot
    if do_create_entropy_boxplot:
        pl.figure(figsize=(10, 2.5))
        fn_eps = os.path.join(dir_results, 'feature_importance_entropy.eps')
        df.boxplot(fontsize=fontsize,
                   vert=False)
        ax = pl.gca()
        ax.set_ylabel('Features', fontsize=fontsize)
        ax.set_xlabel('$H$', fontsize=fontsize)
        ax.set_title('')
        # ax.set_xticklabels(ax.get_xticklabels(), rotation=90)
        for tick in ax.xaxis.get_major_ticks():
            tick.label.set_fontsize(fontsize)
        pl.suptitle("")
        pl.tight_layout()
        pl.grid(True)
        pl.savefig(fn_eps, dpi=300)
        pl.close()

    # (2) run statistical tests to compare entropy values for CNN-based and non-CNN feature extractors
    vals_cnn = feat_imp_entropy[fe_is_cnn, :].flatten()
    vals_non_cnn = feat_imp_entropy[fe_is_cnn == False, :].flatten()

    # t-test to check significant difference in means
    t, p = ttest_ind(vals_cnn, vals_non_cnn)
    d = cohens_d(vals_cnn, vals_non_cnn)

    print('T-test: t = {}, p = {}'.format(t, p))
    print("Cohen's d = {} ".format(d))

    fn_txt = os.path.join(dir_results, 'feature_importance_entropy_tests.txt')
    with open(fn_txt, 'w+') as f:
        f.write("T-test: t = {}, p = {}\nCohen's d = {}".format(t, p, d))

    print('done :)')
