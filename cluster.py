import warnings
import pandas as pd
import numpy as np
import argparse
from copy import deepcopy
# set random seed
RANDOM_SEED=42
np.random.seed(RANDOM_SEED)
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from sklearn.cluster import KMeans, DBSCAN
from sklearn.mixture import GaussianMixture
from sklearn.decomposition import PCA, FastICA, KernelPCA
from sklearn.manifold import TSNE
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.exceptions import DataConversionWarning
from sklearn.metrics import silhouette_score, silhouette_samples, fowlkes_mallows_score, v_measure_score
from sklearn.base import clone
import seaborn as sns
import umap
from lib.autoencoder import AutoEncoder
from lib.fcm import FuzzyCMeans

# supress Tensorflow info
import logging
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # FATAL
logging.getLogger('tensorflow').setLevel(logging.FATAL)

is_pretty_table = True
try:
    from prettytable import PrettyTable
except:
    print ('preetytable not found, drawing tables in \"ugly\" way')
    is_pretty_table = False

warnings.filterwarnings(action='ignore', category=DataConversionWarning)

sns.set(style='white', rc={'figure.figsize': (
    14, 14), 'lines.markersize': 3.2})


def scaleColumns(df, cols_to_scale=None, scaler=MinMaxScaler()):
    if cols_to_scale is None:
        cols_to_scale = df.columns
    for col in cols_to_scale:
        df[col] = pd.DataFrame(scaler.fit_transform(
            pd.DataFrame(df[col])), columns=[col])
    return df


dim_reductions = {
    'PCA': PCA(n_components=2),
    'ICA': FastICA(n_components=2),
    't-SNE': TSNE(n_components=2, perplexity=5),
    'UMAP': umap.UMAP(n_neighbors=200),
    'Auto Encoder' : AutoEncoder(54, 2)
}

AE = AutoEncoder(54, 2)

scalers = {
    'MinMax': MinMaxScaler(),
    'Z-Score': StandardScaler(),
    'None': None
}

clusterers = {
    'KMeans': KMeans(n_clusters=6, n_jobs=-1),
    'GMM': GaussianMixture(n_components=6, covariance_type='full'),
    'FCM' : FuzzyCMeans(n_clusters=6)
    #'DBSCAN': DBSCAN(eps=0.5, min_samples=25, n_jobs=-1)
}
params = {
    'KMeans' : {
        'n_clusters' : [5, 6],
        #'random_state' : [RANDOM_SEED]
    },
    'GMM' : {
        'n_components' : [5, 6],
        'covariance_type' : ['full', 'diag', 'tied', 'spherical'],
        'tol' : [5e-3, 1e-4, 5e-4],
        'max_iter' : [100, 150, 200],
        #'random_state' : [RANDOM_SEED]
    },
    'FCM' : {
        'n_clusters' : [5, 6],
        'max_iter' : [100, 150, 200],
        'error' : [1e-3, 5e-3, 1e-4],
        'm' : [1.1, 1.2, 1.3, 1.4],
        #'random_state' : [RANDOM_SEED]
    }

}


def plot_clusters(df, scaler, dim_reductions, clusterers, ae=None):
    """plots clustering algorithms on data
    
    Arguments:
        df {DataFrame} -- data frame to plot
        scaler {Scaler} -- Scaler for the data
        dim_reductions {Dict[String][DimensionReduction]} -- dict of dimension reduction methods
        clusterers {Dict[String][Clusterer]} -- dict of clustering methods
    """
    global AE
    fig, axs = plt.subplots(len(clusterers), len(dim_reductions))
    #axs = axs.flatten()
    lanes_df = df['lane']
    # indices of elements who are from MID, TOP or JUNGLE
    known_label_indices = [i for i in range(len(lanes_df)) if lanes_df[i] in ['MIDDLE', 'TOP', 'JUNGLE']]
    true_labels = lanes_df[known_label_indices]
    data = df.drop(['lane', 'championId'], axis=1)
    data = scaleColumns(data, scaler=scaler)
    scaled_data = data.copy()
    
    if ae is not None:
        if AE.is_fit is True and AE.mid_size == ae:
            data = AE.predict(scaled_data)
        else:
            AE = AutoEncoder(54, ae)
            data = AE.fit_transform(scaled_data)

    for i, clusterer_name in enumerate(clusterers):
        clusterer = clusterers[clusterer_name]
        labels = clusterer.predict(data)

        for j, dim_reduction_name in enumerate(dim_reductions):
            reduced_data = dim_reductions[dim_reduction_name].fit_transform(scaled_data)

            unique_values = set(labels)
            # for each cluster
            for unique_value in unique_values:
                indices = [k for k in range(len(labels)) if  labels[k] == unique_value]
                cluster_data = reduced_data[indices]
                axs[i][j].scatter(cluster_data[:, 0], cluster_data[:, 1], label=str(unique_value))

            axs[i][j].set_title(f'{dim_reduction_name}, {clusterer_name}')
            axs[i][j].xaxis.set_major_formatter(plt.NullFormatter())
            axs[i][j].yaxis.set_major_formatter(plt.NullFormatter())

    fig.tight_layout()

def comparison_table(df, clusterers, scaler, ae_dims=[]):
    global AE
    table = []
    if is_pretty_table:
        table = PrettyTable(['Method', 'Mean silhouette score', 'V Score', 'Fowlkes Score'])
    lanes_df = df['lane']
    # indices of elements who are from MID, TOP or JUNGLE
    known_label_indices = [i for i in range(len(lanes_df)) if lanes_df[i] in ['MIDDLE', 'TOP', 'JUNGLE']]
    true_labels = lanes_df[known_label_indices]

    data_no_labels = df.drop(['lane', 'championId'], axis=1)
    data = scaleColumns(data_no_labels, scaler=scaler)
    scaled_data = data.copy()
    if not ae_dims:
        for clusterer_name in clusterers:
            clusterer = clusterers[clusterer_name]
            labels = clusterer.predict(data)

            sil_score = silhouette_score(data, labels)
            # labels for elements from MID TOP or JUNGLE
            known_lane_labels = labels[known_label_indices]
            v_score = v_measure_score(true_labels, known_lane_labels)
            fowlkes_score = fowlkes_mallows_score(true_labels, known_lane_labels)

            if is_pretty_table:
                table.add_row([f'{clusterer_name}', 
                                    "{:.3f}".format(sil_score), 
                                    "{:.3f}".format(v_score), 
                                    "{:.3f}".format(fowlkes_score)])
            else:
                print(f'{clusterer_name}: {sil_score}, {v_score}, {fowlkes_score}')
    else:
        for ae_dim in ae_dims:
            if AE.is_fit is True and AE.mid_size == ae_dim:
                pass
            else:
                AE = AutoEncoder(54, ae_dim)

            data = AE.fit_transform(data) 

            for clusterer_name in clusterers:
                clusterer = clusterers[clusterer_name]
                labels = clusterer.predict(data)

                sil_score = silhouette_score(scaled_data, labels)
                # labels for elements from MID TOP or JUNGLE
                known_lane_labels = labels[known_label_indices]
                v_score = v_measure_score(true_labels, known_lane_labels)
                fowlkes_score = fowlkes_mallows_score(true_labels, known_lane_labels)
                if is_pretty_table:
                    table.add_row([f'AE {ae_dim}-{clusterer_name}', 
                                    "{:.3f}".format(sil_score), 
                                    "{:.3f}".format(v_score), 
                                    "{:.3f}".format(fowlkes_score)])
                else:
                    print(f'AE {ae_dim}-{clusterer_name}: {sil_score:.3f}, {v_score:.3f}, {fowlkes_score:.3f}')

    if is_pretty_table:
        print(table)


def gridsearch_params(df, scaler, clusterer, params, ae=None):
    lanes_df = df['lane']
    # indices of elements who are from MID, TOP or JUNGLE
    known_label_indices = [i for i in range(len(lanes_df)) if lanes_df[i] in ['MIDDLE', 'TOP', 'JUNGLE']]
    true_labels = lanes_df[known_label_indices]
    data_no_labels = df.drop(['lane', 'championId'], axis=1)
    data = scaleColumns(data_no_labels, scaler=scaler)
    scaled_data = data.copy()
    
    if ae is not None:
        AE = dim_reductions['Auto Encoder']
        if AE.is_fit is True and AE.mid_size == ae:
            data = AE.predict(data)
        else:
            dim_reductions['Auto Encoder'] = AutoEncoder(54, ae)
            AE = dim_reductions['Auto Encoder']
            data = AE.fit_transform(data)

    def make_generator(parameters):
        """Generates all parameter combinations
        
        Arguments:
            parameters {param dict} -- parameters for model
        
        Yields:
            Tuple -- parameter combination
        """
        if not parameters:
            yield dict()
        else:
            key_to_iterate = list(parameters.keys())[0]
            next_round_parameters = {p : parameters[p]
                        for p in parameters if p != key_to_iterate}
            for val in parameters[key_to_iterate]:
                for pars in make_generator(next_round_parameters):
                    temp_res = pars
                    temp_res[key_to_iterate] = val
                    yield temp_res
    
    best_estimator = None
    best_param = None
    best_score = 0
    best_sil = -1
    best_v = 0
    for param in make_generator(params):
        ca = clusterer.set_params(**param)
        labels = ca.fit_predict(data)
        known_lane_labels = labels[known_label_indices]
        fowlkes_score = fowlkes_mallows_score(true_labels, known_lane_labels)
        sil_score = silhouette_score(scaled_data, labels)
        v_score = v_measure_score(true_labels, known_lane_labels)
        if fowlkes_score > best_score:
            best_param = param
            best_estimator = deepcopy(ca)
            best_score = fowlkes_score
            best_sil = sil_score
            best_v = v_score
    
    return (best_param, best_score, best_sil, best_v),  best_estimator

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-p', '--path', dest='path', help='Path to data file', default='cleaned_data.pkl')
    args = parser.parse_args()
    df = pd.read_pickle(args.path)
    ############## NO AE ###################
    for clusterer_name in clusterers:
        param, estimator = gridsearch_params(df, scalers['Z-Score'], clusterers[clusterer_name], params[clusterer_name])
        print(f'No AE: {clusterer_name} : {param}')
        clusterers[clusterer_name] = estimator
    comparison_table(df, clusterers, scalers['Z-Score'])
    plot_clusters(df, scalers['Z-Score'], dim_reductions, clusterers)
    ############## AE 5 ###################
    for clusterer_name in clusterers:
        param, estimator = gridsearch_params(df, scalers['Z-Score'], clusterers[clusterer_name], params[clusterer_name], ae=5)
        print(f'AE5: {clusterer_name} : {param}')
        clusterers[clusterer_name] = estimator

    plot_clusters(df, scalers['Z-Score'], dim_reductions, clusterers, ae=5)
    comparison_table(df, clusterers, scalers['Z-Score'], ae_dims=[5])
    ############## AE 10 ###################
    for clusterer_name in clusterers:
        param, estimator = gridsearch_params(df, scalers['Z-Score'], clusterers[clusterer_name], params[clusterer_name], ae=10)
        print(f'AE10: {clusterer_name} : {param}')
        clusterers[clusterer_name] = estimator

    plot_clusters(df, scalers['Z-Score'], dim_reductions, clusterers, ae=10)
    comparison_table(df, clusterers, scalers['Z-Score'], ae_dims=[10])
    plt.show()

