import warnings
import pandas as pd
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA, FastICA, KernelPCA
from sklearn.manifold import TSNE
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.exceptions import DataConversionWarning
from lib.autoencoder import AutoEncoder
import seaborn as sns
import umap

warnings.filterwarnings(action='ignore', category=DataConversionWarning)

sns.set(style='white', rc={'figure.figsize': (
    14, 14), 'lines.markersize': 3.2})
np.random.seed(42)


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
    'RBF KPCA': KernelPCA(n_components=2, kernel='rbf'),
    'Cosine KPCA': KernelPCA(n_components=2, kernel='cosine'),
    't-SNE': TSNE(n_components=2, perplexity=5),
    'UMAP': umap.UMAP(n_neighbors=200),
    
}

dim_reductions = {
    'Auto Encoder' : AutoEncoder(54)
}

scalers = {
    'MinMax': MinMaxScaler(),
    'Z-Score': StandardScaler(),
    'None': None
}


def plot_dim_reduction(df, scaler, dim_reductions):
    lanes = ['MIDDLE', 'BOTTOM', 'NONE', 'TOP', 'JUNGLE']
    fig, axs = plt.subplots( (len(dim_reductions)+1)//2,
                             len(dim_reductions) // ((len(dim_reductions)+1)//2) )
    # if axs is an Axes element instead of a list
    try:
        axs = axs.flatten()
    except:
        axs = [axs]

    lanes_df = pd.DataFrame(df['lane'])
    df.drop(['lane', 'championId'], axis=1, inplace=True)
    df = scaleColumns(df, scaler=scalers['Z-Score'])

    for i, dim_reduction in enumerate(dim_reductions):
        # call dim reduction method
        reduced_data = dim_reductions[dim_reduction].fit_transform(df)

        for lane in lanes:
            # take indices with this specific lane
            indices = lanes_df.index[lanes_df['lane'] == lane].tolist()
            lane_data = reduced_data[indices]
            axs[i].scatter(lane_data[:, 0], lane_data[:, 1])

        #axs[i].set_title(dim_reduction)
        axs[i].legend(lanes)
        # Remove axis ticks
        axs[i].xaxis.set_major_formatter(plt.NullFormatter())
        axs[i].yaxis.set_major_formatter(plt.NullFormatter())

    fig.tight_layout()


if __name__ == '__main__':
    df = pd.read_pickle('cleaned_data.pkl')
    plot_dim_reduction(df, scalers['MinMax'], dim_reductions)
    plt.show()
