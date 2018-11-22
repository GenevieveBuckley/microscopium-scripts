"""Perform dimensionality reduction on Vi's data."""
import numpy as np
import pandas as pd
from sklearn.preprocessing import RobustScaler
from sklearn.manifold import TSNE as tSNE
from sklearn.decomposition import PCA
from umap import UMAP


data = pd.read_csv('data.csv', index_col=0, header=0)
data.head()
X = data.drop(columns=['filenames']).values
Xs = RobustScaler().fit_transform(X)
Xt = tSNE().fit_transform(Xs)  # this takes a long time, maybe 30min
Xp = PCA(2).fit_transform(Xs)
Xu = umap.UMAP().fit_transform(Xs)
manifolds = np.concatenate((Xp, Xt, Xu), axis=1)
manifold_cols = ['x.pca', 'y.pca', 'x.tsne', 'y.tsne', 'x.umap', 'y.umap']
manifolds_table = pd.DataFrame(manifolds, columns=manifold_cols)
manifolds_table.index = data.index
manifolds_table['url'] = data['filenames']
info = pd.read_csv('info.csv', header=0, index_col=0)
info2 = info.join(manifolds_table)
info2.to_csv('info2.csv')
