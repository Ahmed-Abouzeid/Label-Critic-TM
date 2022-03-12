from sklearn.cluster import DBSCAN
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import silhouette_score
import scipy.cluster.hierarchy as shc
from sklearn.cluster import SpectralClustering
from sklearn.cluster import KMeans
import numpy as np
import random
from sklearn.model_selection import train_test_split
from my_datasets.data import Synthetic
from matplotlib import pyplot as plt

y_kmeans = []
y_dbscan = []
y_agglomerative = []
y_spectral = []
y_kmeans_std = []
y_dbscan_std = []
y_agglomerative_std = []
y_spectral_std = []

y_tm = [1, 1, 1, 1, 1]
y_tm_std = [0, 0, 0, 0, 0]
for i in [2, 4, 6, 8, 10]:
    km = []
    db = []
    agg = []
    sp = []
    for r in range(10):
        #### K MEANS
        syn = Synthetic(300, 300, 100, i)
        (x, _, y, _), all_patterns = syn.load_data()
        kmeans = KMeans()
        kmeans.fit(x)
        labels = kmeans.fit_predict(x)
        score = silhouette_score(x, labels, metric='euclidean')
        print('KMEANS SIL SCORE:', score)
        km.append(score)

        ## DBScan
        dbscan = DBSCAN().fit(x)  # fitting the model
        labels = dbscan.labels_  # getting the labels
        score = silhouette_score(x, labels, metric='euclidean')
        print('DBSCAN SIL SCORE:', score)
        db.append(score)
        # AgglomerativeClustering

        cluster = AgglomerativeClustering()
        labels = cluster.fit_predict(x)
        score = silhouette_score(x, labels, metric='euclidean')
        print('Agglomerative SIL SCORE:', score)
        agg.append(score)

        # spectral
        clustering = SpectralClustering(assign_labels='discretize', random_state=0).fit(x)
        labels = clustering.labels_
        score = silhouette_score(x, labels, metric='euclidean')
        print('SPECTRAL SIL SCORE:', score)
        sp.append(score)
    y_kmeans.append(np.mean(km))
    y_dbscan.append(np.mean(db))
    y_spectral.append(np.mean(sp))
    y_agglomerative.append(np.mean(agg))

    y_kmeans_std.append(np.std(km))
    y_dbscan_std.append(np.std(db))
    y_spectral_std.append(np.std(sp))
    y_agglomerative_std.append(np.std(agg))

## plots
x_values = ['2', '4', '6', '8', '10']

plt.plot(x_values, y_kmeans, label='Kmeans')
plt.fill_between(x_values, np.array(y_kmeans) - np.array(y_kmeans_std), np.array(y_kmeans) + np.array(y_kmeans_std),
                 color='gray', alpha=0.2)
plt.plot(x_values, y_dbscan, label='DBSCAN')
plt.fill_between(x_values, np.array(y_dbscan) - np.array(y_dbscan_std), np.array(y_dbscan) + np.array(y_dbscan_std),
                 color='gray', alpha=0.2)
plt.plot(x_values, y_agglomerative, label='Agglomerative')
plt.fill_between(x_values, np.array(y_agglomerative) - np.array(y_agglomerative_std), np.array(y_agglomerative) +
                 np.array(y_agglomerative_std),
                 color='gray', alpha=0.2)
plt.plot(x_values, y_spectral, label='Spectral')
plt.fill_between(x_values, np.array(y_spectral) - np.array(y_spectral_std), np.array(y_spectral) + np.array(y_spectral_std),
                 color='gray', alpha=0.2)
plt.plot(x_values, y_tm, label='TM')
plt.fill_between(x_values, np.array(y_tm) - np.array(y_tm_std), np.array(y_tm) + np.array(y_tm_std),
                 color='gray', alpha=0.2)

plt.scatter(x_values, y_kmeans, marker='s')
plt.scatter(x_values, y_dbscan, marker='^')
plt.scatter(x_values, y_agglomerative, marker='o')
plt.scatter(x_values, y_spectral, marker='>')
plt.scatter(x_values, y_tm, marker='<')

plt.xlabel("Number of Unique Sub-patterns in Data")
plt.ylabel("SIL Score")
plt.yticks([0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
plt.grid(axis='y')
plt.legend()
plt.savefig('eval.png')