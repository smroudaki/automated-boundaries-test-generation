__version__ = '5'
__author__ = 'Akram Kalaee'

import numpy as np
from sklearn.cluster import DBSCAN
import hdbscan
from config import clustering_config as clustering_config, genetic_algorithm_config as GA_config, \
    global_config as global_config, test_config

def clustering_test_data_hdbscan(df):
    try:
        min_samples = clustering_config['min_samples']
        eps = clustering_config['eps']
        print('df: ', df)
        df = np.matrix(df)
        print('df matrix: ', df)
        clusterer = hdbscan.HDBSCAN(allow_single_cluster=True)
        # clusterer = hdbscan.HDBSCAN(min_cluster_size=40,min_samples= 1, allow_single_cluster=True).fit(df)
        print('clusterer: ', clusterer)
        clusterer.fit(df)

        labels = clusterer.labels_
        n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)  # clusterer.labels_.max()
        # n_clusters_ = 10
        unique_labels = set(labels)

        clusters = {}
        for i, k in enumerate(unique_labels):
            class_member_mask = (labels == k)
            members = df[class_member_mask]
            clusters[i] = members
        print('clusters: ', clusters)

        domains = []

        for cluster_i in clusters.keys():
            # points = [(point[idx] for idx in range(len(point))) for point in np.array(clusters[cluster_i])]
            points = []
            for point in np.array(clusters[cluster_i]):
                points.append([point[idx] for idx in range(len(point))])
            domains.append(points)

        print('domains: ', domains)
        print('n_clusters_: ', n_clusters_)
        return n_clusters_, domains
    except:

        print('Test data are not sufficient! Retry again or modify the configurations.')
        exit(0)
