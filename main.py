import random
import scipy.cluster.hierarchy as shc
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.cluster import AgglomerativeClustering
import matplotlib.pyplot as plt
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import DBSCAN
from sklearn.decomposition import PCA
from sklearn.preprocessing import normalize


dataset = pd.read_csv("Sales_Transactions_Dataset_Weekly.csv")
X = dataset.iloc[:, 1:813]
X.head()
scaler = StandardScaler()
scaled_X = pd.DataFrame(scaler.fit_transform(X))
scaled_X.columns = X.columns
markers = [
"o",
"v",
"^",
"<",
">",
"1",
"2",
"3",
"4",
"8",
"s",
"p",
"P",
"*",
"h",
"H",
"+",
"x",
"X",
"D",
"d",
"|",
"_" ]
def get_num_of_Clusters_kmeanspp(num_of_clusters=5):
    kmeans = KMeans(n_clusters = num_of_clusters, init = 'k-means++')
    y_kmeans = kmeans.fit_predict(scaled_X)
    for i in range(num_of_clusters):
        rgb = (random.random(), random.random(), random.random())
        plt.scatter(scaled_X[y_kmeans == i]['Normalized 1'], scaled_X[y_kmeans == i]['Normalized 2'], s = 20, c = [rgb], marker=markers[i],label = 'Cluster ')

    plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], s = 30, c = 'black', label = 'Centroids')
    plt.title('Klastry')
    plt.legend()
    plt.show()
def get_num_of_Clusters_kmeans(num_of_clusters=5):
    kmeans = KMeans(n_clusters = num_of_clusters, init = 'random')
    y_kmeans = kmeans.fit_predict(scaled_X)
    for i in range(num_of_clusters):
        rgb = (random.random(), random.random(), random.random())
        plt.scatter(scaled_X[y_kmeans == i]['Normalized 1'], scaled_X[y_kmeans == i]['Normalized 2'], s = 20, c = [rgb], marker=markers[i],label = 'Cluster ')

    plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], s = 30, c = 'black', label = 'Centroids')
    plt.title('Klastry')
    plt.legend()
    plt.show()
def dbscan():
    matrix = np.column_stack((dataset["Normalized 1"], dataset["Normalized 2"]))
    model = DBSCAN(eps=0.25, min_samples=400).fit(matrix)
    plt.scatter(matrix[:, 0], matrix[:, 1], c=model.labels_)
    plt.title('DBSCAN')
    plt.show()


def AgClustering(dataset):
    dataset = dataset.drop('Product_Code', axis=1)
    X = normalize(dataset)
    X = pd.DataFrame(X)
    pca = PCA(n_components=2)
    X_principal = pca.fit_transform(X)
    X_principal = pd.DataFrame(X_principal)
    plt.figure(figsize=(8, 8))
    plt.title('AgglomerativeClustering')
    shc.dendrogram((shc.linkage(X_principal, method='ward')))
    plt.show()

get_num_of_Clusters_kmeans()
get_num_of_Clusters_kmeanspp()
dbscan()
AgClustering(dataset)