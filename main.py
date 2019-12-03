from sklearn import datasets
from sklearn.cluster import KMeans
import numpy as np
import matplotlib.pyplot as plt
from sklearn import metrics
from sklearn.cluster import AgglomerativeClustering


# Step 1 - Import dataset iris
def importLoadIris():
    return datasets.load_iris()

# Step 2 - Use	K-Means	to	build 2, 3, 4,	… 10 clusters.
def kMeansClustering(datasets):
    kmeansCluster = []
    for i in range(2, 11):
        kmeansCluster.append(KMeans(n_clusters=i, random_state=2).fit(datasets.data))

    return kmeansCluster

# Step 3 - Plot	values of the within cluster distance with respect	to
# the number of	clusters
def plotWithin(clustering):
    witincluster = []
    ncluster = []
    for x in clustering:
        witincluster.append(x.inertia_)
        ncluster.append(x.n_clusters)

    plt.scatter(ncluster, witincluster)
    plt.title("Within Clustering")
    plt.show()

# Step 4 - Plot	values of the between cluster distance with respect to
# the number of clusters.
def plotBetween(clustering, datasets):
    # Gets the center of the kmeans dataset
    center = KMeans(n_clusters=1, random_state=5000).fit(datasets.data)

    witincluster = []
    betweenClustering = []
    for x in clustering:
        values, counts = np.unique(x.labels_, return_counts=True)
        distancearray = 0
        # Gets the dot between the number of elements and the similarity
        for cluster, count in zip(x.cluster_centers_, counts):
            distancearray += np.dot(count, np.square(metrics.euclidean_distances([cluster], center.cluster_centers_)))

        betweenClustering.append(distancearray)
        witincluster.append(x.n_clusters)

    plt.scatter(witincluster, betweenClustering)
    plt.title("Between Clustering")
    plt.show()


# Step 5 - Plot	values	of	the	Calinski-Herbasz	index with	respect	to
# the	number	of	clusters.
def plotCalinskiHarabasz(clustering, datasets):
    calinski = []
    ncluster = []
    for x in clustering:
        calinski.append(metrics.calinski_harabasz_score(datasets.data, x.labels_))
        ncluster.append(x.n_clusters)

    plt.scatter(ncluster, calinski)
    plt.title("Calinski Harabasz")
    plt.xlabel("Calinski Harabasz Score")
    plt.ylabel("")
    plt.show()

# Step 6
# The natural cluster arrangement is 3.
# The reason why it is 3 is because it is the peak value.
# It corresponds with three iris types in the dataset

# Step 7 - Use Hierarchical clustering to identify arrangement of the
# data-points.
def plotHierarchicalclustering(datasets):
    clustering = []
    for i in range(2, 11):
        clustering.append(AgglomerativeClustering(n_clusters=i).fit(datasets.data))
    # Plot to Calinski Harabasz
    plotCalinskiHarabasz(clustering, datasets)

# Step 8 - What	is the natural arrangement there and why?
#  The natural arrangement is 3.
#  It is 3 because it is the peak value.
#  We can see the difference between the Kmean and AgglomerativeClustering is that the value 3 is higher in Kmeans.


# Main
def main():
    datasets = importLoadIris()
    clustering = kMeansClustering(datasets)
    plotWithin(clustering)
    plotBetween(clustering, datasets)
    plotCalinskiHarabasz(clustering, datasets)
    plotHierarchicalclustering(datasets)


main()