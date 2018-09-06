import numpy as np
import random
        
## k-means clustering implementation (Lloyd's algorithm) c/o the Data Science Lab (http://datasciencelab.wordpress.com/2013/12/12/clustering-with-k-means-in-python/)

def cluster_points(X, mu):
    clusters  = {}
    for x in X:
        bestmukey = min([(i[0], np.linalg.norm(x-mu[i[0]])) \
                    for i in enumerate(mu)], key=lambda t:t[1])[0]
        try:
            clusters[bestmukey].append(x)
        except KeyError:
            clusters[bestmukey] = [x]
    return clusters
 
def reevaluate_centers(mu, clusters):
    newmu = []
    keys = sorted(clusters.keys())
    for k in keys:
        newmu.append(np.mean(clusters[k], axis = 0))
    return newmu
 
def has_converged(mu, oldmu):
    return (set([tuple(a) for a in mu]) == set([tuple(a) for a in oldmu]))
    
def cluster(X, K):
    # Initialize to K random centers
    oldmu = random.sample(X, K)
    mu = random.sample(X, K)
    clusters = cluster_points(X, mu)
    while not has_converged(mu, oldmu):
        oldmu = mu
        # Assign all points in X to clusters
        clusters = cluster_points(X, mu)
        # Reevaluate centers
        mu = reevaluate_centers(oldmu, clusters)
    return(mu, clusters)

# Given a tuple x and a list of cluster centroids, return the index of the centroid whose cluster is closest 
# (i.e. has the least distance) from the tuple    
def find_closest_cluster(desc, centroids):
    closest_cluster = 0
    least_distance = None
    for cluster in range(len(centroids)):
        c = centroids[cluster]
        distance = np.linalg.norm(desc-c)
        if least_distance is None or distance < least_distance:
            least_distance = distance
            closest_cluster = cluster
        
    return closest_cluster
    

