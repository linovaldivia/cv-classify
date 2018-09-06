import cv2

import kmeans

DEFAULT_K_CLUSTERS = 50

def classify(qimg_pathname, qdescs, threshold, centroids, training_hist):
    print "Computing image histogram, #bins=" + str(len(training_hist)) 
    query_hist = compute_histogram(qdescs, centroids, len(training_hist))
    
    hist_diff = get_hist_difference(query_hist, training_hist)
    print "Histogram distance (chi-squared): " + str(hist_diff)
    
    # Classify the image as a + if the histogram difference is below a certain threshold
    yes_classify = (hist_diff < threshold)
    if yes_classify and qimg_pathname != "":
        print "+++ Image " + qimg_pathname + " is a POSITIVE!"
                
    return yes_classify 
    
def compute_histogram(qdescs, centroids, nbins):
    hist = {}
    for qdesc in qdescs:
        closest_cluster = kmeans.find_closest_cluster(qdesc, centroids)
        try:
            hist[closest_cluster].append(qdesc)
        except KeyError:
            hist[closest_cluster] = [qdesc]
    
    return normalize_hist(hist, nbins)

def extract_centroids_histogram(descs, k = DEFAULT_K_CLUSTERS):
    if len(descs) == 0:
        return [], []
    print "Performing clustering on " + str(len(descs)) + " descriptors (k=" + str(k) + ")..."
    # Perform clustering to find the best grouping of the descriptors
    centroids, hist = kmeans.cluster(descs, k)
    print "Found " + str(len(centroids)) + " clusters in training descriptors "
    hist = normalize_hist(hist, k)
    return centroids, hist
    
def normalize_hist(hist, nbins):
    norm_hist = {}
    num_entries = float(sum(len(v) for v in hist.itervalues()))
    for i in range(0,nbins):
        try:
            norm_hist[i] = len(hist[i])/num_entries
        except KeyError:
            # If this bin doesn't exist, create it with a value of 0
            norm_hist[i] = 0
    return norm_hist
    
def get_hist_difference(query_hist, training_hist):
    ## Trying chi-square distance of two histograms X and Y : sum(((x_i - y_i)^2)/(x_i+y_i))) * 1/2
    total = 0
    for bin in query_hist:
        sq_diff = (query_hist[bin] - training_hist[bin])**2
        sum = (query_hist[bin] + training_hist[bin])
        #print "Cluster " + str(cluster) + ": " + str(diff)
        if sum > 0:
            total += (sq_diff/float(sum))
    return total/2
