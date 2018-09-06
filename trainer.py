import cv2
import os.path

import constants
import kmeans
import drDill
import classifiers.histdiff as histdiff
import imgdirreader

TRAINING_DATA_FILENAME = "cv-classify.train"

def train_classifier(training_dir, output_file = "", validate = False):
    #surf = cv2.SURF(hessian, upright=0)
    sift = cv2.SIFT()
    kps, descs = extract_kps_descs(training_dir, sift)

    if descs is None or len(descs) == 0:
        return

    print str(len(descs)) + " set(s) of descriptors found" 
        
    if len(descs) == 1:
        # Edge case: if we only have one set of training descriptor, we can't do LOOCV
        print "Only 1 training descriptor found; LOOCV can not be performed"
        validate = False
            
    if validate:
        # Try different values of k (number of clusters) and do LOOCV on the training dir (using hist algorithm)
        ks = [25, 50, 75, 100]
        k_results = {}
        best_k = ks[0]
        best_error_rate = None
        for k in ks:
            print "Starting kmeans clustering loocv with k=" + str(k)
            error_rate = do_loocv(descs, k)
            if error_rate is None:
                print "Error rate: N/A\n"
            else:    
                print "Error rate: %.5f" % error_rate + "\n"
            k_results[k] = error_rate
            if error_rate is not None and (best_error_rate is None or best_error_rate > error_rate):
                best_k = k
                best_error_rate = error_rate
                
        print "CV Results on cluster size: " 
        print " k   Error Rate"
        for k in ks:
            error_rate = k_results[k]
            if error_rate is None:
                result = "SKIPPED"
            else:
                result = "%.2f" % error_rate
            print "%3s" % k + ": " + result + ("     <-- BEST" if k == best_k else "")
        print "\n"
        # "Refit" a new clustering and centroids with the best k value
        print "Reclustering using k=" + str(best_k) + " as the best number of clusters" 
        threshold, centroids, hist = find_best_clustering(descs, best_k)
    else:
        # Fix k but try to find the best clustering based on the average distance 
        best_k = 25
        threshold, centroids, hist = find_best_clustering(descs, best_k)
        print "Using %.5f" % threshold + " as the hist-diff classification threshold value"
    
    if output_file == "":
        output_file = os.path.join(training_dir, TRAINING_DATA_FILENAME)
    print "Training done! Saving training data to " + output_file
    drDill.save_training_data(kps, descs, threshold, centroids, hist, output_file)
    
def extract_kps_descs(dir, sift):
    files_to_proc = imgdirreader.get_image_files(dir)
    if len(files_to_proc) == 0:
        print "Found no image files in directory: " + dir
        return [], []

    all_kps = []
    all_descs = []
    for full_path in files_to_proc:
        print "Processing " + full_path
        img = cv2.imread(full_path, 0)
        ## Rescale prior to feature extraction
        img = cv2.resize(img, (constants.RESCALE_WIDTH, constants.RESCALE_HEIGHT))
        
        kp, des = sift.detectAndCompute(img, None)
        
        if des is None:
            print "No descriptors found! Skipping..."
            continue
        
        all_kps.extend([kp])
        all_descs.extend([des])
        
        #if output_dir != "":
        #    if not os.path.exists(output_dir):
        #        os.makedirs(output_dir)
            ## NOTE: we don't really need this for the classification but it's just something cool(ish) to look at
        #    img_kp = cv2.drawKeypoints(img, kp, None, (0,0,255), 4)
        #    _, file = os.path.split(full_path)
        #    if not cv2.imwrite(os.path.join(output_dir, file), img_kp):
        #        print "Unable to save " + os.path.join(output_dir, file) + "!"
    
    return all_kps, all_descs
    
def do_loocv(descs, k):
    correct = 0
    ndescs = len(descs)
    for i in range(0,ndescs):
        # Leave this set of descriptors out of the list of all descriptors
        descs_loocv = descs[0:i] + descs[i+1:ndescs]
    
        # Flatten the list of descriptors
        descs_flat_list = [item for sublist in descs_loocv for item in sublist]
        if k > len(descs_flat_list):
            print "Cluster size " + str(k) + " is greater than the number of descriptors to cluster; skipping..."
            return None
        # Build the histogram
        centroids, hist = histdiff.extract_centroids_histogram(descs_flat_list, k)
        # See if the resulting histogram correctly classifies the set of descriptors that was left out
        if histdiff.classify("", descs[i], 0.3, centroids, hist):
            correct+=1
    # After going through all the descriptors, return the classification error rate
    return (1-(float(correct)/ndescs))

def find_best_clustering(descs, k):
    best_centroids = None
    best_hist = None
    best_distance = None
    ndescs = len(descs)
    descs_flat_list = [item for sublist in descs for item in sublist]
    for i in range(0,5):
        # Try one possible clustering
        centroids, hist = histdiff.extract_centroids_histogram(descs_flat_list, k)
        # Compute the difference of each image from this clustering
        total_diff = 0
        for d in descs:
            h = histdiff.compute_histogram(d, centroids, len(hist))
            total_diff += histdiff.get_hist_difference(h, hist)
        average_distance = total_diff/ndescs
        print str(i+1) + "st iteration: average distance is %.5f" % average_distance
        if best_distance is None or best_distance > average_distance:
            best_distance = average_distance
            best_centroids = centroids
            best_hist = hist
        
    return (best_distance*1.10), best_centroids, best_hist
