import numpy as np
import cv2

import kmeans

# Whether or not to use the min distance brute force "good match" checking algorithm (False = Use Ratio Test)
USE_MIN_DISTANCE_BF_GOOD_MATCH_ALG = False
# (When USE_MIN_DISTANCE_BF_GOOD_MATCH_ALG = True) The minimum distance between a query descriptor and the nearest training descriptor in order to be considered a "good match"
GOOD_MATCH_MIN_DISTANCE = 150.0
# (When USE_MIN_DISTANCE_BF_GOOD_MATCH_ALG = False) The minimum ratio between the two nearest training descriptors to a query descriptor in order to be considered a "good match"
GOOD_MATCH_MIN_DISTANCE_RATIO = 0.8

# Parameters used in brute force classification
# Minimum ratio of "good matches" between a query image's descriptors and a training image's descriptors to be considered a "hit". 
# If the query image "hits" at least some x% of the training image data, a positive classification is concluded.
GOOD_MATCH_TOTAL_MIN_RATIO = 0.15
# If the ratio of "good matches" (i.e. x% of query image descriptors that are a "good match" to a training image descriptors) is considered high enough, 
# the image *might* still be positively classified (given that some other criteria is met).
GOOD_MATCH_HIGH_RATIO = 0.7

def classify(qimg_pathname, qdescs, all_descs, bf, min_num_hits):
    num_hits = 0
    has_high_match_ratio = False
    
    if  USE_MIN_DISTANCE_BF_GOOD_MATCH_ALG:
        print "Using minimum distance good match algorithm (brute force)"
    else:
        print "Using ratio test good match algorithm (brute force)"

    for i, trdescs in enumerate(all_descs):
        print "Matching with training " + str(i + 1) + " descriptors (" + str(len(trdescs)) + " total)" 
        ## Find the best k matches between training descriptor and query descriptor
        ## NOTE: need to do np.asarray() in order for the function to work -- maybe a python version issue?
        ## Each "descriptor" is actually an array of 128 float32 values -- it's SIFT/SURF's numerical representation of a feature
        ## The keypoint structure contains metadata about the feature -- it's x,y location, octave, angle, etc.
        ## If we treat each descriptor as a 128-d vector, knnMatch will then attempt to find the k nearest descriptor vectors
        ## Note that qdesc is a set of descriptors, with each descriptor represented by 128 values 
        ## For each descriptor q in qdesc, bf.knnMatch will attempt to find the k nearest neighbors of q in trdesc 
        matches = bf.knnMatch(np.asarray(qdescs, np.float32),
                              np.asarray(trdescs, np.float32),k=2)
                              
        ## What is the structure of this "matches" list? 
        ## Each element in the "matches" list is another list of k (possibly less) "match" objects. 
        ## The "matches" list will have len(qdescs) elements, i.e. there is an element for each descriptor in the query image. 
        ## Each element in the sublist ("matches[i][j]") contains queryIdx, trainIdx, imgIdx, and distance (between the descriptor qdesc[queryIdx] and trdesc[trainIdx])
        ## Since k=2, m and n will be two match objects for the same query image descriptor (i.e. m.queryIdx == n.queryIdx) but with two different
        ## training image descriptors (i.e. m.trainIdx != n.trainIdx, representing the k=2 nearest neighbors to the query image descriptor)
        
        # Now find out how many of the extracted query descriptors were a "good match" to the training descriptors.
        good_matches = 0
        for m,n in matches:
            if is_good_match(m, n):
                good_matches += 1
                
        if good_matches == 0:
            print "  0 descriptors matched, skipping..."
            continue
            
        # How many query descriptors were "good matches"?
        match_ratio = good_matches / float(len(qdescs))
        msg = str(good_matches) + "/" + str(len(qdescs)) + " (%.2f" % (match_ratio * 100) + "%) query descriptors matched!"
        if match_ratio >= GOOD_MATCH_HIGH_RATIO:
           has_high_match_ratio = True
           
        if match_ratio >= GOOD_MATCH_TOTAL_MIN_RATIO:
            msg = "+ " + msg
        else:
            msg = "  " + msg
        print msg
           
        # Consider a positive classification if there are enough hits
        if match_ratio >= GOOD_MATCH_TOTAL_MIN_RATIO:
            num_hits+=1
            if num_hits == min_num_hits:
                print "+++ Image matched " + str(num_hits) + " in training data; " + qimg_pathname + " is a POSITIVE"
                return True

    # Consider a positive classification if the query image had a high match ratio and has at least one other hit
    if has_high_match_ratio and num_hits > 1:
        print "+++ Image strongly matched at least 1 image in training data; " + qimg_pathname + " is a POSITIVE"
        return True
    else:
        print "--- Image matched " + str(num_hits) + " in training data; " + qimg_pathname + " is a NEGATIVE"
    return False

def is_good_match(match1, match2):
    # print "match1.queryIdx = " + str(match1.queryIdx) + " match1.trainIdx = " + str(match1.trainIdx) + " match1.distance = " + str(match1.distance)
    # print "match2.queryIdx = " + str(match2.queryIdx) + " match2.trainIdx = " + str(match2.trainIdx) + " match2.distance = " + str(match2.distance)
    # What constitutes a "good match"?
    # New alg: "good match" means that the distance of the query descriptor from the first nearest training descriptor < threshold
    if USE_MIN_DISTANCE_BF_GOOD_MATCH_ALG:
        if match1.distance < GOOD_MATCH_MIN_DISTANCE:
            return True
    # Old alg (ratio test): "good match" means that the distance between the k=2 nearest training descriptors is less than a threshold (as mentioned in DLowe SIFT paper)
    elif match1.distance/match2.distance < GOOD_MATCH_MIN_DISTANCE_RATIO:
        return True
    return False