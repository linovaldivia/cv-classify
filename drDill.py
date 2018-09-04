import cPickle as pickle
import cv2

# Module for saving/loading training data using cPickle

def save_training_data(kps, descs, threshold, centroids, hist, output_file):
    # Pickle training data
    training_pickle= [pickle_training_data(kps, descs, threshold, centroids, hist)]
        
    try:
        pickle.dump(training_pickle, open(output_file, "wb"))
        print "Training data saved to " + output_file
    except IOError:
        print "Could not save to training data file " + output_file

def load_training_data(training_db):
    print "Reading from training data file " + training_db
    try:
        training_pickle = pickle.load(open(training_db, "rb" ))
    except IOError:
        print "Could not open keypoints database file " + training_db
        return 0, [], [], [], {}

    print "Loading training data"
    training_data_pickle, idx = read_and_inc(training_pickle, 0)    
    kps, descs, threshold, centroids, hist = unpickle_training_data(training_data_pickle)
    
    return kps, descs, threshold, centroids, hist    
            
def pickle_training_data(kps, descs, threshold, centroids, hist):
    pickle = []
    
    # Pickle keypoints and descriptors
    kp_descs_pickle = [] 
    for i in range(len(kps)):
        kp_arr = kps[i]
        desc = descs[i]
        kp_arr_desc = []
        for j in range(len(kp_arr)):
            # Combine keypoints data and descriptor into one object
            temp = (kp_arr[j].pt, kp_arr[j].size, kp_arr[j].angle, kp_arr[j].response, kp_arr[j].octave, kp_arr[j].class_id, desc[j])
            kp_arr_desc.append(temp)
        if len(kp_arr_desc) == 1:
            print "Can't use single keypoint-descriptor pair! Skipping..."
            continue
        print "Saving " + str(len(kp_arr_desc)) + " keypoints-descriptor pairs"
        kp_descs_pickle.append(kp_arr_desc)
    print str(len(kp_descs_pickle)) + " keypoints-descriptor pair set(s) to be saved total"
    pickle.append(kp_descs_pickle)
    
    # Pickle threshold value 
    pickle.append(threshold)
    
    # Pickle centroids
    print "Saving " + str(len(centroids)) + " cluster centroids..."
    pickle.append(centroids)
    
    # Save histogram
    pickle.append(hist)    
    return pickle    

def read_and_inc(arr, idx):
    val = arr[idx]
    return (val, idx+1)
    
def unpickle_training_data(training_data_pickle):
    kp_descs_pickle, idx = read_and_inc(training_data_pickle, 0)
    
    all_kps = []
    all_descs = []
    for kp_desc in kp_descs_pickle:
        kps = []
        descs = []
        for temp in kp_desc:
            kp = cv2.KeyPoint(x=temp[0][0],y=temp[0][1],_size=temp[1], _angle=temp[2], _response=temp[3], _octave=temp[4], _class_id=temp[5])
            desc = temp[6]
            kps.append(kp)
            descs.append(desc)
        all_kps.append(kps)
        all_descs.append(descs)
        print "Read " + str(len(kps)) + " keypoints-descriptor pairs"
            
    # Read saved threshold
    threshold, idx = read_and_inc(training_data_pickle, idx)
    print "Read %.5f" % threshold + " as the hist-diff classification threshold" 
            
    # Read saved centroids
    centroids, idx = read_and_inc(training_data_pickle, idx)
    print "Read " + str(len(centroids)) + " centroids" 
    
    # Read saved training histogram
    histogram, idx = read_and_inc(training_data_pickle, idx)
    print "Read histogram of size " + str(len(histogram)) 
    
    #for c in histogram:
    #    print "Cluster " + str(c) + ": " + str(len(histogram[c])) + " points" 
    #print "Centroids: " 
    #pp.pprint(centroids)
    return all_kps, all_descs, threshold, centroids, histogram
